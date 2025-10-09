# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:56:43 2025

@author: hungv
"""

# gnn_speed_bench.py
# -*- coding: utf-8 -*-
"""
Benchmark inference speed for trained GNN members on an unseen realization.
Also writes ensemble summary and optional GNN-vs-CMG speedup comparison.

Usage (terminal):
  python gnn_speed_bench.py \
    --results_dir compare91_GNN_ensemble \
    --unseen_dir "ccs-data analytic/data/model-49/csv" \
    --k_layer 1 \
    --threads 4 --warmup 1 \
    --cmg_csv "cmg_times_model49.csv"   # optional

CMG CSV formats (auto-detected):
 A) Per-snapshot:
    year,cmg_ms_per_snapshot
    2032,850
    2041,870
 B) Total:
    year,cmg_total_seconds,timesteps
    2032,540,12
    2041,560,12

Outputs:
- gnn_speed_bench_results.csv   (per member)
- gnn_speed_bench_summary.csv   (ensemble averages)
- gnn_vs_cmg.csv                (if --cmg_csv provided)
"""

import os, re, time, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# -------------------- helpers --------------------
def _now(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()

def _elapsed_ms(t0, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3

def set_deterministic():
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clamp01(x):
    return np.minimum(1.0, np.maximum(0.0, x))

def safe_load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)  # PyTorch >= 2.4
    except TypeError:
        return torch.load(path, map_location=device)                     # Older PyTorch

# -------------------- model --------------------
class ResGATBlock(nn.Module):
    def __init__(self, in_ch, out_ch, heads=4, dropout=0.15):
        super().__init__()
        self.gat1 = GATConv(in_ch, out_ch, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(out_ch*heads, out_ch, heads=heads, concat=True, dropout=dropout)
        self.res_proj = nn.Linear(in_ch, out_ch*heads) if in_ch != out_ch*heads else nn.Identity()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x0 = self.res_proj(x)
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index) + x0)
        return x

class DeepResGNN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden=128, heads=4):
        super().__init__()
        self.fc_in = nn.Linear(in_feats, hidden)
        self.block1 = ResGATBlock(hidden, hidden, heads=heads)
        self.block2 = ResGATBlock(hidden*heads, hidden, heads=heads)
        self.block3 = ResGATBlock(hidden*heads, hidden, heads=heads)
        self.block4 = ResGATBlock(hidden*heads, hidden, heads=heads)
        self.fc_out = nn.Linear(hidden*heads, out_feats)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, edge_index):
        x = F.elu(self.fc_in(x))
        x = self.dropout(x)
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        x = self.block4(x, edge_index)
        x = self.fc_out(x)
        return x

# -------------------- data utils --------------------
YEAR_IN_NAME = re.compile(r'(?<!\d)(?:19|20|21)\d{2}(?!\d)')
def extract_year(filename: str) -> int:
    hits = YEAR_IN_NAME.findall(os.path.basename(filename))
    return int(hits[-1]) if hits else -1

def group_files_by_year(folder):
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    year2files = {}
    for f in csv_files:
        y = extract_year(f)
        if y <= 0:
            continue
        year2files.setdefault(y, []).append(os.path.join(folder, f))
    return dict(sorted(year2files.items(), key=lambda kv: kv[0]))

def read_csv_std(path):
    import pandas as pd, re as _re
    df = pd.read_csv(path)
    def _pick(df_, pats):
        for p in pats:
            rc = _re.compile(p, _re.I)
            for c in df_.columns:
                if rc.search(c):
                    return c
        return None
    i = _pick(df, [r'^i_index$', r'^i$', r'\bI[-_ ]?index\b'])
    j = _pick(df, [r'^j_index$', r'^j$', r'\bJ[-_ ]?index\b'])
    k = _pick(df, [r'^k_index$', r'^k$', r'\bK[-_ ]?index\b'])
    if i is None or j is None or k is None:
        raise KeyError("Missing i/j/k index columns in CSV")
    df = df.rename(columns={i: 'i', j: 'j', k: 'k'})
    drop_cols = [c for c in df.columns if _re.search(r'\b(x_coord|y_coord|z_coord|Time)\b', c, _re.I)]
    return df.drop(columns=drop_cols, errors='ignore')

def merge_year_files(paths):
    import pandas as pd
    combo, seen = None, set()
    for p in paths:
        d = read_csv_std(p)
        keep = ['i', 'j', 'k'] + [c for c in d.columns if c not in ('i', 'j', 'k') and c not in seen]
        d = d[keep]
        seen.update([c for c in keep if c not in ('i', 'j', 'k')])
        combo = d if combo is None else combo.merge(d, on=['i', 'j', 'k'], how='outer')
    return combo.loc[:, ~combo.columns.duplicated()]

def build_edge_index(NI, NJ, diag=False):
    edges = []
    for i in range(NI):
        for j in range(NJ):
            u = i * NJ + j
            if j < NJ - 1:
                edges += [(u, u + 1), (u + 1, u)]
            if i < NI - 1:
                edges += [(u, u + NJ), (u + NJ, u)]
            if diag and i < NI - 1 and j < NJ - 1:
                edges += [(u, u + NJ + 1), (u + NJ + 1, u), (u + 1, u + NJ), (u + NJ, u + 1)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# -------------------- main benchmark --------------------
def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="RESULTS_DIR containing real_* subfolders")
    ap.add_argument("--unseen_dir", required=True, help="Directory with unseen CSVs (year-coded)")
    ap.add_argument("--k_layer", type=int, default=1)
    ap.add_argument("--use_gpu", action="store_true", help="Enable CUDA if available")
    ap.add_argument("--warmup", type=int, default=1, help="Number of warm-up forward passes per snapshot")
    ap.add_argument("--diag_edges", action="store_true", help="Use 8-neighbour connectivity")
    ap.add_argument("--threads", type=int, default=None, help="Set torch.set_num_threads for CPU")
    ap.add_argument("--cmg_csv", type=str, default=None, help="Optional CMG timings CSV (per-snapshot or total format)")
    args = ap.parse_args()

    set_deterministic()
    if args.threads is not None:
        torch.set_num_threads(int(args.threads))

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    # collect members
    members = []
    for d in sorted([d for d in os.listdir(args.results_dir) if d.startswith("real_")]):
        base = os.path.join(args.results_dir, d)
        pth = os.path.join(base, "best_GNN_model.pth")
        scaler = os.path.join(base, "scaler.npz")
        tgt = os.path.join(base, "targets.json")
        if os.path.exists(pth) and os.path.exists(scaler) and os.path.exists(tgt):
            with open(tgt, "r") as f:
                targets = json.load(f)
            members.append({"name": d, "pth": pth, "scaler": scaler, "targets": targets})
    if not members:
        raise RuntimeError("No trained members found under results_dir")

    # unseen input build
    y2files = group_files_by_year(args.unseen_dir)
    years = [y for y in sorted(y2files) if 2030 <= y <= 2130]
    if not years:
        raise RuntimeError("No usable years in unseen_dir")
    df0 = merge_year_files(y2files[years[0]])
    NI, NJ = int(df0['i'].max()), int(df0['j'].max())

    # input columns (fallbacks allowed)
    def pick(df, pats):
        import re as _re
        for p in pats:
            rc = _re.compile(p, _re.I)
            for c in df.columns:
                if rc.search(c):
                    return c
        return None

    INP = {
        "Porosity": [r'Porosity'],
        "PermI": [r'Permeability[\s_]*I\b.*', r'Permeability(?!.*Relative)'],
        "PermK": [r'Permeability[\s_]*K\b.*', r'Permeability(?!.*Relative)'],
        "NetPay": [r'Net[\s_]*Pay', r'Net[-\s_]*Thickness', r'Grid[\s_]*Thickness'],
        "NPV": [r'(Net|Total)?\s*Pore\s*Volume', r'\bNPV\b'],
        "Pressure": [r'^Pressure', r'\bPRES\b'],
        "GBV": [r'Gross[\s_]*Block[\s_]*Volume']
    }

    def grid_prop(df, col, mask):
        g = np.full((NI, NJ), 0.0)
        if col is None:
            return g
        sub = df[mask]
        for _, r in sub.iterrows():
            g[int(r['i']) - 1, int(r['j']) - 1] = r[col]
        return g

    base_inputs = []
    for y in years:
        dfy = merge_year_files(y2files[y])
        mask = (dfy['k'] == args.k_layer)
        poro = grid_prop(dfy, pick(dfy, INP["Porosity"]), mask)
        permi = grid_prop(dfy, pick(dfy, INP["PermI"]), mask)
        permk = grid_prop(dfy, pick(dfy, INP["PermK"]), mask)
        permi = np.log1p(np.abs(permi)) * np.sign(permi)
        permk = np.log1p(np.abs(permk)) * np.sign(permk)
        netpay = grid_prop(dfy, pick(dfy, INP["NetPay"]), mask)
        gbv = pick(dfy, INP["GBV"])
        npv = grid_prop(dfy, pick(dfy, INP["NPV"]), mask) if pick(dfy, INP["NPV"]) else (
              grid_prop(dfy, gbv, mask) * poro if gbv else np.zeros_like(poro))
        pres = grid_prop(dfy, pick(dfy, INP["Pressure"]), mask)
        base_inputs.append(np.stack([poro, permi, permk, netpay, npv, pres], axis=-1))
    base_inputs = np.stack(base_inputs)  # [T,NI,NJ,6]

    # Edge index
    edge_index = build_edge_index(NI, NJ, diag=args.diag_edges).to(device)

    # For each member, time full rollout
    rows = []
    for m in members:
        sc = np.load(m["scaler"])
        in_feats = base_inputs.shape[-1] + 2  # +year +prev
        out_feats = len(m["targets"])

        net = DeepResGNN(in_feats=in_feats, out_feats=out_feats, hidden=128, heads=4).to(device)
        sd = safe_load_state_dict(m["pth"], device)
        net.load_state_dict(sd)
        net.eval()

        # scale inputs using this member's TRAIN-only scaler
        Xmin, Xmax = sc["X_min"], sc["X_max"]
        base_scaled = (base_inputs - Xmin) / (Xmax - Xmin + 1e-8)

        y0, y1 = float(sc["year_min"]), float(sc["year_max"])
        year_norm = (np.array(years, dtype=float) - y0) / (y1 - y0 + 1e-8)
        year_norm = clamp01(year_norm)[:, None, None, None]
        year_chan = np.repeat(year_norm, NI, axis=1).repeat(NJ, axis=2)

        prev = np.zeros((NI, NJ, 1), dtype=np.float32)
        T = base_scaled.shape[0]

        # warmup (optional)
        with torch.no_grad():
            for _ in range(args.warmup):
                for t in range(T):
                    xt = np.concatenate([base_scaled[t], year_chan[t], prev], axis=-1)
                    xnodes = torch.tensor(xt.reshape(-1, xt.shape[-1]), dtype=torch.float32, device=device)
                    _ = net(xnodes, edge_index)

        # timed pass
        t0 = _now(device)
        with torch.no_grad():
            for t in range(T):
                xt = np.concatenate([base_scaled[t], year_chan[t], prev], axis=-1)
                xnodes = torch.tensor(xt.reshape(-1, xt.shape[-1]), dtype=torch.float32, device=device)
                out = net(xnodes, edge_index)
                try:
                    gas_idx = m["targets"].index("Gas_Saturation")
                    prev = out[:, gas_idx:gas_idx+1].reshape(NI, NJ, 1).detach().cpu().numpy()
                except ValueError:
                    prev = np.zeros_like(prev)
        total_ms = _elapsed_ms(t0, device)

        ms_per_snapshot = total_ms / T
        us_per_cell = (ms_per_snapshot * 1e3) / (NI * NJ)
        mcells_per_sec = ((NI * NJ) / (ms_per_snapshot / 1e3)) / 1e6

        rows.append({
            "Member": m["name"],
            "Device": device.type,
            "Timesteps": int(T),
            "NI": int(NI),
            "NJ": int(NJ),
            "Nodes": int(NI * NJ),
            "Total_ms": total_ms,
            "ms_per_snapshot": ms_per_snapshot,
            "us_per_cell": us_per_cell,
            "Mcells_per_sec": mcells_per_sec
        })

    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["Member"]).reset_index(drop=True)
    df.to_csv("gnn_speed_bench_results.csv", index=False)

    # --------------- Ensemble summary ---------------
    summary = {
        "Device": device.type,
        "Members": int(df["Member"].nunique()),
        "Timesteps": int(df["Timesteps"].iloc[0]),
        "NI": int(df["NI"].iloc[0]),
        "NJ": int(df["NJ"].iloc[0]),
        "Nodes": int(df["Nodes"].iloc[0]),
        "avg_ms_per_snapshot": float(df["ms_per_snapshot"].mean()),
        "median_ms_per_snapshot": float(df["ms_per_snapshot"].median()),
        "std_ms_per_snapshot": float(df["ms_per_snapshot"].std(ddof=0)),
        "avg_us_per_cell": float(df["us_per_cell"].mean()),
        "avg_Mcells_per_sec": float(df["Mcells_per_sec"].mean()),
        "total_ms_all_members": float(df["Total_ms"].sum())
    }
    pd.DataFrame([summary]).to_csv("gnn_speed_bench_summary.csv", index=False)

    print("\n=== Per-member results ===")
    print(df)
    print("\n=== Ensemble summary ===")
    print(pd.DataFrame([summary]))

    # --------------- Optional: GNN vs CMG ---------------
    if args.cmg_csv is not None and os.path.exists(args.cmg_csv):
        cmg = pd.read_csv(args.cmg_csv)
        avg_ms = summary["avg_ms_per_snapshot"]
        members = summary["Members"]

        out = None
        if "cmg_ms_per_snapshot" in cmg.columns:
            cmg["gnn_ms_per_snapshot"] = avg_ms
            cmg["speedup_x"] = cmg["cmg_ms_per_snapshot"] / cmg["gnn_ms_per_snapshot"]
            out = cmg[["year", "cmg_ms_per_snapshot", "gnn_ms_per_snapshot", "speedup_x"]]
        elif set(["cmg_total_seconds", "timesteps"]).issubset(cmg.columns):
            cmg["gnn_total_seconds"] = (avg_ms / 1000.0) * cmg["timesteps"] * members
            cmg["speedup_x"] = cmg["cmg_total_seconds"] / cmg["gnn_total_seconds"]
            cols = ["year", "cmg_total_seconds", "gnn_total_seconds", "timesteps", "speedup_x"]
            out = cmg[[c for c in cols if c in cmg.columns]]
        else:
            print("\n[warn] --cmg_csv provided but columns not recognized; expected either "
                  "[year, cmg_ms_per_snapshot] or [year, cmg_total_seconds, timesteps].")

        if out is not None:
            out.to_csv("gnn_vs_cmg.csv", index=False)
            print("\n=== GNN vs CMG ===")
            print(out)
            print("\nSaved: gnn_vs_cmg.csv")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:  # Spyder-friendly defaults
        sys.argv += [
            "--results_dir", r"C:\Users\hungv\Downloads\compare91_GNN_ensemble",
            "--unseen_dir",  r"C:\Users\hungv\Downloads\ccs-data analytic\data\model-49\csv",
            "--k_layer", "1",
            "--threads", "4",
            "--warmup", "1",
            # "--use_gpu",            # uncomment to benchmark on CUDA if available
            # "--cmg_csv", r"C:\Users\hungv\Downloads\compare91_GNN_ensemble\cmg_times_model49.csv",
        ]
    main()
