# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:22:47 2025

@author: hungv
"""

# =========================================================
# CPU-ONLY: Train 45-model GNN ensemble, hold 3 unseen.
# Upgrades:
#   - Temporal train/val/test split; scalers fit on TRAIN only
#   - Val-based early stopping + ReduceLROnPlateau
#   - Ensemble P10/P50/P90
#   - Unseen outputs: PNGs with colorbars (legends) + 1x5 GIFs (Mean, Std, P10, P50, P90)
#   - Optional 8-neighbour graph
#   - Safer I/O + compressed arrays + run config snapshot
# =========================================================
import os, re, time, warnings, json, math, hashlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

# ---------------------------
# SETTINGS
# ---------------------------
DATA_ROOT = r'ccs-data analytic/data'  # folder containing model-*/csv
TRAIN_COUNT  = 45
UNSEEN_COUNT = 3

RESULTS_DIR = 'compare91_GNN_ensemble'          # per-realization training results
PUB_DIR     = 'publication91_GNN_ensemble'      # per-realization publication figs
UNSEEN_OUT_DIR = 'inference_unseen_out3'        # unseen ensemble outputs (PNGs + GIFs)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PUB_DIR, exist_ok=True)
os.makedirs(UNSEEN_OUT_DIR, exist_ok=True)

k_layer = 1
USE_PREV_STATE = True
USE_DIAGONAL_EDGES = False      # set True to use 8-neighbour connectivity
pub_years = [2032, 2041, 2055, 2060, 2070, 2130]

# ---- CPU-friendly defaults ----
EPOCHS       = 300
PATIENCE     = 60
LR           = 3e-4
WEIGHT_DECAY = 1e-40.0
SEED_BASE    = 1337
SAVE_ANIMATIONS_TRAIN = False   # training animations (true vs pred vs err)
MAKE_ANIMS_UNSEEN     = True    # unseen inference 1x5 mean/std/p10/p50/p90

# ---- FORCE CPU ONLY ----
FORCE_CPU = True
def pick_device(force_cpu=False):
    if force_cpu:
        print("[device] Forcing CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = pick_device(force_cpu=FORCE_CPU)
print("[device] Using:", DEVICE)

# Repro
def set_seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# SAFE CSV on Windows (locks)
# ---------------------------
def safe_to_csv(df, path, index=False, max_tries=4, retry_delay=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)
    for attempt in range(max_tries):
        tmp_path = f"{base}.tmp.{os.getpid()}.{attempt}{ext}"
        try:
            df.to_csv(tmp_path, index=index)
            try:
                os.replace(tmp_path, path)
                return path
            except PermissionError:
                alt_path = f"{base}_{time.strftime('%Y%m%d-%H%M%S')}_{attempt}{ext}"
                os.replace(tmp_path, alt_path)
                print(f"[warn] '{path}' locked. Wrote '{alt_path}' instead.")
                return alt_path
        except PermissionError:
            time.sleep(retry_delay)
    final_path = f"{base}_{time.strftime('%Y%m%d-%H%M%S')}_final{ext}"
    df.to_csv(final_path, index=index)
    print(f"[warn] Could not write '{path}'. Saved '{final_path}'")
    return final_path

# ---------------------------
# YEAR detection
# ---------------------------
YEAR_IN_NAME = re.compile(r'(?<!\d)(?:19|20|21)\d{2}(?!\d)')  # standalone 4-digit year
def extract_year(filename: str) -> int:
    hits = YEAR_IN_NAME.findall(os.path.basename(filename))
    return int(hits[-1]) if hits else -1

# ---------------------------
# DISCOVERY (model-*/csv with year-coded CSVs)
# ---------------------------
def _has_year_csv(dir_path: str) -> bool:
    try:
        for f in os.listdir(dir_path):
            if f.lower().endswith(".csv") and YEAR_IN_NAME.search(f):
                return True
    except FileNotFoundError:
        pass
    return False

def _numeric_tail(path):
    base = os.path.basename(path.rstrip(os.sep))
    m = re.search(r'(\d+)$', base)
    return int(m.group(1)) if m else 10**9

def discover_realization_folders(root):
    out = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p): continue
        csv_dir = os.path.join(p, "csv")
        if os.path.isdir(csv_dir) and _has_year_csv(csv_dir):
            out.append(csv_dir)
        elif _has_year_csv(p):
            out.append(p)
    if not out and _has_year_csv(root):
        out = [root]
    out.sort(key=lambda d: _numeric_tail(os.path.dirname(d) if d.endswith(os.sep+'csv') else d))
    return out

# -------------------------------------
# FLEXIBLE COLUMN RESOLUTION + FALLBACKS
# -------------------------------------
from difflib import get_close_matches

INPUT_PATTERNS = {
    'Porosity':       [r'Porosity.*(Current|Base|Initial)?', r'\bPorosity\b'],
    'PermI':          [r'Permeability[\s_]*I\b.*', r'\bPerm(?!.*Relative)[^_]*I\b.*'],
    'PermK':          [r'Permeability[\s_]*K\b.*', r'\bPerm(?!.*Relative)[^_]*K\b.*'],
    'PermAny':        [r'(?i)\bPermeability\b(?!.*Relative).*'],  # generic
    'NetPay':         [r'\bNet[\s_]*Pay\b.*', r'\bNet[-\s_]*Thickness\b.*', r'\bGrid[\s_]*Thickness\b.*'],
    'NetPoreVolume':  [r'\b(Net|Total)?[\s_]*Pore[\s_]*Volume\b.*', r'\bNPV\b.*'],
    'Pressure':       [r'^Pressure\b.*', r'\bPRES\b.*', r'\bPressure\(.*\)\b.*'],
    'GrossBlockVolume': [r'\bGross[\s_]*Block[\s_]*Volume\b.*'],
}

OUTPUT_PATTERNS = {
    'Gas_Saturation': [r'\bGas[\s_]*Saturation\b.*', r'^(Sg|SGAS)\b.*', r'\bGas[-_ ]?Sat\b.*'],
    'pH':             [r'(^|\b)pH\b.*'],
    'Subsidence':     [r'\bSubsidence\b.*', r'VDISPL.*Z.*', r'\bSurface.*(Z|Vertical).*displacement.*'],
}

INDEX_PATTERNS = {
    'i': [r'^i_index$', r'^i$', r'\bI[-_ ]?index\b'],
    'j': [r'^j_index$', r'^j$', r'\bJ[-_ ]?index\b'],
    'k': [r'^k_index$', r'^k$', r'\bK[-_ ]?index\b'],
}

def _regex_find(cols, patterns):
    for pat in patterns:
        rc = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rc.search(c):
                return c
    return None

def _find_col(df, patterns, required=True, name_for_msg=''):
    col = _regex_find(list(df.columns), patterns)
    if col is None and required:
        candidates = get_close_matches(name_for_msg, list(df.columns), n=8, cutoff=0.3)
        raise KeyError(
            f"Could not find a column for '{name_for_msg}'. Tried: {patterns}\n"
            f"Closest headers: {candidates}\n"
            f"First 30 headers: {list(df.columns)[:30]}"
        )
    return col

def _find_index_cols(df):
    i_col = _find_col(df, INDEX_PATTERNS['i'], True, 'i_index')
    j_col = _find_col(df, INDEX_PATTERNS['j'], True, 'j_index')
    k_col = _find_col(df, INDEX_PATTERNS['k'], True, 'k_index')
    return i_col, j_col, k_col

def resolve_columns_from_df(df, require_outputs=False):
    """Resolve inputs/optional inputs/outputs using regex with permeability fallbacks."""
    i_col, j_col, k_col = _find_index_cols(df)

    poro   = _find_col(df, INPUT_PATTERNS['Porosity'], True, 'Porosity')
    permi  = _find_col(df, INPUT_PATTERNS['PermI'],   False, 'Permeability_I')
    permk  = _find_col(df, INPUT_PATTERNS['PermK'],   False, 'Permeability_K')
    permAny= _find_col(df, INPUT_PATTERNS['PermAny'], False, 'Permeability (non-relative)')

    if permi is None and permAny is not None: permi = permAny
    if permk is None and permAny is not None: permk = permAny

    netpay = _find_col(df, INPUT_PATTERNS['NetPay'],        False, 'Net_Pay')
    npv    = _find_col(df, INPUT_PATTERNS['NetPoreVolume'], False, 'Net_Pore_Volume')
    pres   = _find_col(df, INPUT_PATTERNS['Pressure'],      False, 'Pressure')
    gbv    = _find_col(df, INPUT_PATTERNS['GrossBlockVolume'], False, 'Gross_Block_Volume')

    gas = _find_col(df, OUTPUT_PATTERNS['Gas_Saturation'], False, 'Gas_Saturation')
    ph  = _find_col(df, OUTPUT_PATTERNS['pH'],              False, 'pH')
    sub = _find_col(df, OUTPUT_PATTERNS['Subsidence'],      False, 'Subsidence')

    outputs_map = {k: v for k, v in [('Gas_Saturation', gas), ('pH', ph), ('Subsidence', sub)] if v is not None}
    if require_outputs and not outputs_map:
        raise KeyError("None of the expected outputs (Gas_Saturation, pH, Subsidence) were found.")

    return {
        'i': i_col, 'j': j_col, 'k': k_col,
        'inputs': {'Porosity': poro, 'PermI': permi, 'PermK': permk,
                   'NetPay': netpay, 'NetPoreVolume': npv, 'Pressure': pres, 'GBV': gbv},
        'outputs_map': outputs_map,
    }

# -----------------------
# PRESCAN
# -----------------------
REQUESTED_TARGETS = ['Gas_Saturation', 'pH', 'Subsidence']

def detect_outputs_in_cols(cols):
    found = set()
    for name, pats in OUTPUT_PATTERNS.items():
        if _regex_find(cols, pats): found.add(name)
    return found

def prescan_outputs(folder):
    cols_union = set()
    for f in os.listdir(folder):
        if not f.lower().endswith('.csv'): continue
        p = os.path.join(folder, f)
        try:
            df_head = pd.read_csv(p, nrows=0)
            cols_union.update(list(df_head.columns))
        except Exception:
            pass
    return detect_outputs_in_cols(list(cols_union))

# -----------------------
# MERGE ALL CSVs PER YEAR
# -----------------------
def group_files_by_year(folder):
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    year2files = {}
    for f in csv_files:
        y = extract_year(f)
        if y <= 0: continue
        year2files.setdefault(y, []).append(os.path.join(folder, f))
    return dict(sorted(year2files.items(), key=lambda kv: kv[0]))

def read_and_standardize(path):
    df = pd.read_csv(path)
    i_col, j_col, k_col = _find_index_cols(df)
    df = df.rename(columns={i_col: 'i', j_col: 'j', k_col: 'k'})
    drop_cols = [c for c in df.columns if re.search(r'\b(x_coord|y_coord|z_coord|Time)\b', c, re.I)]
    return df.drop(columns=drop_cols, errors='ignore')

def merge_year_files(paths):
    combo = None
    have = set()
    for p in paths:
        d = read_and_standardize(p)
        keep_cols = ['i','j','k'] + [c for c in d.columns if c not in ('i','j','k') and c not in have]
        d = d[keep_cols]
        have.update([c for c in keep_cols if c not in ('i','j','k')])
        combo = d if combo is None else combo.merge(d, on=['i','j','k'], how='outer')
    return combo.loc[:, ~combo.columns.duplicated()]

# -----------------------
# Determine common valid years across all training realizations
# -----------------------
def valid_years_for_targets(folder, targets):
    y2files = group_files_by_year(folder)
    years = [y for y in sorted(y2files) if 2030 <= y <= 2130]
    ok = []
    for y in years:
        try:
            d = merge_year_files(y2files[y]).rename(columns={'i':'i_index','j':'j_index','k':'k_index'})
            resolved = resolve_columns_from_df(d, require_outputs=False)
            out_map = resolved['outputs_map']
            missing = [t for t in targets if t not in out_map]
            if not missing:
                ok.append(y)
        except Exception:
            pass
    return ok

# -----------------------
# LOAD / PREPROCESS (per realization) — return RAW (unscaled)
# -----------------------
def load_series_from_folder(folder, k_layer, targets, force_years=None):
    """Return raw inputs/outputs; scaling is done later on train-only."""
    year2files = group_files_by_year(folder)
    if not year2files:
        raise ValueError(f"No time-stamped CSV files found in {folder}")

    years_all = sorted(list(year2files.keys()))
    years_sorted = [y for y in years_all if (2030 <= y <= 2130) and (force_years is None or y in set(force_years))]

    print(f"[discover] {folder} -> years used: {years_sorted[:10]}{'...' if len(years_sorted)>10 else ''}")
    if not years_sorted:
        raise ValueError(f"No usable years for {folder} after filtering.")

    X_series, Y_series, all_years = [], [], []
    N_I = N_J = None

    for y in years_sorted:
        df_y = merge_year_files(year2files[y]).rename(columns={'i':'i_index', 'j':'j_index', 'k':'k_index'})
        resolved = resolve_columns_from_df(df_y, require_outputs=False)
        i_col, j_col, k_col = resolved['i'], resolved['j'], resolved['k']
        in_cols = resolved['inputs']; out_map = resolved['outputs_map']

        missing = [t for t in targets if t not in out_map]
        if missing:
            print(f"  [skip year {y}] missing targets: {missing}")
            continue

        if N_I is None:
            N_I = int(df_y[i_col].max()); N_J = int(df_y[j_col].max())
        else:
            Ni, Nj = int(df_y[i_col].max()), int(df_y[j_col].max())
            if (Ni, Nj) != (N_I, N_J):
                raise ValueError(f"Grid size changed in {folder} year {y}: {(Ni,Nj)} vs {(N_I,N_J)}")

        def grid_property(df, prop, mask):
            grid = np.full((N_I, N_J), np.nan)
            for _, row in df[mask].iterrows():
                i = int(row[i_col]) - 1
                j = int(row[j_col]) - 1
                grid[i, j] = row[prop]
            return grid

        mask = (df_y[k_col] == k_layer)

        poro_grid  = grid_property(df_y, in_cols['Porosity'], mask)

        def perm_grid_or_zero(col_name):
            if col_name is None:
                return np.zeros_like(poro_grid)
            g = grid_property(df_y, col_name, mask)
            return np.log1p(np.abs(g)) * np.sign(g)

        permI_grid = perm_grid_or_zero(in_cols['PermI'])
        permK_grid = perm_grid_or_zero(in_cols['PermK'])

        input_channels = [poro_grid, permI_grid, permK_grid]

        if in_cols['NetPay'] is not None:
            netpay_grid = grid_property(df_y, in_cols['NetPay'], mask)
        else:
            netpay_grid = np.zeros_like(poro_grid)
        input_channels.append(netpay_grid)

        if in_cols['NetPoreVolume'] is not None:
            npv_grid = grid_property(df_y, in_cols['NetPoreVolume'], mask)
        else:
            if in_cols['GBV'] is not None:
                gbv_grid = grid_property(df_y, in_cols['GBV'], mask)
                npv_grid = poro_grid * gbv_grid
            else:
                npv_grid = np.zeros_like(poro_grid)
        input_channels.append(npv_grid)

        if in_cols['Pressure'] is not None:
            pres_grid = grid_property(df_y, in_cols['Pressure'], mask)
        else:
            pres_grid = np.zeros_like(poro_grid)
        input_channels.append(pres_grid)

        output_channels = [grid_property(df_y, out_map[t], mask) for t in targets]

        X_series.append(np.stack(input_channels, axis=-1))
        Y_series.append(np.stack(output_channels, axis=-1))
        all_years.append(y)

    if len(X_series) == 0:
        raise ValueError(f"After merging/filtering, no usable years remained in {folder} for targets {targets}")

    X = np.stack(X_series)
    Y = np.stack(Y_series)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    all_years = np.array(all_years)

    return {
        'X_raw': X, 'Y_raw': Y, 'years': all_years,
        'N_I': X.shape[1], 'N_J': X.shape[2]
    }

def build_edge_index(NI, NJ, diag=False):
    edges = []
    for i in range(NI):
        for j in range(NJ):
            u = i*NJ + j
            if j < NJ-1: edges += [(u,u+1),(u+1,u)]
            if i < NI-1: edges += [(u,u+NJ),(u+NJ,u)]
            if diag and i < NI-1 and j < NJ-1:
                edges += [(u, u+NJ+1),(u+NJ+1,u),(u+1, u+NJ),(u+NJ, u+1)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def make_data_loader(X_arr, Y_arr, edge_index, batch_size=1, shuffle=True):
    data_list = []
    for t in range(X_arr.shape[0]):
        X_step = X_arr[t].reshape(-1, X_arr.shape[-1])
        Y_step = Y_arr[t].reshape(-1, Y_arr.shape[-1])
        data_list.append(Data(
            x=torch.tensor(X_step, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(Y_step, dtype=torch.float32)
        ))
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

# -------------
# GNN
# -------------
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

# -----------------
# TRAIN / EVAL utils
# -----------------
def train_epoch(model, loader, optimizer, device):
    model.train(); total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch.x, batch.edge_index)
        loss = F.smooth_l1_loss(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); total = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index)
        loss = F.smooth_l1_loss(pred, batch.y)
        total += loss.item()
    return total / max(1, len(loader))

def plot_train_val_test_loss(loss_dict, name, outdir):
    plt.figure(figsize=(7,4))
    plt.plot(loss_dict['train'], label="Train")
    plt.plot(loss_dict['val'],   label="Val")
    if loss_dict.get('test'):
        plt.plot(loss_dict['test'],  label="Test")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss Curve: {name}")
    plt.legend(); plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}_loss_curve.png"), dpi=140)
    plt.close()

def publication_triple_map(true_map, pred_map, err_map, target, year, vmin=None, vmax=None, err_abs=None, savepath=None):
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    ax = axs[0]
    im0 = ax.imshow(np.flipud(true_map.T), cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'CMG {target}\n{year}'); ax.axis('off')
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.04)
    plt.colorbar(im0, cax=cax).set_label('Value')
    ax = axs[1]
    im1 = ax.imshow(np.flipud(pred_map.T), cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'GNN {target}\n{year}'); ax.axis('off')
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.04)
    plt.colorbar(im1, cax=cax).set_label('Value')
    ax = axs[2]
    im2 = ax.imshow(np.flipud(err_map.T), cmap='coolwarm', vmin=-err_abs, vmax=err_abs)
    ax.set_title(f'Error\n{year}'); ax.axis('off')
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.04)
    plt.colorbar(im2, cax=cax).set_label('Error')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

# ---------- Unseen: 1x5 animation (Mean | Std | P10 | P50 | P90) with colorbars ----------
def animate_five_panel_with_cbs(mean_stack, std_stack, p10_stack, p50_stack, p90_stack,
                                years, NI, NJ, target, out_dir):
    if not MAKE_ANIMS_UNSEEN: return
    os.makedirs(out_dir, exist_ok=True)

    vmin_mean = float(np.nanmin([np.nanmin(mean_stack), np.nanmin(p10_stack),
                                 np.nanmin(p50_stack), np.nanmin(p90_stack)]))
    vmax_mean = float(np.nanmax([np.nanmax(mean_stack), np.nanmax(p10_stack),
                                 np.nanmax(p50_stack), np.nanmax(p90_stack)]))
    vmin_std  = 0.0
    vmax_std  = float(np.nanmax(std_stack))

    fig, axs = plt.subplots(1, 5, figsize=(18,4), gridspec_kw={'wspace':0.15})
    ims = []
    titles = ['Mean','Std','P10','P50','P90']
    cmaps = ['viridis','magma','viridis','viridis','viridis']
    vmins = [vmin_mean, vmin_std, vmin_mean, vmin_mean, vmin_mean]
    vmaxs = [vmax_mean, vmax_std, vmax_mean, vmax_mean, vmax_mean]
    stacks= [mean_stack, std_stack, p10_stack, p50_stack, p90_stack]

    colorbars = []
    for ax, st, ttl, cmap, vmin, vmax in zip(axs, stacks, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(np.flipud(st[0].reshape(NI,NJ).T), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(ttl); ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbars.append(cb)
        ims.append(im)

    supt = fig.suptitle(f"{target} — {years[0]}")

    def update(i):
        ims[0].set_data(np.flipud(mean_stack[i].reshape(NI,NJ).T))
        ims[1].set_data(np.flipud(std_stack [i].reshape(NI,NJ).T))
        ims[2].set_data(np.flipud(p10_stack [i].reshape(NI,NJ).T))
        ims[3].set_data(np.flipud(p50_stack [i].reshape(NI,NJ).T))
        ims[4].set_data(np.flipud(p90_stack [i].reshape(NI,NJ).T))
        supt.set_text(f"{target} — {years[i]}")
        return tuple(ims) + (supt,)

    anim = FuncAnimation(fig, update, frames=len(years), interval=800)
    out_path = os.path.join(out_dir, f"{target}_mean_std_p10_p50_p90.gif")
    try:
        anim.save(out_path, writer=PillowWriter(fps=1))
    finally:
        plt.close(fig)

# ---------- PNG saver with colorbar (legend) ----------
def save_png_with_colorbar(stack, stat_label, target, years, NI, NJ, out_dir,
                           cmap='viridis', vmin=None, vmax=None):
    os.makedirs(out_dir, exist_ok=True)
    if vmin is None: vmin = float(np.nanmin(stack))
    if vmax is None: vmax = float(np.nanmax(stack))
    for ti, y in enumerate(years):
        m = stack[ti].reshape(NI, NJ)
        fig, ax = plt.subplots(figsize=(4.6, 5.2))
        im = ax.imshow(np.flipud(m.T), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{target} — {stat_label} — {y}")
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_ylabel('Value' if cmap!='coolwarm' else 'Error')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{target}_{stat_label}_{y}.png"),
                    dpi=170, bbox_inches='tight')
        plt.close(fig)

def unscale_channel(y_scaled, ch, Y_min, Y_max):
    return y_scaled * (Y_max[0,0,0,ch] - Y_min[0,0,0,ch]) + Y_min[0,0,0,ch]

def clamp01(x): return np.minimum(1.0, np.maximum(0.0, x))

# ==========================
# TRAIN 45, HOLD OUT 3
# ==========================
set_seed_all(SEED_BASE)
all_folders = discover_realization_folders(DATA_ROOT)
if not all_folders:
    raise RuntimeError(f"No realizations found under {DATA_ROOT}")

train_folders = all_folders[:TRAIN_COUNT]
unseen_folders = all_folders[TRAIN_COUNT:TRAIN_COUNT+UNSEEN_COUNT]
print(f"Training on {len(train_folders)} realizations:")
for r in train_folders: print("  -", r)
print(f"Held-out unseen ({len(unseen_folders)}):")
for r in unseen_folders: print("  -", r)

# Determine common targets across training realizations
outputs_per_real = [prescan_outputs(r) for r in train_folders]
common = set(REQUESTED_TARGETS)
for s in outputs_per_real: common &= s

if not common:
    from collections import Counter
    flat = [t for s in outputs_per_real for t in s if t in REQUESTED_TARGETS]
    if not flat:
        raise RuntimeError("None of the requested targets found in any training realization.")
    top_target, _ = Counter(flat).most_common(1)[0]
    keep_idx = [i for i,s in enumerate(outputs_per_real) if top_target in s]
    train_folders = [train_folders[i] for i in keep_idx]
    TARGETS = [top_target]
    print(f"\nNOTE: Proceeding with most frequent target: {top_target}")
    print(f"Kept {len(train_folders)} training realizations.")
else:
    TARGETS = [t for t in REQUESTED_TARGETS if t in common]
    print("\nTargets across training realizations:", TARGETS)

# --- intersection of valid years across training realizations ---
valid_sets = []
for rf in train_folders:
    vy = valid_years_for_targets(rf, TARGETS)
    valid_sets.append(set(vy))
common_years_sorted = sorted(list(set.intersection(*valid_sets))) if valid_sets else []
if not common_years_sorted:
    raise RuntimeError("No common years across training realizations for the chosen targets.")
print("[years] Using common years across all training realizations:",
      common_years_sorted[:10], "..." if len(common_years_sorted)>10 else "")

# First training realization defines grid & years (forced to intersection)
first_pack = load_series_from_folder(train_folders[0], k_layer, TARGETS, force_years=common_years_sorted)
N_I0, N_J0 = first_pack['N_I'], first_pack['N_J']
edge_index = build_edge_index(N_I0, N_J0, diag=USE_DIAGONAL_EDGES)
years_reference = np.array(common_years_sorted, dtype=int)

all_metrics_rows = []
summary_times = []

# -------- helper: augment with year + prev-state channels --------
def augment_with_year_prev(X_scaled, Y_scaled, years, use_prev_state=True, year_min=None, year_max=None):
    """Append normalized year and prev-state(channel 0 target) to X."""
    if year_min is None: year_min = years.min()
    if year_max is None: year_max = years.max()
    years_norm = (years - year_min) / (year_max - year_min + 1e-8)
    X_aug = []
    for i in range(len(years)):
        year_channel = np.full((X_scaled.shape[1], X_scaled.shape[2], 1), years_norm[i])
        X_aug.append(np.concatenate([X_scaled[i], year_channel], axis=-1))
    X_scaled = np.stack(X_aug)
    if use_prev_state:
        X_prev = []
        for i in range(len(X_scaled)):
            prev = np.zeros((X_scaled.shape[1], X_scaled.shape[2], 1)) if i == 0 else Y_scaled[i-1][:,:,0:1]
            X_prev.append(np.concatenate([X_scaled[i], prev], axis=-1))
        X_scaled = np.stack(X_prev)
    return X_scaled, (float(year_min), float(year_max))

for ridx, rfolder in enumerate(train_folders, start=1):
    set_seed_all(SEED_BASE + ridx)
    rname = f"real_{ridx:02d}"
    print(f"\n=== Training {rname}: {rfolder}")

    pack = load_series_from_folder(rfolder, k_layer, TARGETS, force_years=years_reference)
    X_raw, Y_raw, all_years = pack['X_raw'], pack['Y_raw'], pack['years']
    N_I, N_J = pack['N_I'], pack['N_J']

    if (N_I, N_J) != (N_I0, N_J0):
        raise ValueError(f"Grid size mismatch in {rfolder}: {(N_I, N_J)} vs {(N_I0, N_J0)}")
    if len(all_years) != len(years_reference) or not np.all(all_years == years_reference):
        raise ValueError(f"Years timeline mismatch in {rfolder} after forcing intersection.")

    # ------- temporal split -------
    n_total = X_raw.shape[0]
    n_test  = max(2, int(0.15*n_total))
    n_val   = max(2, int(0.15*n_total))
    train_end = n_total - n_val - n_test
    val_end   = n_total - n_test
    if train_end < 1:
        raise RuntimeError("Too few timesteps to form train/val/test splits.")

    # Fit scalers on TRAIN ONLY
    def fit_minmax(arr):
        mn = np.nanmin(arr[:train_end], axis=(0,1,2), keepdims=True)
        mx = np.nanmax(arr[:train_end], axis=(0,1,2), keepdims=True)
        return mn, mx

    X_min, X_max = fit_minmax(X_raw)
    Y_min, Y_max = fit_minmax(Y_raw)

    X_scaled = (np.nan_to_num(X_raw) - X_min) / (X_max - X_min + 1e-8)
    Y_scaled = (np.nan_to_num(Y_raw) - Y_min) / (Y_max - Y_min + 1e-8)

    # Augment with year + prev (prev uses truth)
    X_scaled_aug, (year_min_fit, year_max_fit) = augment_with_year_prev(
        X_scaled, Y_scaled, all_years, use_prev_state=USE_PREV_STATE,
        year_min=all_years[:train_end].min(), year_max=all_years[:train_end].max()
    )

    # Split
    X_train, Y_train = X_scaled_aug[:train_end], Y_scaled[:train_end]
    X_val,   Y_val   = X_scaled_aug[train_end:val_end], Y_scaled[train_end:val_end]
    X_test,  Y_test  = X_scaled_aug[val_end:],          Y_scaled[val_end:]

    train_loader = make_data_loader(X_train, Y_train, edge_index, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = make_data_loader(X_val,   Y_val,   edge_index, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = make_data_loader(X_test,  Y_test,  edge_index, batch_size=BATCH_SIZE, shuffle=False)

    in_feats  = X_scaled_aug.shape[-1]
    out_feats = Y_scaled.shape[-1]

    model = DeepResGNN(in_feats=in_feats, out_feats=out_feats, hidden=128, heads=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=12, verbose=False)

    best_val = float('inf'); patience_counter = 0
    loss_history = {'train': [], 'val': [], 'test': []}
    start_time = time.time()
    best_ckpt = os.path.join(RESULTS_DIR, rname, 'best_GNN_model.pth')
    os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)

    for epoch in range(EPOCHS):
        tr = train_epoch(model, train_loader, optimizer, DEVICE)
        va = eval_epoch(model,   val_loader,   DEVICE)
        loss_history['train'].append(tr)
        loss_history['val'].append(va)
        scheduler.step(va)

        if va < best_val - 1e-6:
            best_val = va; patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1

        if (epoch+1) % 50 == 0 or epoch == 1:
            te = eval_epoch(model, test_loader, DEVICE)
            loss_history['test'].append(te)
            print(f"[{rname}] Epoch {epoch+1:4d}/{EPOCHS}  train={tr:.5f}  val={va:.5f}  test={te:.5f}  best_val={best_val:.5f}")

        if patience_counter >= PATIENCE:
            print(f"[{rname}] Early stopping at epoch {epoch+1}  (best val {best_val:.6f})")
            break

    train_time = time.time() - start_time
    plot_train_val_test_loss(loss_history, rname, os.path.join(RESULTS_DIR, rname))

    # Save scaler artifacts for later unseen inference (fit on TRAIN ONLY)
    rdir = os.path.join(RESULTS_DIR, rname); os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(os.path.join(rdir, "scaler.npz"),
             X_min=X_min, X_max=X_max, Y_min=Y_min, Y_max=Y_max,
             year_min=float(year_min_fit), year_max=float(year_max_fit),
             N_I=int(N_I), N_J=int(N_J))
    with open(os.path.join(rdir, 'targets.json'), 'w') as f:
        json.dump(TARGETS, f)
    np.save(os.path.join(rdir, 'years.npy'), all_years)

    # Predict full timeline (scaled -> unscaled) using TRUTH prev (for comparability on train folds)
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model.eval()
    preds_scaled = []
    start_inf = time.time()
    with torch.no_grad():
        for t in range(X_scaled_aug.shape[0]):
            x_torch = torch.tensor(X_scaled_aug[t].reshape(-1, X_scaled_aug.shape[-1]), dtype=torch.float32, device=DEVICE)
            data = Data(x=x_torch, edge_index=edge_index.to(DEVICE))
            pred = model(data.x, data.edge_index).cpu().numpy()
            preds_scaled.append(pred)
    infer_time = time.time() - start_inf
    preds_scaled = np.stack(preds_scaled)  # [T, N*N, out_feats]

    preds_unscaled = np.empty_like(preds_scaled)
    for c in range(out_feats):
        preds_unscaled[..., c] = unscale_channel(preds_scaled[..., c], c, Y_min, Y_max)
    np.save(os.path.join(rdir, 'predictions_unscaled.npy'), preds_unscaled)

    Y_true_unscaled = Y_scaled.reshape(Y_scaled.shape[0], -1, Y_scaled.shape[-1]).copy()
    for c in range(out_feats):
        Y_true_unscaled[..., c] = unscale_channel(Y_true_unscaled[..., c], c, Y_min, Y_max)

    # Metrics & maps
    for c, target in enumerate(TARGETS):
        targ_dir = os.path.join(rdir, target); os.makedirs(targ_dir, exist_ok=True)

    for t_idx, year in enumerate(all_years):
        y_true = Y_true_unscaled[t_idx]; y_pred = preds_unscaled[t_idx]
        for c, target in enumerate(TARGETS):
            true_map = y_true[:, c].reshape(N_I0, N_J0)
            pred_map = y_pred[:, c].reshape(N_I0, N_J0)
            err_map  = pred_map - true_map

            plt.imsave(os.path.join(rdir, target, f"cmg_{year}.png"), np.flipud(true_map.T), cmap='viridis')
            plt.imsave(os.path.join(rdir, target, f"gnn_{year}.png"), np.flipud(pred_map.T), cmap='viridis')
            plt.imsave(os.path.join(rdir, target, f"err_{year}.png"), np.flipud(err_map.T), cmap='coolwarm')

            mask = ~np.isnan(true_map.flatten())
            if mask.sum() > 0:
                _r2   = r2_score(true_map.flatten()[mask],  pred_map.flatten()[mask])
                _mae  = mean_absolute_error(true_map.flatten()[mask], pred_map.flatten()[mask])
                _rmse = np.sqrt(mean_squared_error(true_map.flatten()[mask], pred_map.flatten()[mask]))
            else:
                _r2 = _mae = _rmse = np.nan
            all_metrics_rows.append({'Realization': rname, 'Target': target, 'Year': int(year),
                                     'R2': _r2, 'MAE': _mae, 'RMSE': _rmse})

    # publication maps for training folds
    for c, target in enumerate(TARGETS):
        true_stack = np.stack([Y_true_unscaled[t,:,c].reshape(N_I0, N_J0) for t in range(len(all_years))])
        pred_stack = np.stack([preds_unscaled[t,:,c].reshape(N_I0, N_J0) for t in range(len(all_years))])
        err_stack  = pred_stack - true_stack

        vmin = min(np.nanmin(true_stack), np.nanmin(pred_stack))
        vmax = max(np.nanmax(true_stack), np.nanmax(pred_stack))
        err_abs = np.nanmax(np.abs(err_stack))

        save_dir = os.path.join(PUB_DIR, rname, target); os.makedirs(save_dir, exist_ok=True)
        for year in pub_years:
            if year in all_years:
                idx = np.where(all_years == year)[0][0]
                publication_triple_map(true_stack[idx], pred_stack[idx], err_stack[idx],
                                       target, year, vmin, vmax, err_abs,
                                       savepath=os.path.join(save_dir, f"pub_map_{year}.png"))
        # (optional) train animations (off by default)

    summary_times.append({'Realization': rname, 'TrainTime_sec': train_time, 'InferenceTime_sec': infer_time})

# Save global training tables (safe)
metrics_df = pd.DataFrame(all_metrics_rows)
safe_to_csv(metrics_df, os.path.join(RESULTS_DIR, "metrics_all_realizations.csv"), index=False)

times_df = pd.DataFrame(summary_times)
safe_to_csv(times_df, os.path.join(RESULTS_DIR, "computational_times_all_realizations.csv"), index=False)

summary_stats = (metrics_df.groupby(['Target','Year'])
                 .agg(R2_mean=('R2','mean'), R2_std=('R2','std'),
                      MAE_mean=('MAE','mean'), MAE_std=('MAE','std'),
                      RMSE_mean=('RMSE','mean'), RMSE_std=('RMSE','std'))
                 .reset_index())
safe_to_csv(summary_stats, os.path.join(RESULTS_DIR, "metrics_summary_by_target_year.csv"), index=False)

# Snapshot run configuration
run_cfg = dict(
    DATA_ROOT=DATA_ROOT, TRAIN_COUNT=TRAIN_COUNT, UNSEEN_COUNT=UNSEEN_COUNT,
    RESULTS_DIR=RESULTS_DIR, PUB_DIR=PUB_DIR, UNSEEN_OUT_DIR=UNSEEN_OUT_DIR,
    k_layer=k_layer, USE_PREV_STATE=USE_PREV_STATE, USE_DIAGONAL_EDGES=USE_DIAGONAL_EDGES,
    EPOCHS=EPOCHS, PATIENCE=PATIENCE, LR=LR, WEIGHT_DECAY=WEIGHT_DECAY, BATCH_SIZE=BATCH_SIZE,
    SEED_BASE=SEED_BASE, SAVE_ANIMATIONS_TRAIN=SAVE_ANIMATIONS_TRAIN, MAKE_ANIMS_UNSEEN=MAKE_ANIMS_UNSEEN,
    TARGETS=TARGETS, YEARS_REFERENCE=common_years_sorted
)
with open(os.path.join(RESULTS_DIR, "run_config.json"), "w") as f:
    json.dump(run_cfg, f, indent=2)

print("\nFinished training.")
print(f"- Metrics: {os.path.join(RESULTS_DIR, 'metrics_all_realizations.csv')}")
print(f"- Summary: {os.path.join(RESULTS_DIR, 'metrics_summary_by_target_year.csv')}")
print(f"- Times:   {os.path.join(RESULTS_DIR, 'computational_times_all_realizations.csv')}")

# ====================================
# ENSEMBLE AGGREGATION (TRAIN folds) + P50
# ====================================
print("\nAggregating training ensemble predictions...")
pred_files = []
for ridx, rfolder in enumerate(train_folders, start=1):
    rname = f"real_{ridx:02d}"
    rdir = os.path.join(RESULTS_DIR, rname)
    p = os.path.join(rdir, 'predictions_unscaled.npy')
    y = os.path.join(rdir, 'years.npy')
    tjson = os.path.join(rdir, 'targets.json')
    if os.path.exists(p) and os.path.exists(y) and os.path.exists(tjson):
        yrs = np.load(y)
        with open(tjson, 'r') as f:
            tgts = json.load(f)
        if tgts != TARGETS:
            raise ValueError(f"Targets mismatch in {rname}. Expected {TARGETS}, found {tgts}")
        if not np.all(yrs == years_reference):
            raise ValueError(f"Year mismatch in {rname} during ensemble aggregation.")
        pred_files.append(p)

if len(pred_files) > 0:
    preds_list = [np.load(p) for p in pred_files]
    base_shape = preds_list[0].shape
    if any(arr.shape != base_shape for arr in preds_list):
        raise ValueError("Prediction shapes differ across realizations.")
    preds_stack = np.stack(preds_list, axis=0)  # [R, T, NN, C]
    ens_mean = np.nanmean(preds_stack, axis=0)
    ens_std  = np.nanstd(preds_stack,  axis=0)
    ens_p10  = np.nanpercentile(preds_stack, 10, axis=0)
    ens_p50  = np.nanpercentile(preds_stack, 50, axis=0)
    ens_p90  = np.nanpercentile(preds_stack, 90, axis=0)

    ens_dir = os.path.join(PUB_DIR, "ensemble"); os.makedirs(ens_dir, exist_ok=True)
    np.savez_compressed(os.path.join(ens_dir, "train_ensemble_stats.npz"),
                        mean=ens_mean, std=ens_std, p10=ens_p10, p50=ens_p50, p90=ens_p90,
                        years=years_reference, targets=np.array(TARGETS, dtype=object))
    print(f"Training ensemble arrays saved under: {ens_dir}")
else:
    print("No training predictions found to aggregate.")

# =========================================================
# UNSEEN INFERENCE: run ensemble on each held-out realization
# =========================================================
def load_scaler(npz_path):
    z = np.load(npz_path)
    return {
        'X_min': z['X_min'], 'X_max': z['X_max'],
        'Y_min': z['Y_min'], 'Y_max': z['Y_max'],
        'year_min': float(z['year_min']), 'year_max': float(z['year_max']),
        'N_I': int(z['N_I']), 'N_J': int(z['N_J'])
    }

def resolve_inputs_only(df):
    def pick(pats): return _regex_find(list(df.columns), pats)
    poro = pick(INPUT_PATTERNS['Porosity'])
    permi = pick(INPUT_PATTERNS['PermI']) or pick(INPUT_PATTERNS['PermAny'])
    permk = pick(INPUT_PATTERNS['PermK']) or pick(INPUT_PATTERNS['PermAny'])
    netpay = pick(INPUT_PATTERNS['NetPay'])
    npv    = pick(INPUT_PATTERNS['NetPoreVolume'])
    pres   = pick(INPUT_PATTERNS['Pressure'])
    gbv    = pick(INPUT_PATTERNS['GrossBlockVolume'])
    if poro is None: raise KeyError("Porosity column not found in unseen data.")
    return {'Porosity':poro, 'PermI':permi, 'PermK':permk, 'NetPay':netpay, 'NPV':npv, 'Pressure':pres, 'GBV':gbv}

def _find_idx_cols_unseen(df):
    i_col = _find_col(df, INDEX_PATTERNS['i'], True, 'i_index')
    j_col = _find_col(df, INDEX_PATTERNS['j'], True, 'j_index')
    k_col = _find_col(df, INDEX_PATTERNS['k'], True, 'k_index')
    return i_col, j_col, k_col

def read_and_std_unseen(path):
    df = pd.read_csv(path)
    i_col, j_col, k_col = _find_idx_cols_unseen(df)
    df = df.rename(columns={i_col: 'i', j_col: 'j', k_col: 'k'})
    drop = [c for c in df.columns if re.search(r'\b(x_coord|y_coord|z_coord|Time)\b', c, re.I)]
    return df.drop(columns=drop, errors='ignore')

def merge_year_files_unseen(paths):
    combo, seen = None, set()
    for p in paths:
        d = read_and_std_unseen(p)
        keep = ['i','j','k'] + [c for c in d.columns if c not in ('i','j','k') and c not in seen]
        d = d[keep]; seen.update([c for c in keep if c not in ('i','j','k')])
        combo = d if combo is None else combo.merge(d, on=['i','j','k'], how='outer')
    return combo.loc[:, ~combo.columns.duplicated()]

def build_edge_index2(NI, NJ, diag=False):
    return build_edge_index(NI, NJ, diag=diag)

def ensemble_infer_unseen(unseen_folder, members, targets):
    # 1) Build base inputs from unseen CSVs
    y2files = group_files_by_year(unseen_folder)
    years = [y for y in sorted(y2files) if 2030 <= y <= 2130]
    if not years: raise RuntimeError(f"No usable years [2030,2130] in {unseen_folder}")

    df0 = merge_year_files_unseen(y2files[years[0]])
    i_col, j_col, k_col = 'i', 'j', 'k'
    NI = int(df0[i_col].max()); NJ = int(df0[j_col].max())

    def grid_prop(df, col, mask):
        g = np.full((NI, NJ), np.nan)
        for _, r in df[mask].iterrows():
            g[int(r[i_col])-1, int(r[j_col])-1] = r[col]
        return g

    base_inputs = []
    for y in years:
        dfy = merge_year_files_unseen(y2files[y])
        inp_cols = resolve_inputs_only(dfy)
        mask = (dfy['k'] == k_layer)

        poro  = grid_prop(dfy, inp_cols['Porosity'], mask)
        permi = grid_prop(dfy, inp_cols['PermI'], mask) if inp_cols['PermI'] else np.zeros_like(poro)
        permk = grid_prop(dfy, inp_cols['PermK'], mask) if inp_cols['PermK'] else np.zeros_like(poro)
        permi = np.log1p(np.abs(permi)) * np.sign(permi)
        permk = np.log1p(np.abs(permk)) * np.sign(permk)

        netpay = grid_prop(dfy, inp_cols['NetPay'], mask) if inp_cols['NetPay'] else np.zeros_like(poro)
        gbv_col = inp_cols.get('GBV')
        npv    = grid_prop(dfy, inp_cols['NPV'], mask) if inp_cols['NPV'] else (
                 grid_prop(dfy, gbv_col, mask)*poro if gbv_col else np.zeros_like(poro))
        pres   = grid_prop(dfy, inp_cols['Pressure'], mask) if inp_cols['Pressure'] else np.zeros_like(poro)

        base_inputs.append(np.stack([poro, permi, permk, netpay, npv, pres], axis=-1))
    base_inputs = np.stack(base_inputs)  # [T,NI,NJ,6]
    base_inputs = np.nan_to_num(base_inputs, 0.0, 0.0, 0.0)

    edge_index2 = build_edge_index2(NI, NJ, diag=USE_DIAGONAL_EDGES).to(DEVICE)

    # Determine idx for prev-state if Gas_Saturation exists
    gas_idx = targets.index('Gas_Saturation') if 'Gas_Saturation' in targets else None

    # 2) Run each trained member sequentially on unseen data
    all_member_preds_unscaled = []  # [R,T,NI*NJ,C]
    for m in members:
        sc = load_scaler(m['scaler'])
        if (sc['N_I'], sc['N_J']) != (NI, NJ):
            raise ValueError(f"Grid mismatch: unseen {NI,NJ} vs member {m['name']} scaler {sc['N_I'],sc['N_J']}")

        Xmin, Xmax = sc['X_min'], sc['X_max']
        Ymin, Ymax = sc['Y_min'], sc['Y_max']
        base_scaled = (base_inputs - Xmin) / (Xmax - Xmin + 1e-8)

        y0, y1 = sc['year_min'], sc['year_max']
        year_norm = (np.array(years, dtype=float) - y0) / (y1 - y0 + 1e-8)
        clipped = np.any((year_norm < 0) | (year_norm > 1))
        year_norm = clamp01(year_norm)[:,None,None,None]
        if clipped:
            print(f"[warn] Year normalization clipped for {unseen_folder} (outside [{y0}, {y1}]).")
        year_chan = np.repeat(year_norm, NI, axis=1).repeat(NJ, axis=2)  # [T,NI,NJ,1]

        prev = np.zeros((NI, NJ, 1), dtype=np.float32)  # prev-state channel (gas if available)

        in_feats  = base_scaled.shape[-1] + 2  # +year +prev
        out_feats = len(targets)
        net = DeepResGNN(in_feats=in_feats, out_feats=out_feats, hidden=128, heads=4).to(DEVICE)
        net.load_state_dict(torch.load(m['pth'], map_location=DEVICE))
        net.eval()

        preds_scaled = []
        with torch.no_grad():
            for t in range(len(years)):
                xt = np.concatenate([base_scaled[t], year_chan[t], prev], axis=-1)  # [NI,NJ,8]
                xnodes = torch.tensor(xt.reshape(-1, xt.shape[-1]), dtype=torch.float32, device=DEVICE)
                out = net(xnodes, edge_index2).cpu().numpy()
                preds_scaled.append(out)
                if gas_idx is not None:
                    prev = out[:,gas_idx:gas_idx+1].reshape(NI, NJ, 1)  # roll predictions as prev
                else:
                    prev = np.zeros_like(prev)

        preds_scaled = np.stack(preds_scaled)  # [T,NI*NJ,C]
        preds_unscaled = np.empty_like(preds_scaled)
        for c in range(out_feats):
            ymin = Ymin[0,0,0,c]; ymax = Ymax[0,0,0,c]
            preds_unscaled[..., c] = preds_scaled[..., c]*(ymax-ymin) + ymin
        all_member_preds_unscaled.append(preds_unscaled)

    all_member_preds_unscaled = np.stack(all_member_preds_unscaled, axis=0)  # [R,T,NI*NJ,C]
    return years, NI, NJ, np.array(all_member_preds_unscaled)

# Collect trained members
members = []
for rd in sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("real_")]):
    based = os.path.join(RESULTS_DIR, rd)
    pth = os.path.join(based, "best_GNN_model.pth")
    scaler = os.path.join(based, "scaler.npz")
    tj = os.path.join(based, "targets.json")
    if os.path.exists(pth) and os.path.exists(scaler) and os.path.exists(tj):
        with open(tj,'r') as f: tgts = json.load(f)
        if tgts == TARGETS:
            members.append({'name': rd, 'pth': pth, 'scaler': scaler})
print(f"\n[ensemble] Found {len(members)} trained members with scalers.")

# Run unseen inference + save animations/PNGs (with colorbars; includes P50)
for uidx, ufolder in enumerate(unseen_folders, start=1):
    print(f"\n=== Unseen inference on: {ufolder}")
    years_u, NI_u, NJ_u, preds_stack = ensemble_infer_unseen(ufolder, members, TARGETS)  # [R,T,NN,C]
    R,T,NN,C = preds_stack.shape
    ens_mean = np.nanmean(preds_stack, axis=0)   # [T,NN,C]
    ens_std  = np.nanstd (preds_stack, axis=0)
    ens_p10  = np.nanpercentile(preds_stack, 10, axis=0)
    ens_p50  = np.nanpercentile(preds_stack, 50, axis=0)
    ens_p90  = np.nanpercentile(preds_stack, 90, axis=0)

    # Save arrays (compressed)
    out_base = os.path.join(UNSEEN_OUT_DIR, f"unseen_{uidx:02d}")
    os.makedirs(out_base, exist_ok=True)
    np.save(os.path.join(out_base, "years.npy"), np.array(years_u))
    np.savez_compressed(os.path.join(out_base, "ensemble_stats.npz"),
                        mean=ens_mean, std=ens_std, p10=ens_p10, p50=ens_p50, p90=ens_p90,
                        targets=np.array(TARGETS, dtype=object))

    # Per-target PNGs + comprehensive 1x5 GIF (colorbars on each)
    for c, tgt in enumerate(TARGETS):
        tdir = os.path.join(out_base, tgt); os.makedirs(tdir, exist_ok=True)

        # stable ranges across years in PNGs
        vmin_mean = float(np.nanmin([ens_mean[...,c], ens_p10[...,c], ens_p50[...,c], ens_p90[...,c]]))
        vmax_mean = float(np.nanmax([ens_mean[...,c], ens_p10[...,c], ens_p50[...,c], ens_p90[...,c]]))
        vmin_std  = 0.0
        vmax_std  = float(np.nanmax(ens_std[...,c]))

        save_png_with_colorbar(ens_mean[...,c], "Mean", tgt, years_u, NI_u, NJ_u, tdir,
                               'viridis', vmin_mean, vmax_mean)
        save_png_with_colorbar(ens_std [...,c], "Std",  tgt, years_u, NI_u, NJ_u, tdir,
                               'magma',  vmin_std,  vmax_std)
        save_png_with_colorbar(ens_p10 [...,c], "P10",  tgt, years_u, NI_u, NJ_u, tdir,
                               'viridis', vmin_mean, vmax_mean)
        save_png_with_colorbar(ens_p50 [...,c], "P50",  tgt, years_u, NI_u, NJ_u, tdir,
                               'viridis', vmin_mean, vmax_mean)
        save_png_with_colorbar(ens_p90 [...,c], "P90",  tgt, years_u, NI_u, NJ_u, tdir,
                               'viridis', vmin_mean, vmax_mean)

        animate_five_panel_with_cbs(ens_mean[...,c], ens_std[...,c], ens_p10[...,c], ens_p50[...,c], ens_p90[...,c],
                                    years_u, NI_u, NJ_u, tgt, tdir)

print("\nAll done. Training results in:")
print(" -", RESULTS_DIR)
print(" -", PUB_DIR)
print("Unseen ensemble outputs (PNGs + GIFs) in:")
print(" -", UNSEEN_OUT_DIR)
