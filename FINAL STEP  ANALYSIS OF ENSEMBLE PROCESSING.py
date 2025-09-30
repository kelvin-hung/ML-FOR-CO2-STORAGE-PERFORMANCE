# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 12:50:57 2025

@author: hungv
"""

# post_train_analysis.py
# Comprehensive post-hoc analysis for the upgraded GNN ensemble workflow.
# Produces figures & CSVs for TRAIN ensemble + each UNSEEN model (P10/P50/P90).
# Requires outputs from the training script (compare71_GNN_ensemble + inference_unseen_out2).

import os, re, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --------------------------- CONFIG ---------------------------
# These should match your training run; they'll be auto-read from run_config.json if present.
RESULTS_DIR    = 'compare91_GNN_ensemble'
PUB_DIR        = 'publication91_GNN_ensemble'
UNSEEN_OUT_DIR = 'inference_unseen_out3'
ANALYSIS_DIR   = 'analysis_outputsfi'  # all analysis artifacts go here
K_LAYER        = 1                   # layer to rebuild truth for

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# --------------------------- I/O helpers ---------------------------
def safe_to_csv(df, path, index=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)
    tmp = f"{base}.tmp.{os.getpid()}{ext}"
    df.to_csv(tmp, index=index)
    try:
        os.replace(tmp, path)
        return path
    except PermissionError:
        alt = f"{base}_{time.strftime('%Y%m%d-%H%M%S')}{ext}"
        os.replace(tmp, alt)
        print(f"[warn] '{path}' locked; wrote '{alt}'")
        return alt

def list_dirs(path, prefix=None):
    if not os.path.isdir(path): return []
    ds = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    if prefix: ds = [d for d in ds if d.startswith(prefix)]
    return ds

# --------------------------- Training run config ---------------------------
RUN_CFG_PATH = os.path.join(RESULTS_DIR, 'run_config.json')
if os.path.exists(RUN_CFG_PATH):
    with open(RUN_CFG_PATH, 'r') as f:
        _rc = json.load(f)
    DATA_ROOT     = _rc.get('DATA_ROOT', r'ccs-data analytic/data')
    TRAIN_COUNT   = int(_rc.get('TRAIN_COUNT', 45))
    UNSEEN_COUNT  = int(_rc.get('UNSEEN_COUNT', 3))
    YEARS_REF     = _rc.get('YEARS_REFERENCE', [])
    TARGETS       = _rc.get('TARGETS', ['Gas_Saturation','pH','Subsidence'])
    K_LAYER       = int(_rc.get('k_layer', K_LAYER))
else:
    print(f"[warn] {RUN_CFG_PATH} not found. Falling back to defaults.")
    DATA_ROOT = r'ccs-data analytic/data'
    TRAIN_COUNT, UNSEEN_COUNT = 45, 3
    TARGETS = ['Gas_Saturation','pH','Subsidence']
    YEARS_REF = []

# --------------------------- Discovery (matching training script) ---------------------------
YEAR_IN_NAME = re.compile(r'(?<!\d)(?:19|20|21)\d{2}(?!\d)')
def extract_year(filename: str) -> int:
    hits = YEAR_IN_NAME.findall(os.path.basename(filename))
    return int(hits[-1]) if hits else -1

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
    if not os.path.isdir(root):
        return out
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

def group_files_by_year(folder):
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    year2files = {}
    for f in csv_files:
        y = extract_year(f)
        if y <= 0: continue
        year2files.setdefault(y, []).append(os.path.join(folder, f))
    return dict(sorted(year2files.items(), key=lambda kv: kv[0]))

# --------------------------- Column detection (matching training) ---------------------------
def _regex_find(cols, patterns):
    for pat in patterns:
        rc = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rc.search(c): return c
    return None

INDEX_PATTERNS = {
    'i': [r'^i_index$', r'^i$', r'\bI[-_ ]?index\b'],
    'j': [r'^j_index$', r'^j$', r'\bJ[-_ ]?index\b'],
    'k': [r'^k_index$', r'^k$', r'\bK[-_ ]?index\b'],
}
OUTPUT_PATTERNS = {
    'Gas_Saturation': [r'\bGas[\s_]*Saturation\b.*', r'^(Sg|SGAS)\b.*', r'\bGas[-_ ]?Sat\b.*'],
    'pH':             [r'(^|\b)pH\b.*'],
    'Subsidence':     [r'\bSubsidence\b.*', r'VDISPL.*Z.*', r'\bSurface.*(Z|Vertical).*displacement.*'],
}

def _find_idx_cols(df):
    def _f(pats, nm):
        c = _regex_find(list(df.columns), pats)
        if c is None: raise KeyError(f"Missing index column {nm}")
        return c
    i_col = _f(INDEX_PATTERNS['i'], 'i_index')
    j_col = _f(INDEX_PATTERNS['j'], 'j_index')
    k_col = _f(INDEX_PATTERNS['k'], 'k_index')
    return i_col, j_col, k_col

def merge_year_files(paths):
    import pandas as pd, re as _re
    combo, have = None, set()
    for p in paths:
        df = pd.read_csv(p)
        i_col, j_col, k_col = _find_idx_cols(df)
        df = df.rename(columns={i_col:'i', j_col:'j', k_col:'k'})
        drop_cols = [c for c in df.columns if _re.search(r'\b(x_coord|y_coord|z_coord|Time)\b', c, _re.I)]
        df = df.drop(columns=drop_cols, errors='ignore')
        keep_cols = ['i','j','k'] + [c for c in df.columns if c not in ('i','j','k') and c not in have]
        df = df[keep_cols]
        have.update([c for c in keep_cols if c not in ('i','j','k')])
        combo = df if combo is None else combo.merge(df, on=['i','j','k'], how='outer')
    return combo.loc[:, ~combo.columns.duplicated()]

def detect_output_cols(df, targets):
    out_map = {}
    for t in targets:
        col = _regex_find(list(df.columns), OUTPUT_PATTERNS[t])
        if col is not None:
            out_map[t] = col
    return out_map

# --------------------------- Grid helpers ---------------------------
def get_grid_shape_from_scaler(results_dir):
    reals = list_dirs(results_dir, prefix='real_')
    for rd in reals:
        sc = os.path.join(results_dir, rd, 'scaler.npz')
        if os.path.exists(sc):
            z = np.load(sc)
            return int(z['N_I']), int(z['N_J'])
    return None, None

def best_factor_pair(n):
    best = (1, n); best_diff = n-1
    r = int(np.sqrt(n))
    for a in range(1, r+1):
        if n % a == 0:
            b = n // a
            if abs(a-b) < best_diff:
                best = (a, b); best_diff = abs(a-b)
    return best

# --------------------------- Truth builder (for unseen folds) ---------------------------
def build_truth_stack_for_unseen(data_root, unseen_index, years_u, targets, k_layer):
    """
    unseen_index: zero-based index into discovery ordering (TRAIN_COUNT..TRAIN_COUNT+UNSEEN_COUNT-1).
    Returns [T, NN, C].
    """
    model_dirs = discover_realization_folders(data_root)
    if unseen_index >= len(model_dirs):
        raise RuntimeError(f"Unseen index {unseen_index} exceeds discovered dirs ({len(model_dirs)})")
    csv_dir = model_dirs[unseen_index]

    y2files = group_files_by_year(csv_dir)
    if years_u[0] not in y2files:
        raise RuntimeError(f"[truth] First unseen year {years_u[0]} missing in {csv_dir}")

    df0 = merge_year_files(y2files[years_u[0]])
    i_col, j_col, k_col = 'i','j','k'
    NI, NJ = int(df0[i_col].max()), int(df0[j_col].max())

    perT = []
    for y in years_u:
        if y not in y2files:
            raise RuntimeError(f"[truth] Year {y} missing in {csv_dir}")
        dfy = merge_year_files(y2files[y])
        out_map = detect_output_cols(dfy, targets)
        missing = [t for t in targets if t not in out_map]
        if missing:
            raise RuntimeError(f"[truth] Missing outputs {missing} in {csv_dir} year {y}")
        mask = (dfy['k'] == k_layer)

        def grid_prop(df, col):
            g = np.full((NI, NJ), np.nan)
            for _, r in df[mask].iterrows():
                g[int(r['i'])-1, int(r['j'])-1] = r[col]
            return g

        chans = [grid_prop(dfy, out_map[t]) for t in targets]  # list of (NI,NJ)
        perT.append(np.stack(chans, axis=-1).reshape(-1, len(targets)))  # [NN, C]
    return np.stack(perT, axis=0)  # [T, NN, C]

# --------------------------- Plot helpers ---------------------------
def save_map_with_cb(mat, title, out_path, cmap='viridis', vmin=None, vmax=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    im = ax.imshow(np.flipud(mat.T), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_ylabel('Value' if cmap!='coolwarm' else 'Error')
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)

def pinball_loss(y, q, tau):
    d = y - q
    return np.maximum(tau*d, (tau-1)*d)

# --------------------------- Load TRAIN ensemble stats (optional) ---------------------------
train_ens_npz = os.path.join(PUB_DIR, "ensemble", "train_ensemble_stats.npz")
if os.path.exists(train_ens_npz):
    tren = np.load(train_ens_npz, allow_pickle=True)
    train_targets = list(tren['targets'])
    train_years   = tren['years']
    print(f"[load] training ensemble stats: {train_ens_npz} (targets={train_targets})")

# --------------------------- Grid shape from scaler ---------------------------
NI_saved, NJ_saved = get_grid_shape_from_scaler(RESULTS_DIR)
if (NI_saved is None) or (NJ_saved is None):
    raise RuntimeError("Could not find scaler.npz to determine grid shape.")

# --------------------------- Unseen dirs ---------------------------
unseen_dirs = list_dirs(UNSEEN_OUT_DIR, prefix='unseen_')
if not unseen_dirs:
    raise RuntimeError(f"No unseen_* dirs found in {UNSEEN_OUT_DIR}")

# --------------------------- Analyze each unseen ---------------------------
agg_rows = []  # collect summary across unseen models
for uidx, udir in enumerate(unseen_dirs, start=1):
    base = os.path.join(UNSEEN_OUT_DIR, udir)
    # Ensemble stats
    stats_npz = os.path.join(base, "ensemble_stats.npz")
    if not os.path.exists(stats_npz):
        # backward-compat: older files
        raise RuntimeError(f"Missing stats npz in {base}: {stats_npz}")

    st = np.load(stats_npz, allow_pickle=True)
    ens_mean = st['mean']     # [T, NN, C]
    ens_std  = st['std']
    ens_p10  = st['p10']
    ens_p50  = st['p50']
    ens_p90  = st['p90']
    tgts_u   = list(st['targets'])

    # Years
    years_path = os.path.join(base, "years.npy")
    if not os.path.exists(years_path):
        raise RuntimeError(f"Missing years.npy in {base}")
    years_u = np.load(years_path)

    # Truth stack for this unseen (use discovery ordering: TRAIN_COUNT..TRAIN_COUNT+UNSEEN_COUNT-1)
    unseen_discovery_index = TRAIN_COUNT + (uidx-1)
    truth_stack = build_truth_stack_for_unseen(DATA_ROOT, unseen_discovery_index, years_u, tgts_u, K_LAYER)
    T, NN, C = ens_p50.shape
    if truth_stack.shape != (T, NN, C):
        raise RuntimeError(f"[shape] truth {truth_stack.shape} vs preds {ens_p50.shape} mismatch for {udir}")

    # Output dirs
    OUT_U = os.path.join(ANALYSIS_DIR, udir)
    os.makedirs(OUT_U, exist_ok=True)

    # ---------- Per-target analysis ----------
    for c_idx, tgt in enumerate(tgts_u):
        OUT_T = os.path.join(OUT_U, tgt); os.makedirs(OUT_T, exist_ok=True)

        # Time-series metrics
        rows = []
        for t_idx, yy in enumerate(years_u):
            y_true = truth_stack[t_idx, :, c_idx]
            y_p50  = ens_p50   [t_idx, :, c_idx]
            y_p10  = ens_p10   [t_idx, :, c_idx]
            y_p90  = ens_p90   [t_idx, :, c_idx]
            s_map  = ens_std   [t_idx, :, c_idx]

            m = ~np.isnan(y_true) & ~np.isnan(y_p50)
            if not m.any():
                rows.append({"Year": int(yy), "R2": np.nan, "MAE": np.nan, "RMSE": np.nan,
                             "Coverage_P10P90": np.nan, "Sharpness_meanStd": np.nan,
                             "Bias_mean": np.nan, "Pinball_0.1": np.nan,
                             "Pinball_0.5": np.nan, "Pinball_0.9": np.nan})
                continue

            yt = y_true[m]; yp = y_p50[m]; p10 = y_p10[m]; p90 = y_p90[m]; ss = s_map[m]
            denom = np.sum((yt - yt.mean())**2)
            r2 = 1 - np.sum((yt-yp)**2)/denom if denom > 0 else np.nan
            mae = np.mean(np.abs(yt-yp))
            rmse= np.sqrt(np.mean((yt-yp)**2))
            cov = np.mean((yt >= p10) & (yt <= p90))
            sharp = np.nanmean(ss)
            bias = np.mean(yp - yt)

            # pinball losses
            pb_10 = np.mean(pinball_loss(yt, p10, 0.10))
            pb_50 = np.mean(pinball_loss(yt, yp,  0.50))
            pb_90 = np.mean(pinball_loss(yt, p90, 0.90))

            rows.append({"Year": int(yy), "R2": r2, "MAE": mae, "RMSE": rmse,
                         "Coverage_P10P90": cov, "Sharpness_meanStd": sharp,
                         "Bias_mean": bias, "Pinball_0.1": pb_10,
                         "Pinball_0.5": pb_50, "Pinball_0.9": pb_90})

        df_t = pd.DataFrame(rows).sort_values("Year")
        safe_to_csv(df_t, os.path.join(OUT_T, f"{tgt}_metrics_vs_year.csv"), index=False)

        # Plots: metrics vs year
        def _plot_line(ycol, ylabel, hlines=None, ylim=None, fname=None):
            plt.figure(figsize=(7.2,4.2))
            plt.plot(df_t["Year"], df_t[ycol], marker='o', label=ycol)
            if hlines:
                for yv, lab in hlines:
                    plt.axhline(yv, color='gray', linestyle='--', label=lab)
            if ylim: plt.ylim(*ylim)
            plt.xlabel("Year"); plt.ylabel(ylabel)
            plt.title(f"{udir} — {tgt}: {ycol} vs Year")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(OUT_T, fname or f"{ycol}_vs_year.png"), dpi=160)
            plt.close()

        _plot_line("R2", "R²", fname="R2_vs_year.png")
        _plot_line("MAE", "MAE", fname="MAE_vs_year.png")
        _plot_line("RMSE","RMSE",fname="RMSE_vs_year.png")
        _plot_line("Coverage_P10P90","Coverage",
                   hlines=[(0.80,"Nominal 0.8")], ylim=(0,1),
                   fname="Coverage_vs_year.png")
        _plot_line("Sharpness_meanStd","Mean ensemble std", fname="Sharpness_vs_year.png")
        _plot_line("Bias_mean","Mean(P50 - truth)", fname="Bias_vs_year.png")
        _plot_line("Pinball_0.5","Pinball loss (τ=0.5)", fname="Pinball_tau50_vs_year.png")

        # Spaghetti: spatial mean with P10/P50/P90 & truth
        m_truth = np.nanmean(truth_stack[:,:,c_idx], axis=1)
        m_p50   = np.nanmean(ens_p50[:,:,c_idx], axis=1)
        m_p10   = np.nanmean(ens_p10[:,:,c_idx], axis=1)
        m_p90   = np.nanmean(ens_p90[:,:,c_idx], axis=1)
        plt.figure(figsize=(7.2,4.2))
        plt.plot(years_u, m_truth, label="Truth (mean)", linewidth=2)
        plt.plot(years_u, m_p50, label="P50 (mean)", linewidth=2)
        plt.fill_between(years_u, m_p10, m_p90, alpha=0.25, label="P10–P90 band")
        plt.xlabel("Year"); plt.ylabel(tgt)
        plt.title(f"{udir} — {tgt}: Spatial mean ± band")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_T, "spatial_mean_band.png"), dpi=170)
        plt.close()

        # Spread–skill: |err| vs std (subsample to keep light)
        errs_all, stds_all = [], []
        for t_idx in range(T):
            yt = truth_stack[t_idx, :, c_idx]
            yp = ens_p50   [t_idx, :, c_idx]
            ss = ens_std   [t_idx, :, c_idx]
            m  = ~np.isnan(yt) & ~np.isnan(yp) & ~np.isnan(ss)
            if m.any():
                errs_all.append(np.abs(yp[m]-yt[m]))
                stds_all.append(ss[m])
        if errs_all:
            ecat = np.concatenate(errs_all); scat = np.concatenate(stds_all)
            nmax = min(200000, ecat.size)
            if ecat.size > nmax:
                rng = np.random.default_rng(123)
                idx = rng.choice(ecat.size, size=nmax, replace=False)
                ecat = ecat[idx]; scat = scat[idx]
            corr = np.corrcoef(ecat, scat)[0,1] if ecat.size>1 else np.nan
            plt.figure(figsize=(6.2,6.2))
            plt.scatter(scat, ecat, s=1, alpha=0.05)
            plt.xlabel("Ensemble std"); plt.ylabel("|P50 - truth|")
            plt.title(f"{udir} — {tgt}: Spread–skill (ρ={corr:.3f})")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_T, "spread_skill.png"), dpi=170)
            plt.close()

        # Histograms: residuals & std (last year for concreteness)
        last = T-1
        ytL = truth_stack[last,:,c_idx]; ypL = ens_p50[last,:,c_idx]; ssL = ens_std[last,:,c_idx]
        mL  = ~np.isnan(ytL) & ~np.isnan(ypL) & ~np.isnan(ssL)
        if mL.any():
            res = ypL[mL]-ytL[mL]
            plt.figure(figsize=(7,4))
            plt.hist(res, bins=60, alpha=0.8)
            plt.axvline(0, color='k', linestyle='--', linewidth=1)
            plt.xlabel("Residual (P50 - truth)")
            plt.title(f"{udir} — {tgt}: Residuals histogram (year {int(years_u[last])})")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_T, "residual_hist_last_year.png"), dpi=160)
            plt.close()

            plt.figure(figsize=(7,4))
            plt.hist(ssL[mL], bins=60, alpha=0.8)
            plt.xlabel("Ensemble std")
            plt.title(f"{udir} — {tgt}: Std histogram (year {int(years_u[last])})")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_T, "std_hist_last_year.png"), dpi=160)
            plt.close()

        # Reliability bars per year: fractions in four bins: (<P10, P10–P50, P50–P90, >P90)
        bars = []
        for t_idx, yy in enumerate(years_u):
            yt = truth_stack[t_idx, :, c_idx]
            q10= ens_p10[t_idx, :, c_idx]
            q50= ens_p50[t_idx, :, c_idx]
            q90= ens_p90[t_idx, :, c_idx]
            m  = ~np.isnan(yt) & ~np.isnan(q10) & ~np.isnan(q50) & ~np.isnan(q90)
            if not m.any():
                bars.append((0,0,0,0))
                continue
            a = np.mean(yt[m] <  q10[m])              # expect ~0.10
            b = np.mean((yt[m] >= q10[m]) & (yt[m] < q50[m]))  # ~0.40
            c = np.mean((yt[m] >= q50[m]) & (yt[m] <= q90[m])) # ~0.40
            d = np.mean(yt[m] >  q90[m])              # ~0.10
            bars.append((a,b,c,d))
        bars = np.array(bars)  # [T,4]
        plt.figure(figsize=(8.5,4.2))
        labels = ['<P10','P10–P50','P50–P90','>P90']
        bottoms = np.zeros(len(years_u))
        for k, lab in enumerate(labels):
            plt.bar(years_u, bars[:,k], bottom=bottoms, width=0.8, label=lab)
            bottoms += bars[:,k]
        plt.axhline(0.10, color='gray', ls='--', lw=1)
        plt.axhline(0.50, color='gray', ls='--', lw=1)
        plt.axhline(0.90, color='gray', ls='--', lw=1)
        plt.ylim(0,1); plt.ylabel("Fraction of cells")
        plt.title(f"{udir} — {tgt}: Reliability by bins (expect 0.10/0.40/0.40/0.10)")
        plt.legend(ncol=4); plt.tight_layout()
        plt.savefig(os.path.join(OUT_T, "reliability_bars.png"), dpi=170)
        plt.close()

        # Maps for selected years (if present)
        SELECT_YEARS = [2032, 2041, 2055, 2060, 2070, 2130]
        for yy in SELECT_YEARS:
            if yy not in years_u: continue
            ti = int(np.where(years_u == yy)[0][0])
            # reshape robustly
            def _reshape(vec):
                try:
                    return vec.reshape(NI_saved, NJ_saved)
                except ValueError:
                    try: return vec.reshape(NJ_saved, NI_saved).T
                    except ValueError:
                        a,b = best_factor_pair(vec.size)
                        return vec.reshape(a,b)

            truth_map = _reshape(truth_stack[ti,:,c_idx])
            p50_map   = _reshape(ens_p50[ti,:,c_idx])
            err_map   = p50_map - truth_map
            cov_map   = ((truth_stack[ti,:,c_idx] >= ens_p10[ti,:,c_idx]) &
                         (truth_stack[ti,:,c_idx] <= ens_p90[ti,:,c_idx])).astype(float)
            cov_map   = _reshape(cov_map)

            vmin = np.nanmin([truth_map, p50_map]); vmax = np.nanmax([truth_map, p50_map])
            eabs = np.nanmax(np.abs(err_map))

            save_map_with_cb(truth_map, f"Truth — {tgt} — {yy}",
                             os.path.join(OUT_T, f"map_truth_{yy}.png"),
                             cmap='viridis', vmin=vmin, vmax=vmax)
            save_map_with_cb(p50_map, f"P50 — {tgt} — {yy}",
                             os.path.join(OUT_T, f"map_p50_{yy}.png"),
                             cmap='viridis', vmin=vmin, vmax=vmax)
            save_map_with_cb(err_map, f"Error (P50 - Truth) — {tgt} — {yy}",
                             os.path.join(OUT_T, f"map_error_{yy}.png"),
                             cmap='coolwarm', vmin=-eabs, vmax=eabs)
            save_map_with_cb(cov_map, f"Coverage mask (1 if truth ∈ [P10,P90]) — {tgt} — {yy}",
                             os.path.join(OUT_T, f"map_coverage_{yy}.png"),
                             cmap='magma', vmin=0, vmax=1)

        # Hotspots (top 0.5% std per year)
        rows_hot = []
        for ti, yy in enumerate(years_u):
            s = ens_std[ti,:,c_idx]
            if np.all(np.isnan(s)): continue
            k = max(1, int(0.005 * s.size))
            idx = np.argpartition(np.nan_to_num(s, -np.inf), -k)[-k:]
            for node in idx:
                rows_hot.append({"Year": int(yy), "Target": tgt, "Node": int(node), "Std": float(s[node])})
        df_hot = pd.DataFrame(rows_hot)
        safe_to_csv(df_hot, os.path.join(OUT_T, f"{tgt}_hotspots_top0p5pct.csv"), index=False)

        # Aggregation row (last-year snapshot)
        last = T-1
        agg_rows.append({
            "Unseen": udir, "Target": tgt, "Year": int(years_u[last]),
            "R2_last": float(df_t["R2"].values[-1]),
            "MAE_last": float(df_t["MAE"].values[-1]),
            "RMSE_last": float(df_t["RMSE"].values[-1]),
            "Coverage_last": float(df_t["Coverage_P10P90"].values[-1]),
            "Sharpness_last": float(df_t["Sharpness_meanStd"].values[-1]),
            "Bias_last": float(df_t["Bias_mean"].values[-1]),
            "Pinball0.5_last": float(df_t["Pinball_0.5"].values[-1]),
        })

# --------------------------- Save cross-unseen summary ---------------------------
if agg_rows:
    df_agg = pd.DataFrame(agg_rows)
    outp = os.path.join(ANALYSIS_DIR, "summary_across_unseen_last_year.csv")
    safe_to_csv(df_agg, outp, index=False)
    # quick pivot for a compact view
    piv = (df_agg.pivot_table(index=["Unseen","Target"],
                              values=["R2_last","MAE_last","RMSE_last",
                                      "Coverage_last","Sharpness_last","Bias_last","Pinball0.5_last"])
                 .reset_index())
    safe_to_csv(piv, os.path.join(ANALYSIS_DIR, "summary_across_unseen_last_year_pivot.csv"), index=False)

print("\n[analysis] Done. Outputs written to:")
print(" -", ANALYSIS_DIR)
