import os
import pandas as pd
import numpy as np
import re
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# ---- SETTINGS ----
data_folder = 'ccs-data analytic/data/model-4/csv'
RESULTS_DIR = 'compare41'
PUB_DIR = 'publication41'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PUB_DIR, exist_ok=True)
input_cols = [
    '3D_MODEmechL_MOD2_0003_Porosity_-_Current_2030-Jan-01',
    '3D_MODEmechL_MOD2_0003_Permeability_I_2030-Jan-01',
    '3D_MODEmechL_MOD2_0003_Permeability_K_2030-Jan-01',
    '3D_MODEmechL_MOD2_0003_Net_Pay_2030-Jan-01',
    '3D_MODEmechL_MOD2_0003_Net_Pore_Volume_-_Current_2030-Jan-01',
    'Pressure'
]
output_cols = ['Gas_Saturation', 'pH', 'Subsidence']
k_layer = 1
USE_PREV_STATE = True
pub_years = [2032, 2041, 2055, 2060, 2070, 2130]
EPOCHS = 2000

# ---- 1. LOAD & SORT FILES ----
def extract_year(filename):
    years = re.findall(r'(\d{4})', filename)
    if len(years) >= 2:
        return int(years[1])
    elif len(years) == 1:
        return int(years[0])
    else:
        return -1

csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
file_years = [(f, extract_year(f)) for f in csv_files if extract_year(f) > 0]
file_years_sorted = sorted(file_years, key=lambda x: x[1])
sorted_files = [os.path.join(data_folder, f) for f, y in file_years_sorted]
sorted_years = [y for f, y in file_years_sorted]
print("Loaded years:", sorted_years[:10], "...", sorted_years[-10:])

# ---- 2. Load Data for All Years ----
df0 = pd.read_csv(sorted_files[0])
N_I = int(df0['i_index'].max())
N_J = int(df0['j_index'].max())

def grid_property(df, prop, mask):
    grid = np.full((N_I, N_J), np.nan)
    for _, row in df[mask].iterrows():
        i = int(row['i_index']) - 1
        j = int(row['j_index']) - 1
        grid[i, j] = row[prop]
    return grid

X_series, Y_series, all_years = [], [], []
for file, y in zip(sorted_files, sorted_years):
    if 2031 <= y <= 2130:
        df = pd.read_csv(file)
        mask = df['k_index'] == k_layer
        input_channels = []
        for col in input_cols:
            vals = grid_property(df, col, mask)
            if 'Permeability' in col:
                vals = np.log1p(np.abs(vals)) * np.sign(vals)
            input_channels.append(vals)
        output_channels = [grid_property(df, col, mask) for col in output_cols]
        X_series.append(np.stack(input_channels, axis=-1))
        Y_series.append(np.stack(output_channels, axis=-1))
        all_years.append(y)

X = np.stack(X_series)
Y = np.stack(Y_series)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
all_years = np.array(all_years)

def min_max_scale(arr, axis=(0,1,2)):
    minv = np.nanmin(arr, axis=axis, keepdims=True)
    maxv = np.nanmax(arr, axis=axis, keepdims=True)
    return (arr - minv) / (maxv - minv + 1e-8), minv, maxv
X_scaled, X_min, X_max = min_max_scale(X)
Y_scaled, Y_min, Y_max = min_max_scale(Y)

# ---- 3. Feature Engineering (year + prev state) ----
years_norm = (all_years - all_years.min()) / (all_years.max() - all_years.min())
X_aug = []
for i in range(len(all_years)):
    year_channel = np.full((N_I, N_J, 1), years_norm[i])
    X_aug.append(np.concatenate([X_scaled[i], year_channel], axis=-1))
X_scaled = np.stack(X_aug)
if USE_PREV_STATE:
    X_prev = []
    for i in range(len(X_scaled)):
        if i == 0:
            prev_gas = np.zeros((N_I, N_J, 1))
        else:
            prev_gas = Y_scaled[i-1][:,:,0:1]
        X_prev.append(np.concatenate([X_scaled[i], prev_gas], axis=-1))
    X_scaled = np.stack(X_prev)

# ---- 4. Train/Test Split: use first 85% for train, rest for test (by year) ----
n_total = X_scaled.shape[0]
n_test = max(2, int(0.15 * n_total))
X_train, Y_train = X_scaled[:-n_test], Y_scaled[:-n_test]
X_test, Y_test = X_scaled[-n_test:], Y_scaled[-n_test:]
train_years, test_years = all_years[:-n_test], all_years[-n_test:]

# ---- 5. Edge Index (for GNN) ----
edge_index = []
for i in range(N_I):
    for j in range(N_J):
        node = i * N_J + j
        if j < N_J-1:
            edge_index.append([node, node+1])
            edge_index.append([node+1, node])
        if i < N_I-1:
            edge_index.append([node, node+N_J])
            edge_index.append([node+N_J, node])
edge_index = torch.tensor(edge_index).t().contiguous()

def make_data_loader(X_arr, Y_arr, batch_size=1, shuffle=True):
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

train_loader = make_data_loader(X_train, Y_train)
test_loader = make_data_loader(X_test, Y_test, shuffle=False)

# ---- 6. Model Definitions ----
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

class SimpleMLP(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_feats)
        )
    def forward(self, x, edge_index=None):
        return self.seq(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, 1)
    def forward(self, x, edge_index=None):
        B = x.shape[0] // (N_I * N_J)
        if B == 0: B = 1
        x = x.view(B, N_I, N_J, -1).permute(0,3,1,2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.permute(0,2,3,1).reshape(-1, x.shape[1])
        return x

# ---- 7. Training/Eval Routines ----
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if isinstance(model, DeepResGNN):
            pred = model(batch.x, batch.edge_index)
        else:
            pred = model(batch.x)
        loss = F.smooth_l1_loss(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if isinstance(model, DeepResGNN):
                pred = model(batch.x, batch.edge_index)
            else:
                pred = model(batch.x)
            loss = F.smooth_l1_loss(pred, batch.y)
            total_loss += loss.item()
    return total_loss / len(loader)

def plot_train_test_loss(loss_dict, name, results_dir):
    plt.figure(figsize=(7,4))
    plt.plot(loss_dict['train'], label="Train loss")
    plt.plot(loss_dict['test'], label="Test loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss Curve: {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{name}_loss_curve.png"), dpi=120)
    plt.close()

# ---- 8. Train All Models (with timing, record loss history) ----
device = torch.device('cpu')
models = {
    'GNN': DeepResGNN(in_feats=X_scaled.shape[-1], out_feats=Y_scaled.shape[-1], hidden=128, heads=4).to(device),
    'MLP': SimpleMLP(in_feats=X_scaled.shape[-1], out_feats=Y_scaled.shape[-1]).to(device),
    'CNN': SimpleCNN(in_ch=X_scaled.shape[-1], out_ch=Y_scaled.shape[-1]).to(device),
}
optimizers = {name: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) for name, model in models.items()}
best_test_loss = {name: float('inf') for name in models}
train_times, infer_times, loss_history = {}, {}, {}

for name, model in models.items():
    loss_history[name] = {'train':[], 'test':[]}
    start = time.time()
    for epoch in range(EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizers[name], device)
        loss_history[name]['train'].append(tr_loss)
        if epoch % 5 == 0 or epoch == EPOCHS-1:
            te_loss = eval_epoch(model, test_loader, device)
            loss_history[name]['test'].append(te_loss)
            if te_loss < best_test_loss[name]:
                best_test_loss[name] = te_loss
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"best_{name}_model.pth"))
    train_times[name] = time.time() - start

# ---- 9. Collect Predictions For All Years (timed) ----
def unscale(y_scaled, ch):
    return y_scaled * (Y_max[0,0,0,ch] - Y_min[0,0,0,ch]) + Y_min[0,0,0,ch]

all_preds = {}
for name, model in models.items():
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"best_{name}_model.pth")))
    model.eval()
    preds = []
    start = time.time()
    for t in range(X_scaled.shape[0]):
        x_torch = torch.tensor(X_scaled[t].reshape(-1, X_scaled.shape[-1]), dtype=torch.float32)
        data = Data(x=x_torch, edge_index=edge_index)
        data = data.to(device)
        with torch.no_grad():
            if isinstance(model, DeepResGNN):
                pred = model(data.x, data.edge_index).cpu().numpy()
            else:
                pred = model(data.x).cpu().numpy()
        preds.append(pred)
    infer_times[name] = time.time() - start
    all_preds[name] = np.stack(preds)  # shape: [T, N*N, n_out]

# ---- 10. Publication Figure Functions ----
def publication_triple_map(true_map, pred_map, err_map, target, year, vmin=None, vmax=None, err_abs=None, savepath=None):
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    # CMG
    ax = axs[0]
    im0 = ax.imshow(np.flipud(true_map.T), cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'CMG {target}\n{year}')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.04)
    cb = plt.colorbar(im0, cax=cax)
    cb.set_label('Value')
    # DL
    ax = axs[1]
    im1 = ax.imshow(np.flipud(pred_map.T), cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'DL {target}\n{year}')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.04)
    cb = plt.colorbar(im1, cax=cax)
    cb.set_label('Value')
    # Error
    ax = axs[2]
    im2 = ax.imshow(np.flipud(err_map.T), cmap='coolwarm', vmin=-err_abs, vmax=err_abs)
    ax.set_title(f'Error\n{year}')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.04)
    cb = plt.colorbar(im2, cax=cax)
    cb.set_label('Error')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

def save_all_publication_maps(
    all_years, pub_years, true_maps, pred_maps, error_maps, 
    target, model_name, pub_dir="publication41"):
    save_dir = os.path.join(pub_dir, model_name, target)
    os.makedirs(save_dir, exist_ok=True)
    vmin = min(np.nanmin(true_maps), np.nanmin(pred_maps))
    vmax = max(np.nanmax(true_maps), np.nanmax(pred_maps))
    err_abs = np.nanmax(np.abs(error_maps))
    for year in pub_years:
        if year not in all_years: continue
        idx = np.where(all_years == year)[0][0]
        publication_triple_map(
            true_map=true_maps[idx],
            pred_map=pred_maps[idx],
            err_map=error_maps[idx],
            target=target,
            year=year,
            vmin=vmin, vmax=vmax, err_abs=err_abs,
            savepath=os.path.join(save_dir, f"pub_map_{year}.png")
        )

def animate_with_colorbars(true_maps, pred_maps, error_maps, all_years, target, name, out_dir):
    from matplotlib import gridspec
    fig = plt.figure(figsize=(13,5))
    gs = gridspec.GridSpec(1, 6, width_ratios=[1,1,1,0.05,0.05,0.05])
    ax0 = plt.subplot(gs[0]); ax1 = plt.subplot(gs[1]); ax2 = plt.subplot(gs[2])
    cax0 = plt.subplot(gs[3]); cax1 = plt.subplot(gs[4]); cax2 = plt.subplot(gs[5])
    vmin = min(np.nanmin(true_maps), np.nanmin(pred_maps))
    vmax = max(np.nanmax(true_maps), np.nanmax(pred_maps))
    err_abs = np.nanmax(np.abs(error_maps))
    def update(i):
        ax0.clear(); ax1.clear(); ax2.clear(); cax0.clear(); cax1.clear(); cax2.clear()
        im0 = ax0.imshow(np.flipud(true_maps[i].T), cmap='viridis', vmin=vmin, vmax=vmax)
        ax0.set_title(f"CMG {target} {all_years[i]}")
        im1 = ax1.imshow(np.flipud(pred_maps[i].T), cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f"{name} {target} {all_years[i]}")
        im2 = ax2.imshow(np.flipud(error_maps[i].T), cmap='coolwarm', vmin=-err_abs, vmax=err_abs)
        ax2.set_title("Error")
        for ax in [ax0, ax1, ax2]: ax.axis('off')
        fig.colorbar(im0, cax=cax0); fig.colorbar(im1, cax=cax1); fig.colorbar(im2, cax=cax2)
    anim = FuncAnimation(fig, update, frames=len(all_years), interval=1000)
    anim.save(os.path.join(out_dir, "animation_with_colorbar.gif"), writer=PillowWriter(fps=1))
    plt.close()

# ---- 11. Result Saving (Metrics, Crossplot, Publication, Animation) ----
metrics_rows = []
for name, pred_flat in all_preds.items():
    model_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    for c, target in enumerate(output_cols):
        targ_dir = os.path.join(model_dir, target)
        os.makedirs(targ_dir, exist_ok=True)
        pred_maps, true_maps, error_maps = [], [], []
        r2s, maes, rmses = [], [], []
        for t, year in enumerate(all_years):
            pred_map = unscale(pred_flat[t][:,c].reshape(N_I,N_J), c)
            true_map = unscale(Y_scaled[t][:,:,c], c)
            error_map = pred_map - true_map
            pred_maps.append(pred_map)
            true_maps.append(true_map)
            error_maps.append(error_map)
            plt.imsave(os.path.join(targ_dir, f"cmg_{year}.png"), np.flipud(true_map.T), cmap='viridis')
            plt.imsave(os.path.join(targ_dir, f"{name.lower()}_{year}.png"), np.flipud(pred_map.T), cmap='viridis')
            plt.imsave(os.path.join(targ_dir, f"err_{year}.png"), np.flipud(error_map.T), cmap='coolwarm')
            mask = ~np.isnan(true_map.flatten())
            if mask.sum() > 0:
                r2s.append(r2_score(true_map.flatten()[mask], pred_map.flatten()[mask]))
                maes.append(mean_absolute_error(true_map.flatten()[mask], pred_map.flatten()[mask]))
                rmses.append(np.sqrt(mean_squared_error(true_map.flatten()[mask], pred_map.flatten()[mask])))
            else:
                r2s.append(np.nan); maes.append(np.nan); rmses.append(np.nan)
            metrics_rows.append({'Model': name, 'Target': target, 'Year': int(year), 'R2': r2s[-1], 'MAE': maes[-1], 'RMSE': rmses[-1]})
        # Crossplot
        all_true = np.concatenate([true_map.flatten() for true_map in true_maps])
        all_pred = np.concatenate([pred_map.flatten() for pred_map in pred_maps])
        mask = ~np.isnan(all_true)
        plt.figure(figsize=(6,6))
        plt.scatter(all_true[mask], all_pred[mask], alpha=0.1, s=2)
        minv, maxv = all_true[mask].min(), all_true[mask].max()
        plt.plot([minv, maxv], [minv, maxv], 'r--')
        plt.xlabel("CMG True"); plt.ylabel(f"{name} Prediction"); plt.title(f"Crossplot {name} {target}")
        plt.savefig(os.path.join(targ_dir, "crossplot.png"), dpi=120); plt.close()
        # --- Publication maps (tight colorbar, selected years)
        save_all_publication_maps(
            all_years=all_years,
            pub_years=pub_years,
            true_maps=true_maps,
            pred_maps=pred_maps,
            error_maps=error_maps,
            target=target,
            model_name=name,
            pub_dir="publication41"
        )
        # --- Animation for all years (with colorbars)
        animate_with_colorbars(
            true_maps=true_maps,
            pred_maps=pred_maps,
            error_maps=error_maps,
            all_years=all_years,
            target=target,
            name=name,
            out_dir=targ_dir
        )
    plot_train_test_loss(loss_history[name], name, RESULTS_DIR)

# ---- 12. Save Metrics Table ----
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_table.csv"), index=False)

# ---- 13. Save Computational Time Table ----
comp_times_df = pd.DataFrame([
    {'Model': name, 'TrainTime_sec': train_times[name], 'InferenceTime_sec': infer_times[name]}
    for name in models
])
comp_times_df.to_csv(os.path.join(RESULTS_DIR, "computational_times.csv"), index=False)
print(comp_times_df)

print(f"\nAll results, publication figures, crossplots, loss curves, metrics, and computational time tables are saved in:\n- {RESULTS_DIR}\n- {PUB_DIR}\n")
