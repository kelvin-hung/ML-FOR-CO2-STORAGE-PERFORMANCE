import os
import glob
import pandas as pd
import re
import io

input_dir = 'data/model-48/input'
output_dir = 'data/model-48/output'
csv_dir = 'data/model-48/csv'
os.makedirs(csv_dir, exist_ok=True)

# --- 1. Read all input features into one DataFrame ---
input_files = sorted(glob.glob(os.path.join(input_dir, '*.gslib')))
input_feature_names = [os.path.splitext(os.path.basename(f))[0].replace(' ', '_') for f in input_files]
df_inputs = None
for feature, file in zip(input_feature_names, input_files):
    # Skip the header lines
    df = pd.read_csv(file, sep=r'\s+', skiprows=9, header=None)
    if df.shape[1] == 7:
        df.columns = ['i_index', 'j_index', 'k_index', 'x', 'y', 'z', feature]
        df = df[['i_index', 'j_index', 'k_index', feature]]
    elif df.shape[1] == 4:
        df.columns = ['i_index', 'j_index', 'k_index', feature]
    else:
        raise ValueError(f"Unexpected number of columns ({df.shape[1]}) in {file}")
    if df_inputs is None:
        df_inputs = df
    else:
        df_inputs = pd.merge(df_inputs, df, on=['i_index', 'j_index', 'k_index'])

# --- 2. Parse ALL time steps from each OUTPUT file ---
output_patterns = {
    "Gas_Saturation": '*Gas Saturation.gslib',
    "Pressure": '*Pressure.gslib',
    "Subsidence": '*Subsidence from Geomechanics.gslib',
    "pH": '*pH.gslib',
}

output_files = {k: glob.glob(os.path.join(output_dir, pat))[0] for k, pat in output_patterns.items()}

def parse_gslib_blocks(filepath, value_colname):
    with open(filepath) as f:
        lines = f.readlines()
    n_lines = len(lines)
    i = 0
    expected_ncol = 7
    time_blocks = {}
    while i < n_lines:
        if not lines[i].startswith('CMG Results:'):
            i += 1
            continue
        header = lines[i]
        m = re.search(r'Time\s+(\d+)\s+\(([\d\-A-Za-z]+)\)', header)
        if m:
            time_step = m.group(1)
            date_str = m.group(2).replace('-', '_')
            block_key = (time_step, date_str)
        else:
            block_key = (f"unknown_{i}", "")
        ncol = int(lines[i+1].strip())
        block_start = i + 3
        block_end = block_start
        while block_end < n_lines and not lines[block_end].startswith('CMG Results:'):
            block_end += 1
        filtered_lines = [ln for ln in lines[block_start:block_end] if len(ln.strip().split()) == expected_ncol]
        if not filtered_lines:
            i = block_end
            continue
        df = pd.read_csv(io.StringIO(''.join(filtered_lines)), sep=r'\s+', header=None)
        df.columns = ['i_index', 'j_index', 'k_index', 'x', 'y', 'z', value_colname]
        time_blocks[block_key] = df[['i_index', 'j_index', 'k_index', value_colname]]
        i = block_end
    return time_blocks

# Parse all time steps for each output
blocks_dict = {k: parse_gslib_blocks(path, k) for k, path in output_files.items()}

# Find time steps present in ALL outputs
all_keys = sorted(set.intersection(*(set(blocks_dict[k].keys()) for k in blocks_dict)))
print(f"Found {len(all_keys)} time steps in common.")

# --- 3. Merge INPUTS and all OUTPUTS, write CSV per time step ---
block_count = 0
for (time_step, date_str) in all_keys:
    csv_name = f"ml_data_time_{time_step}_{date_str}.csv"
    csv_path = os.path.join(csv_dir, csv_name)
    df = df_inputs.copy()
    # Merge outputs one by one
    for k in ['Gas_Saturation', 'Pressure', 'Subsidence', 'pH']:
        df = pd.merge(df, blocks_dict[k][(time_step, date_str)], on=['i_index', 'j_index', 'k_index'])
    # Filter invalid
    df = df[(df['Gas_Saturation'] != -9999) & (df['Pressure'] != -9999)]
    df.to_csv(csv_path, index=False)
    block_count += 1
    print(f"Saved: {csv_path}")

print(f"=== ALL DONE! Processed {block_count} time steps ===")
