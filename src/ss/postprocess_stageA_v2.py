import os
import glob
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# --- Configuration ---
DEFAULT_PATTERN = "_calib_008_A_"
RESULTS_DIR = "./results"

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def load_yaml(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None

def flatten_alpha_dict(alpha_data, run_id, source_label="Used_In_Sim"):
    rows = []
    if not alpha_data:
        return rows
    
    # Handle if alpha_data is nested under "alpha" key (common in GA output)
    if "alpha" in alpha_data and isinstance(alpha_data["alpha"], dict):
         alpha_data = alpha_data["alpha"]

    for mat_name, data in alpha_data.items():
        bands = data.get('bands_hz') or data.get('bands')
        alphas = data.get('alpha') or data.get('absorption_coefficients')
        
        if bands and alphas:
            for b, a in zip(bands, alphas):
                row = {
                    "Run_ID": run_id,
                    "Material": mat_name,
                    "Band_Hz": int(b),
                    f"Alpha_{source_label}": float(a)
                }
                rows.append(row)
    return rows

def analyze_single_run(folder_path):
    folder = Path(folder_path)
    run_name = folder.name
    
    # 1. Load Files
    files = {
        "receipt": load_json(folder / "run_receipt.json"),
        "sim_info": load_json(folder / "simulation_info.json"),
        "stage1_alpha": load_json(folder / "stage1_alpha.json"),
        "best_stage1": load_json(folder / "best_individual_stageA.json"),
        "yaml": load_yaml(folder / "project.yaml"),
    }
    
    # Load CSVs
    joined_csv_path = folder / "joined.csv"
    joined_df = pd.read_csv(joined_csv_path) if joined_csv_path.exists() else pd.DataFrame()

    # --- METRIC 1: Operational Stats (Overview) ---
    overview = {
        "Run_ID": run_name,
        "Timestamp": files["receipt"].get("timestamp", "N/A") if files["receipt"] else "N/A",
        "Loss_Huber": files["receipt"].get("loss", np.nan) if files["receipt"] else np.nan,
        "Tokens_Total": 0,
        "Compute_Cost_Credit": 0,
        "Sim_Duration_Sec": 0
    }
    
    if files["sim_info"]:
        if "token_usage" in files["sim_info"] and len(files["sim_info"]["token_usage"]) > 0:
            overview["Tokens_Total"] = files["sim_info"]["token_usage"][0].get("total_tokens", 0)
        overview["Compute_Cost_Credit"] = files["sim_info"].get("total_cost", 0)
        
        t_start = files["sim_info"].get("timings", {}).get("submitted_at")
        t_end = files["sim_info"].get("timings", {}).get("finished_at")
        if t_start and t_end:
            try:
                fmt = "%Y-%m-%dT%H:%M:%S.%f"
                # Handle potential timezone 'Z' or offset if necessary, strict strip here:
                t_start = t_start.split('+')[0].replace('Z','')
                t_end = t_end.split('+')[0].replace('Z','')
                dur = datetime.strptime(t_end, fmt) - datetime.strptime(t_start, fmt)
                overview["Sim_Duration_Sec"] = dur.total_seconds()
            except Exception as e:
                pass

    # --- METRIC 2: Alpha Extraction ---
    # We extract what was actually used in this specific run
    used_alphas_rows = flatten_alpha_dict(files["stage1_alpha"], run_name, "Used")
    df_alphas = pd.DataFrame(used_alphas_rows)

    # Note: We don't merge YAML/Best here to keep the dataframe size manageable. 
    # We can add the YAML baseline once at the end of the main loop.

    # --- METRIC 3: Error Stats ---
    df_error_stats = pd.DataFrame()
    if not joined_df.empty:
        if 'error' not in joined_df.columns:
            joined_df['error'] = joined_df['prediction'] - joined_df['target']
        
        joined_df['abs_error'] = joined_df['error'].abs()
        
        # Summarize by metric (scalar average across bands per metric)
        # Creates: Run_ID | Metric | MAE
        df_error_stats = joined_df.groupby(['metric'])[['abs_error']].mean().reset_index()
        df_error_stats['Run_ID'] = run_name
        
        # Also keep the full joined data but add Run_ID
        joined_df['Run_ID'] = run_name

    return overview, df_alphas, df_error_stats, joined_df

def main():
    parser = argparse.ArgumentParser(description="Post-process Batch Treble Sim Results")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Name string to search for")
    parser.add_argument("--input_dir", default=RESULTS_DIR, help="Root folder containing result subfolders")
    args = parser.parse_args()

    search_path = Path(args.input_dir)
    # Find all folders matching pattern
    all_folders = [f for f in search_path.iterdir() if f.is_dir() and args.pattern in f.name]
    
    if not all_folders:
        print(f"No folders found matching '{args.pattern}' in {args.input_dir}")
        return

    print(f"Found {len(all_folders)} runs. Analyzing batch...")

    all_overviews = []
    all_alphas = []
    all_error_stats = []
    all_joined_data = []

    # Loop through ALL folders
    for i, folder in enumerate(all_folders):
        print(f"[{i+1}/{len(all_folders)}] Analyzing {folder.name}...")
        try:
            ov, al, err, jd = analyze_single_run(folder)
            
            # Only append if data exists (avoid crashing on empty failed runs)
            if ov: all_overviews.append(ov)
            if not al.empty: all_alphas.append(al)
            if not err.empty: all_error_stats.append(err)
            if not jd.empty: all_joined_data.append(jd)
        except Exception as e:
            print(f"  !! Error processing {folder.name}: {e}")

    # --- Consolidation ---
    
    # 1. Master Overview (League Table)
    df_overview = pd.DataFrame(all_overviews)
    if not df_overview.empty and "Loss_Huber" in df_overview.columns:
        # Sort by Loss (Ascending = Lower is Better)
        df_overview.sort_values(by="Loss_Huber", ascending=True, inplace=True)

    # 2. Master Alpha Table
    df_all_alphas = pd.concat(all_alphas, ignore_index=True) if all_alphas else pd.DataFrame()
    # Pivot to make it readable: Rows = Material/Band, Columns = Run_IDs
    if not df_all_alphas.empty:
        # Optional: Pivot strictly on alpha values if you want a wide matrix
        # For now, let's keep it long format but ensure it's sorted by Run Performance if possible
        pass

    # 3. Master Error Stats
    df_all_errors = pd.concat(all_error_stats, ignore_index=True) if all_error_stats else pd.DataFrame()
    
    # 4. Master Detailed Data (Optional - might be large)
    df_all_joined = pd.concat(all_joined_data, ignore_index=True) if all_joined_data else pd.DataFrame()

    # --- Excel Export ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"Batch_Analysis_{args.pattern.strip('_')}_{timestamp}.xlsx"
    
    print(f"Writing report to {output_filename}...")
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Sheet 1: League Table (Overview)
        df_overview.to_excel(writer, sheet_name='League_Table', index=False)
        
        # Sheet 2: Alpha Evolution
        if not df_all_alphas.empty:
            df_all_alphas.to_excel(writer, sheet_name='All_Alphas', index=False)
            
        # Sheet 3: Error Metrics
        if not df_all_errors.empty:
            # Pivot for readability: Run_ID vs Metric
            pivot_err = df_all_errors.pivot(index="Run_ID", columns="metric", values="abs_error")
            pivot_err.to_excel(writer, sheet_name='Error_Matrix')
            
        # Sheet 4: Detailed Data (First 10000 rows only to prevent Excel crash if huge)
        if not df_all_joined.empty:
            df_all_joined.head(10000).to_excel(writer, sheet_name='Detailed_Data_Top10k', index=False)

    print("Batch analysis complete.")

if __name__ == "__main__":
    main()