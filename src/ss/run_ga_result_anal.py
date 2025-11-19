import json
import pandas as pd
from pathlib import Path
import re
from typing import Dict, Any, List, Optional

# --- Configuration (Based on User Input) ---
# NOTE: The working folder is based on the user's input: 
# C:\Users\usuario\Documents\TrebleSDK\v2\configs\results
# We assume the GA run started one level higher, so the 'results' folder contains the run directories.
WORKING_DIR = Path("C:/Users/usuario/Documents/TrebleSDK/v2/configs/results")
# Define the path to the initial seed run log to manually extract Generation 0 data
# You need to manually replace the timestamp below with your actual seed run's name
INITIAL_SEED_DIR_NAME = "seed_run_001_20251107_161733" # Using the name from the error message
# We assume the single, non-GA run is stored directly inside the WORKING_DIR

# --- Helper Functions ---

def find_run_dirs(base_path: Path) -> List[Path]:
    """Finds all top-level GA run directories (e.g., seed_run_001_...)."""
    # Look for folders that match the typical timestamp pattern or contain 'seed_run'.
    
    # We are looking for folders like: 
    # C:\Users\usuario\Documents\TrebleSDK\v2\configs\results\seed_run_001_20251107_091608 (Full GA)
    # C:\Users\usuario\Documents\TrebleSDK\v2\configs\results\seed_run_001_20251107_161733 (Single Seed)
    
    run_folders = [p for p in base_path.iterdir() if p.is_dir() and re.search(r'seed_run_\d+_\d{8}_\d{6}', p.name)]
    
    if not run_folders:
        print(f"Warning: No GA run directories found in {base_path}")
        # Fallback search for any directory that looks like a GA run
        run_folders = [p for p in base_path.iterdir() if p.is_dir() and 'seed_run' in p.name.lower()]
        
    return run_folders

def load_json_safe(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely loads a JSON file."""
    try:
        if file_path.exists():
            return json.loads(file_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

def find_best_individual_log(run_dir: Path) -> Optional[Path]:
    """
    Finds the detailed log file of the overall best individual.
    This handles three structures: Full GA, Nested Single Run, and Direct Single Run.
    """
    
    # --- Attempt 0: Direct Single Run Structure (Common for Stage 1 Seed) ---
    # Path: run_dir / detailed_results_log.json
    detailed_log_direct_root = run_dir / "detailed_results_log.json"
    if detailed_log_direct_root.exists():
        print(f"Found log via direct root structure: {detailed_log_direct_root.relative_to(run_dir)}")
        return detailed_log_direct_root
        
    # --- Attempt 1: Full GA Structure (Post-GA run) ---
    final_params_path = run_dir / "final_parameters.json"
    if final_params_path.exists():
        final_params = load_json_safe(final_params_path)
        if final_params and 'eval_dir' in final_params:
            best_eval_dir_name = Path(final_params['eval_dir']).name 
            best_eval_path = run_dir / best_eval_dir_name
            if best_eval_path.is_dir():
                # Path: run_dir / eval_A_... / GA_calib_... / detailed_results_log.json
                sim_folders = [p for p in best_eval_path.iterdir() if p.is_dir() and 'GA_calib' in p.name]
                if sim_folders:
                    detailed_log = sim_folders[0] / "detailed_results_log.json"
                    if detailed_log.exists():
                        print(f"Found log via final_parameters.json: {detailed_log.relative_to(run_dir)}")
                        return detailed_log
    
    # --- Attempt 2: Nested Single Run Structure (Initial Seed or Standalone run_stage1_and_forward) ---
    # Path: run_dir / GA_calib_... / detailed_results_log.json
    sim_folders_direct = [p for p in run_dir.iterdir() if p.is_dir() and 'GA_calib' in p.name]
    if sim_folders_direct:
        detailed_log_direct = sim_folders_direct[0] / "detailed_results_log.json"
        if detailed_log_direct.exists():
            print(f"Found log via nested folder structure: {detailed_log_direct.relative_to(run_dir)}")
            return detailed_log_direct
            
    # --- Attempt 3: Fallback (Latest detailed log) ---
    all_logs = []
    # Search nested eval directories
    for eval_dir in run_dir.glob("eval_A_*"):
        sim_folders = [p for p in eval_dir.iterdir() if p.is_dir() and 'GA_calib' in p.name]
        if sim_folders:
            log_path = sim_folders[0] / "detailed_results_log.json"
            if log_path.exists():
                all_logs.append((log_path.stat().st_mtime, log_path))
    
    if all_logs:
        # Return the latest written log file
        latest_log = max(all_logs, key=lambda x: x[0])[1]
        print(f"Found log via Fallback search: {latest_log.relative_to(run_dir)}")
        return latest_log
        
    return None

# --- Main Processor ---

def aggregate_ga_data(run_dir: Path) -> pd.DataFrame:
    """Aggregates generation-wise loss and parameter data from all receipts."""
    
    receipt_data = []
    
    # 1. Gather all generation receipts from the top run directory (for Gen 1 to N)
    receipt_paths = sorted(run_dir.glob("receipt_gen_A_*.json"))
    
    for receipt_path in receipt_paths:
        receipt = load_json_safe(receipt_path)
        if not receipt:
            continue

        gen = receipt.get("gen", -1)
        # Note: best_loss_H and mean_loss_H are the correct keys for Stage A
        best_loss = receipt.get("best_loss_H", float('inf')) 
        mean_loss = receipt.get("mean_loss_H", float('inf'))
        
        row = {
            'Generation': gen,
            'Best_Loss_H': best_loss,
            'Mean_Loss_H': mean_loss,
        }
        receipt_data.append(row)

    # 1a. Create DataFrame for Gen 1 to N. We defer sorting until Gen 0 data is added.
    df_ga = pd.DataFrame(receipt_data)

    # --- Step 2: Manually find and insert Generation 0 (Initial Seed) data ---
    # The initial seed run is likely the separate folder mentioned by the user.
    seed_dir = WORKING_DIR / INITIAL_SEED_DIR_NAME 
    if seed_dir.is_dir():
        seed_receipt_path = seed_dir / "run_receipt.json"
        if seed_receipt_path.exists():
            seed_receipt = load_json_safe(seed_receipt_path)
            if seed_receipt and 'loss' in seed_receipt:
                seed_loss = seed_receipt['loss']
                seed_row = {
                    'Generation': 0,
                    'Best_Loss_H': seed_loss,
                    'Mean_Loss_H': seed_loss, # Assume best and mean are the same for the single seed run
                }
                
                # Prepend the seed row to the GA run data
                if not df_ga.empty:
                    df_ga_seed = pd.DataFrame([seed_row])
                    df_ga = pd.concat([df_ga_seed, df_ga], ignore_index=True)
                    print(f"Successfully added Generation 0 (Seed) data from {seed_dir.name}.")
                elif not receipt_data:
                    # If only the seed data exists (e.g., GA failed to start), use just the seed.
                    df_ga = pd.DataFrame([seed_row])
                    print(f"Only Generation 0 (Seed) data available.")
                
    if df_ga.empty:
        return pd.DataFrame()
    
    # --- Step 3: Now that the DataFrame is guaranteed to have data (and the 'Generation' column), sort it. ---
    df_ga = df_ga.sort_values('Generation').reset_index(drop=True)
        
    return df_ga

def process_acoustic_match_data(run_dir: Path, is_seed_run: bool = False) -> pd.DataFrame:
    """Processes the detailed_results_log.json for final metric match and parameters."""
    
    log_path = find_best_individual_log(run_dir)
    if not log_path:
        print(f"Skipping acoustic metrics processing for {run_dir.name}: Best individual log not found.")
        return pd.DataFrame()

    detailed_log = load_json_safe(log_path)
    if not detailed_log:
        # This occurs if the JSON file exists but is empty or malformed.
        print(f"Warning: Log file found at {log_path}, but content is empty or invalid.")
        return pd.DataFrame()

    # The log is a list of dictionaries, one per metric/band/receiver combination
    df_metrics = pd.DataFrame(detailed_log)
    
    # Define the mapping from the ACTUAL JSON keys in your file to desired DataFrame columns
    COLUMN_MAP = {
        'predicted_val': 'Predicted', # <-- Corrected key based on uploaded JSON
        'target_val': 'Target',       # <-- Corrected key based on uploaded JSON
        'metric': 'Metric',
        'f_hz': 'Band_Hz',            # <-- Corrected key based on uploaded JSON
        'rcv_code': 'Receiver'
    }

    # --- Robustness Check ---
    # Check for required source columns before renaming/calculation
    required_source_cols = ['predicted_val', 'target_val'] # <-- Corrected to match uploaded JSON
    missing_cols = [col for col in required_source_cols if col not in df_metrics.columns]

    if missing_cols:
        print(f"FATAL ERROR in {run_dir.name}: Missing required columns in detailed_results_log.json: {missing_cols}")
        print(f"Available columns are: {list(df_metrics.columns)}")
        print("Please check the schema of the JSON log file.")
        return pd.DataFrame() # Return empty DataFrame on failure
    # ---------------------------------------------

    # Add Source (Initial Seed vs. Calibrated)
    df_metrics['Source'] = 'Initial_Seed' if is_seed_run else 'Calibrated_GA'
    
    # Rename columns for clarity (now guaranteed to work if the check passed)
    df_metrics = df_metrics.rename(columns=COLUMN_MAP)
    
    # Calculate Error
    df_metrics['Error'] = df_metrics['Predicted'] - df_metrics['Target']
    df_metrics['Abs_Error'] = df_metrics['Error'].abs()
    
    return df_metrics


def process_final_parameters(run_dir: Path) -> Optional[Dict]:
    """Extracts the final alpha and scatter parameters from the best individual."""
    
    # --- Attempt 1: Full GA Structure (Post-GA run) ---
    final_params_path = run_dir / "final_parameters.json"
    if final_params_path.exists():
        print(f"Found final parameters at: {final_params_path.name}")
        return load_json_safe(final_params_path)

    # --- Attempt 2: Single Seed Structure (Initial Seed) ---
    # For a single seed run, the alpha parameters are in 'stage1_alpha.json'
    # and scattering is usually 0.0 or fixed in the config.
    seed_alpha_path = run_dir / "stage1_alpha.json"
    if seed_alpha_path.exists():
        alpha_params = load_json_safe(seed_alpha_path)
        # We need the scatter values, which are not stored here. 
        # A full GA run is required to get the calibrated scatter.
        print(f"Found seed alpha parameters at: {seed_alpha_path.name}")
        return {"alpha": alpha_params, "scatter": {}} # Return partial data
        
    return None


# --- Execution ---

def main():
    
    print(f"--- Starting GA Results Analysis ---")
    print(f"Searching in: {WORKING_DIR.resolve()}")
    
    run_dirs = find_run_dirs(WORKING_DIR)
    
    if not run_dirs:
        print("No valid GA run directories found. Exiting.")
        return

    # --- Identify the GA Run (Generations 1 to N) and the Seed Run (Generation 0) ---
    # Assuming the most recent/largest folder is the full GA run
    full_ga_run_dir = max(run_dirs, key=lambda p: len(list(p.glob("receipt_gen_A_*.json"))))
    
    # The user provided the name of the troubled seed run explicitly
    initial_seed_run_dir = WORKING_DIR / INITIAL_SEED_DIR_NAME 
    
    if not full_ga_run_dir.exists():
        print("Error: Could not find the main GA run directory.")
        return
        
    print(f"Main GA Run (Gen 1 to N): {full_ga_run_dir.name}")
    print(f"Initial Seed Run (Gen 0): {initial_seed_run_dir.name}")
    
    # IMPORTANT NOTE: If Main GA Run and Initial Seed Run names are identical, 
    # it means the script failed to find a full multi-generation GA run folder 
    # (one containing receipt_gen_A_*.json files). Please verify your folder names.
    if full_ga_run_dir.name == initial_seed_run_dir.name:
        print("NOTE: Since 'Main GA Run' and 'Initial Seed Run' are the same, the script assumes the GA run didn't finish or receipts are missing.")

    
    # --- Part 1: GA Convergence Data (for Graph 1 & 2) ---
    # This function now aggregates Gen 1 to N from the full_ga_run_dir 
    # AND prepends Gen 0 data from the initial_seed_run_dir.
    df_ga_summary = aggregate_ga_data(full_ga_run_dir)
    
    if not df_ga_summary.empty:
        df_ga_summary.to_csv(WORKING_DIR / "ga_summary_data.csv", index=False)
        print(f"SUCCESS: Generated ga_summary_data.csv (Rows: {len(df_ga_summary)})")
    else:
        print("FATAL WARNING: Could not aggregate GA convergence data, Graph 1 cannot be created.")
        
    # --- Part 2: Final Acoustic Metric Match Data (for Graph 3 & 4 and Table 2) ---
    # We need to process both the Calibrated GA result and the Initial Seed result
    
    # A. Initial Seed Data (Source = 'Initial_Seed')
    df_seed_metrics = process_acoustic_match_data(initial_seed_run_dir, is_seed_run=True)

    # B. Calibrated GA Data (Source = 'Calibrated_GA')
    df_ga_metrics = process_acoustic_match_data(full_ga_run_dir, is_seed_run=False)
    
    # Combine the two dataframes
    if not df_seed_metrics.empty and not df_ga_metrics.empty:
        df_acoustic_metrics = pd.concat([df_seed_metrics, df_ga_metrics], ignore_index=True)
        df_acoustic_metrics.to_csv(WORKING_DIR / "acoustic_metrics_data.csv", index=False)
        print(f"SUCCESS: Generated acoustic_metrics_data.csv (Rows: {len(df_acoustic_metrics)}) combining Seed and GA results.")
    elif not df_seed_metrics.empty:
        # If the GA run is missing/empty, we still save the seed metrics
        df_seed_metrics.to_csv(WORKING_DIR / "acoustic_metrics_data.csv", index=False)
        print("SUCCESS: Generated acoustic_metrics_data.csv with Initial Seed data only.")
    elif not df_ga_metrics.empty:
        # Only the GA result is complete (shouldn't happen if seed log is found)
        df_ga_metrics.to_csv(WORKING_DIR / "acoustic_metrics_data.csv", index=False)
        print("WARNING: Only Calibrated GA data available for acoustic metrics.")
    else:
        print("FATAL WARNING: Could not aggregate final acoustic metrics data, Graphs 3/4 and Table 2 cannot be created.")
        
    # --- Part 3: Final Parameters (for Table 1) ---
    final_params = process_final_parameters(full_ga_run_dir)
    if final_params:
        (WORKING_DIR / "final_calibrated_params.json").write_text(
            json.dumps(final_params, indent=2)
        )
        print("SUCCESS: Generated final_calibrated_params.json")
    else:
        # Since we found seed alpha parameters, this is misleading.
        print("FATAL WARNING: Could not extract final parameters (beyond initial seed), Table 1 may be incomplete.")


if __name__ == "__main__":
    # To run this script, you will need to ensure your Python environment has pandas.
    # If this script is run within the system, you may need to adjust the path separators.
    # Since this is a utility script to be run by the user, we keep it standard.
    main()