
# usage:
# python .\src\postprocess_stageA_v3.py --pattern "_calib_008_A_" --input_dir "C:\Users\usuario\Documents\TrebleSDK\v2\results" --project_config "C:\Users\usuario\Documents\TrebleSDK\v2\configs\project.yaml" --stage1_best "C:\Users\usuario\Documents\TrebleSDK\v2\results\seed_run_006_20251119_174112\best_individual_stageA.json" --output_excel "C:\Users\usuario\Documents\TrebleSDK\v2\results\postprocess_calib008A.xlsx"


import os
import json
from pathlib import Path
from datetime import datetime
import argparse

import yaml
import pandas as pd


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

DEFAULT_PATTERN = "_calib_008_A_"
DEFAULT_RESULTS_DIR = "./results"
DEFAULT_OUTPUT_XLSX = "postprocess_batch_stageA.xlsx"


# --------------------------------------------------------------------
# Basic loaders
# --------------------------------------------------------------------

def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_yaml(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------
# Joined CSV loader (normalise schema)
# --------------------------------------------------------------------

def load_joined_csv(path: Path) -> pd.DataFrame:
    """
    Normalise joined.csv to a common schema:
      columns: receiver, metric, band_hz, target, prediction, error, abs_error
    """
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Handle older schema if ever encountered
    if "band_hz" not in df.columns and "f_hz" in df.columns:
        df = df.rename(columns={"f_hz": "band_hz"})
    if "target" not in df.columns and "target_val" in df.columns:
        df = df.rename(columns={"target_val": "target"})
    if "prediction" not in df.columns and "predicted_val" in df.columns:
        df = df.rename(columns={"predicted_val": "prediction"})

    if "error" not in df.columns:
        df["error"] = df["prediction"] - df["target"]

    df["abs_error"] = df["error"].abs()
    return df


# --------------------------------------------------------------------
# Alpha helpers
# --------------------------------------------------------------------

def project_yaml_to_alpha_dict(project_cfg: dict) -> dict:
    """
    Convert project.yaml material anchors into:
      { material_name: { "bands_hz": [...], "alpha": [...] } }
    """
    out = {}
    if not project_cfg:
        return out

    bands = project_cfg.get("bands", {}).get("f_hz", [])
    materials = project_cfg.get("materials", {})

    for mat_name, mat_cfg in materials.items():
        anchors = mat_cfg.get("anchors", {})
        alpha = anchors.get("absorption")
        if alpha is None:
            continue
        out[mat_name] = {
            "bands_hz": list(bands),
            "alpha": list(alpha),
        }
    return out


def flatten_alpha_dict(alpha_dict: dict, run_id: str, source_label: str):
    """
    Convert:
      { mat: { "bands_hz": [...], "alpha": [...] } }
    into a list of rows suitable for a dataframe.
    """
    rows = []
    if not alpha_dict:
        return rows

    for mat, data in alpha_dict.items():
        bands = data.get("bands_hz", [])
        alphas = data.get("alpha", [])
        for f, a in zip(bands, alphas):
            rows.append(
                {
                    "Run_ID": run_id,
                    "Source": source_label,
                    "Material": mat,
                    "Band_Hz": f,
                    "Alpha": a,
                }
            )
    return rows


# --------------------------------------------------------------------
# Simulation info timing
# --------------------------------------------------------------------

def extract_runtime_seconds(sim_info: dict) -> float:
    """
    Estimate runtime from createdAt / updatedAt in simulation_info.json.
    """
    if not sim_info:
        return 0.0
    t_start = sim_info.get("createdAt")
    t_end = sim_info.get("updatedAt")
    if not (t_start and t_end):
        return 0.0

    try:
        # Handle ISO with Z suffix
        t_start = t_start.replace("Z", "")
        t_end = t_end.replace("Z", "")
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        dt_start = datetime.strptime(t_start, fmt)
        dt_end = datetime.strptime(t_end, fmt)
        return (dt_end - dt_start).total_seconds()
    except Exception:
        return 0.0


# --------------------------------------------------------------------
# Per-run analysis
# --------------------------------------------------------------------

def analyse_single_run(run_folder: Path,
                       project_alpha_dict: dict,
                       stage1_best_dict: dict):
    """
    Analyse a single GA_calib_* run folder.
    Returns:
      overview_row (dict),
      alpha_rows (list of dict),
      error_stats_df (DataFrame),
      joined_df (DataFrame with Run_ID column)
    """
    run_name = run_folder.name

    # Required files
    run_receipt = load_json(run_folder / "run_receipt.json")
    sim_info = load_json(run_folder / "simulation_info.json")
    stage1_alpha = load_json(run_folder / "stage1_alpha.json")
    joined_df = load_joined_csv(run_folder / "joined.csv")

    if run_receipt is None:
        raise RuntimeError(f"Missing run_receipt.json in {run_folder}")

    # --- Overview row ---
    overview = {
        "Run_ID": run_name,
        "Run_Label": run_receipt.get("run_label"),
        "Timestamp": run_receipt.get("timestamp"),
        "Loss_Huber": run_receipt.get("loss"),
        "Bands_Hz": ",".join(str(b) for b in run_receipt.get("bands_hz", [])),
        "Metrics": ",".join(run_receipt.get("metrics", [])),
        "Receivers": ",".join(run_receipt.get("receivers", [])),
        "Sim_Runtime_Sec": extract_runtime_seconds(sim_info),
    }

    # --- Alpha rows ---
    alpha_rows = []

    # YAML anchors (baseline)
    yaml_rows = flatten_alpha_dict(project_alpha_dict, run_name, "YAML")
    alpha_rows.extend(yaml_rows)

    # Stage-1 GA best (global)
    alpha_rows.extend(flatten_alpha_dict(stage1_best_dict, run_name, "Stage1Best"))

    # Stage-1 alpha actually used in THIS run
    alpha_rows.extend(flatten_alpha_dict(stage1_alpha or {}, run_name, "Used"))

    # --- Error stats per metric ---
    error_stats_df = pd.DataFrame()
    if not joined_df.empty:
        # group by metric, compute MAE over all receivers & bands
        error_stats_df = (
            joined_df.groupby("metric")["abs_error"]
            .mean()
            .reset_index()
            .rename(columns={"abs_error": "MAE_abs_error"})
        )
        error_stats_df.insert(0, "Run_ID", run_name)

        # tag joined rows with Run_ID for detailed sheet
        joined_df = joined_df.copy()
        joined_df.insert(0, "Run_ID", run_name)

    return overview, alpha_rows, error_stats_df, joined_df


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-process Treble GA Stage A batch results")
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Substring to search for in run folder names (e.g. '_calib_008_A_')",
    )
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_RESULTS_DIR,
        help="Root folder containing GA_calib_* result subfolders",
    )
    parser.add_argument(
        "--project_config",
        required=True,
        help="Path to project.yaml (for YAML anchors)",
    )
    parser.add_argument(
        "--stage1_best",
        required=True,
        help="Path to best_individual_stageA.json (global Stage-1 GA best alphas)",
    )
    parser.add_argument(
        "--output_excel",
        default=DEFAULT_OUTPUT_XLSX,
        help="Output Excel file name",
    )

    args = parser.parse_args()

    results_root = Path(args.input_dir).expanduser().resolve()
    if not results_root.exists():
        print(f"[ERROR] input_dir does not exist: {results_root}")
        return

    # Discover run folders matching pattern
    run_folders = [
        p for p in results_root.iterdir() if p.is_dir() and args.pattern in p.name
    ]
    if not run_folders:
        print(f"[WARN] No folders found in {results_root} containing pattern '{args.pattern}'")
        return

    print(f"[info] Found {len(run_folders)} run folders matching '{args.pattern}'")

    # Load global YAML + Stage1 best once
    project_cfg = load_yaml(Path(args.project_config).expanduser().resolve())
    if project_cfg is None:
        print(f"[ERROR] Could not load project.yaml from {args.project_config}")
        return
    project_alpha_dict = project_yaml_to_alpha_dict(project_cfg)

    stage1_best_dict = load_json(Path(args.stage1_best).expanduser().resolve())
    if stage1_best_dict is None:
        print(f"[ERROR] Could not load best_individual_stageA.json from {args.stage1_best}")
        return

    # Accumulators
    overview_rows = []
    all_alpha_rows = []
    all_error_stats = []
    all_joined_rows = []

    for run_folder in sorted(run_folders):
        try:
            print(f"[info] Analysing run folder: {run_folder.name}")
            overview, alpha_rows, err_df, joined_df = analyse_single_run(
                run_folder, project_alpha_dict, stage1_best_dict
            )
            overview_rows.append(overview)
            all_alpha_rows.extend(alpha_rows)
            if not err_df.empty:
                all_error_stats.append(err_df)
            if not joined_df.empty:
                all_joined_rows.append(joined_df)
        except Exception as e:
            print(f"[WARN] Skipping {run_folder.name} due to error: {e}")

    if not overview_rows:
        print("[WARN] No successful runs analysed.")
        return

    df_league = pd.DataFrame(overview_rows)
    df_league = df_league.sort_values(by="Loss_Huber", ascending=True)

    df_alphas = pd.DataFrame(all_alpha_rows) if all_alpha_rows else pd.DataFrame()
    df_errors = pd.concat(all_error_stats, ignore_index=True) if all_error_stats else pd.DataFrame()
    df_joined = pd.concat(all_joined_rows, ignore_index=True) if all_joined_rows else pd.DataFrame()

    # Pivot error stats: Run_ID vs metric
    df_error_matrix = pd.DataFrame()
    if not df_errors.empty:
        df_error_matrix = df_errors.pivot(index="Run_ID", columns="metric", values="MAE_abs_error")

    # Write Excel
    out_xlsx = Path(args.output_excel).expanduser().resolve()
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_league.to_excel(writer, sheet_name="League_Table", index=False)

        if not df_alphas.empty:
            df_alphas.to_excel(writer, sheet_name="All_Alphas", index=False)

        if not df_error_matrix.empty:
            df_error_matrix.to_excel(writer, sheet_name="Error_Matrix", index=True)

        if not df_joined.empty:
            df_joined.head(10000).to_excel(writer, sheet_name="Detailed_Data_Top10k", index=False)

    print(f"[info] Batch analysis complete. Excel saved to: {out_xlsx}")


if __name__ == "__main__":
    main()
