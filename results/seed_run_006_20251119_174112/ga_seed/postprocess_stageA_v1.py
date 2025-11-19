

# postprocess_stageA.py

# Post-processing for Treble GA Stage A calibrations.

# - Picks the latest GA run folder whose name contains a pattern (e.g. "_calib_008_A_")
# - Loads:
#   - project.yaml (initial anchors / bands)
#   - best_individual_stageA.json (Stage-1 GA best alphas)
#   - stage1_alpha.json (alphas actually used in THIS forward run)
#   - run_receipt.json (basic loss, bands, receivers)
#   - simulation_info.json (tokens + time from Treble tasks)
#   - joined.csv / targets_tidy.csv / pred_tidy.csv (metric tables)
#   - detailed_results_log.json (verbose errors)

# Outputs a human-readable summary to stdout and optionally to JSON.


import argparse
import json
import csv
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_run(results_root: Path, name_pattern: str) -> Path:
    """
    Find latest subfolder under results_root whose name contains name_pattern.
    Assumes folder names end with _YYYYMMDD_HHMMSS.

    e.g. GA_calib_008_A_20251119_213503
    """
    candidates: List[Tuple[datetime, Path]] = []

    for p in results_root.iterdir():
        if not p.is_dir():
            continue
        if name_pattern not in p.name:
            continue

        # Grab the last 15 chars "YYYYMMDD_HHMMSS" if possible
        parts = p.name.split("_")
        if len(parts) < 3:
            continue
        stamp = "_".join(parts[-2:])  # e.g. 20251119_213503
        try:
            dt = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        except ValueError:
            # Fallback to mtime
            dt = datetime.fromtimestamp(p.stat().st_mtime)

        candidates.append((dt, p))

    if not candidates:
        raise RuntimeError(f"No run folders under {results_root} matching pattern '{name_pattern}'")

    candidates.sort(key=lambda tup: tup[0])
    latest_dt, latest_path = candidates[-1]
    print(f"[info] Latest run folder: {latest_path} (timestamp={latest_dt.isoformat()})")
    return latest_path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_project_yaml(project_path: Path) -> Dict[str, Any]:
    with project_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Section: materials / alphas
# ---------------------------------------------------------------------------

def extract_bands_from_project(project_cfg: Dict[str, Any]) -> List[int]:
    bands = project_cfg.get("bands", {})
    return bands.get("f_hz", [])


def extract_material_anchors(project_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Return dict:
      material_name -> { 'bands_hz': [...], 'alpha_anchors': [...], 'scatter_anchors': [...] }
    """
    out = {}
    materials = project_cfg.get("materials", {})
    bands = extract_bands_from_project(project_cfg)

    for mat_name, mat_cfg in materials.items():
        anchors = mat_cfg.get("anchors", {})
        abs_anchors = anchors.get("absorption", None)
        scat_anchors = anchors.get("scattering", None)
        if abs_anchors is None:
            continue
        out[mat_name] = {
            "bands_hz": bands,
            "alpha_anchors": abs_anchors,
            "scatter_anchors": scat_anchors,
        }
    return out


def pretty_print_alpha_table(title: str, alpha_dict: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    for mat_name, mat_data in alpha_dict.items():
        bands = mat_data.get("bands_hz", [])
        alpha_vals = mat_data.get("alpha", mat_data.get("alpha_anchors", []))
        print(f"\n-- {mat_name} --")
        for f, a in zip(bands, alpha_vals):
            print(f"  {f:4d} Hz : alpha = {a:.3f}")


def json_to_alpha_table(json_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert stage1_alpha.json / best_individual_stageA.json schema into
    a simpler { material: { 'bands_hz': [...], 'alpha': [...] } }.
    """
    out = {}
    for mat_name, mat_data in json_dict.items():
        out[mat_name] = {
            "bands_hz": mat_data.get("bands_hz", []),
            "alpha": mat_data.get("alpha", []),
        }
    return out


# ---------------------------------------------------------------------------
# Section: tokens + time from simulation_info.json
# ---------------------------------------------------------------------------

def extract_tokens_and_runtime(sim_info: Dict[str, Any]) -> Tuple[float, float]:
    """
    Sum tokensCost across tasks and estimate runtime in seconds
    from earliest createdAt to latest completedAt.
    """
    sources = sim_info.get("sources", [])
    token_sum = 0.0
    times: List[datetime] = []

    for src in sources:
        for task in src.get("tasks", []):
            tokens = task.get("tokensCost")
            if tokens is not None:
                token_sum += float(tokens)
            for key in ("createdAt", "startedAt", "completedAt"):
                t_str = task.get(key)
                if t_str:
                    try:
                        times.append(datetime.fromisoformat(t_str.replace("Z", "+00:00")))
                    except Exception:
                        pass

    runtime = 0.0
    if times:
        runtime = (max(times) - min(times)).total_seconds()

    return token_sum, runtime


# ---------------------------------------------------------------------------
# Section: metric tables (joined.csv / detailed_results_log.json)
# ---------------------------------------------------------------------------

def load_joined_csv(joined_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with joined_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise numeric fields
            for key in ("f_hz", "target_val", "predicted_val", "error"):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def band_stats(joined_rows: List[Dict[str, Any]]) -> Dict[float, Dict[str, float]]:
    """
    Compute MAE and RMSE across all receivers per frequency band.
    """
    by_band: Dict[float, List[float]] = {}
    for r in joined_rows:
        band = float(r["f_hz"])
        err = float(r["error"])
        by_band.setdefault(band, []).append(err)

    stats = {}
    for f, errs in by_band.items():
        mae = statistics.fmean(abs(e) for e in errs)
        rmse = (statistics.fmean(e * e for e in errs)) ** 0.5
        stats[f] = {"mae": mae, "rmse": rmse, "n": len(errs)}
    return stats


def pretty_print_band_stats(stats: Dict[float, Dict[str, float]]) -> None:
    print("\n=== Band error stats (all receivers) ===")
    print("  f_hz   |   MAE    |  RMSE   |  N")
    print("---------+---------+--------+-----")
    for f in sorted(stats.keys()):
        s = stats[f]
        print(f" {int(f):5d}  | {s['mae']:.4f} | {s['rmse']:.4f} | {s['n']:3d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-process Treble GA Stage A run")
    parser.add_argument("--results_root", required=True, help="Folder containing GA_calib_* run folders")
    parser.add_argument("--pattern", required=True, help="Substring to match run folder names, e.g. _calib_008_A_")
    parser.add_argument("--project_config", required=True, help="Path to project.yaml")
    parser.add_argument("--stage1_best", required=True, help="Path to best_individual_stageA.json")
    parser.add_argument("--output_json", help="Optional path to save a JSON summary")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    project_path = Path(args.project_config).expanduser().resolve()
    stage1_best_path = Path(args.stage1_best).expanduser().resolve()

    latest_run = find_latest_run(results_root, args.pattern)

    # Load run-level files
    run_receipt = load_json(latest_run / "run_receipt.json")
    sim_info = load_json(latest_run / "simulation_info.json")
    stage1_alpha = load_json(latest_run / "stage1_alpha.json")
    detailed_log = load_json(latest_run / "detailed_results_log.json")
    joined_rows = load_joined_csv(latest_run / "joined.csv")

    # Global config / stage-1 best
    project_cfg = parse_project_yaml(project_path)
    stage1_best = load_json(stage1_best_path)

    # ---- Assemble report ----
    report: Dict[str, Any] = {}

    # A) Basic run info
    report["run"] = {
        "run_folder": str(latest_run),
        "project": run_receipt.get("project"),
        "model": run_receipt.get("model"),
        "run_label": run_receipt.get("run_label"),
        "timestamp": run_receipt.get("timestamp"),
        "loss": run_receipt.get("loss"),
        "bands_hz": run_receipt.get("bands_hz"),
        "metrics": run_receipt.get("metrics"),
        "receivers": run_receipt.get("receivers"),
        "results_dir": run_receipt.get("results_dir"),
    }

    tokens, runtime = extract_tokens_and_runtime(sim_info)
    report["tokens_runtime"] = {
        "tokens_estimate": tokens,
        "runtime_seconds_estimate": runtime,
    }

    # B) Alpha tables
    anchors = extract_material_anchors(project_cfg)
    stage1_anchors = {
        mat: {"bands_hz": v["bands_hz"], "alpha_anchors": v["alpha_anchors"]}
        for mat, v in anchors.items()
    }

    stage1_alpha_tbl = json_to_alpha_table(stage1_alpha)
    stage1_best_tbl = json_to_alpha_table(stage1_best)

    report["alphas"] = {
        "from_project_yaml": stage1_anchors,
        "stage1_alpha_this_run": stage1_alpha_tbl,
        "stage1_best_ga": stage1_best_tbl,
    }

    # C) Metric stats
    stats = band_stats(joined_rows)
    report["metrics"] = {
        "band_stats": stats,
        "n_rows": len(joined_rows),
    }

    # D) Detailed log passthrough (if you want to inspect later)
    report["detailed_results"] = detailed_log

    # ---- Pretty print key info to console ----
    print("\n=== RUN SUMMARY ===")
    print(f"Run folder:  {report['run']['run_folder']}")
    print(f"Run label:   {report['run']['run_label']}")
    print(f"Loss:        {report['run']['loss']:.6f}")
    print(f"Bands (Hz):  {report['run']['bands_hz']}")
    print(f"Receivers:   {report['run']['receivers']}")
    print(f"Tokens (≈):  {tokens:.4f}")
    print(f"Runtime (s): {runtime:.1f}")

    print("\n=== MATERIAL ALPHAS ===")
    pretty_print_alpha_table("Initial anchors from project.yaml", {
        k: {"bands_hz": v["bands_hz"], "alpha": v["alpha_anchors"]}
        for k, v in anchors.items()
    })
    pretty_print_alpha_table("Stage-1 GA BEST (best_individual_stageA)", stage1_best_tbl)
    pretty_print_alpha_table("Stage-1 alpha used in THIS run (stage1_alpha.json)", stage1_alpha_tbl)

    # Metrics
    pretty_print_band_stats(stats)

    # Optional JSON dump
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[info] JSON summary saved to: {out_path}")


if __name__ == "__main__":
    main()
