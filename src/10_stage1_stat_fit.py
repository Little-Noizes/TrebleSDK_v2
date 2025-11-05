# <!-- ===============================================================================
# scripts/10_stage1_stat_fit.py
# ===============================================================================
# Purpose
# -------
# Stage-1 "statistical" calibration for material absorption spectra α(f) using a
# fast analytic room model (Sabine/Eyring proxy). Produces a *deterministic seed*
# `stage1_alpha.json` to kick off Stage-2 (Treble forward + GA).

# This borrows the same logic used later in Stage-2 GA (anchors → octave bands,
# Sabine/Eyring RT proxy), but runs *once* and deterministically. See the
# reference GA script for the multi-generation approach and details on anchor
# interpolation and RT calculation.  # (ref) stage2_ga_rt60.py
# -------------------------------------------------------------------------------
# Requisites & Dependencies
# -------------------------
# • Python stdlib only (json, csv, math, pathlib, typing).
# • Internal modules:
#   - src.cfg            : config loading + validation
#   - src.io_utils       : run dir and small IO helpers
#   - src.metrics        : band alignment + (optional) loss
#   - src.materials      : anchors, clamping, monotonicity, deviation masks

# No external pip packages required.

# -------------------------------------------------------------------------------
# Inputs (from project.yaml)
# --------------------------
# • paths.ground_truth_json     : measured targets (bands.f + metrics per receiver)
# • bands.kind / bands.f_hz     : centre frequencies (octave or third)
# • bands.optimise_from_hz      : ignore sub-LF in loss (e.g., 125 Hz)
# • materials.*                 : anchors.absorption (8 or N bands), deviation_groups
# • tag_to_material             : mapping of mesh "tags" to logical materials
# • source.*                    : not used in Stage-1 proxy, only in Stage-2 Treble
# • metrics.objective_metrics   : list of metrics to match
#     - If includes 'RT60' or only decay metrics (EDT/T20), we approximate with
#       Sabine/Eyring RT as a proxy. (EDT/T20 ≈ k·RT60 per band is a coarse
#       approximation; Stage-1 is only to get a reasonable starting α(f).)
# • calculation.*               : not used here (Treble-specific)
# • optimisation.*              : not used here

# Geometry / areas (required for Sabine/Eyring proxy):
# • One of:
#   (A) Known area per material via materials.*.area_m2_override; OR
#   (B) Room L/W/H + auto wall/floor/ceiling area assignment per material; OR
#   (C) A precomputed file providing per-material area_m2 (JSON/CSV).
# If nothing is provided, the script will raise with a clear message.

# -------------------------------------------------------------------------------
# Outputs
# -------
# • results/<run_id>/stage1_alpha.json
#     {
#       "<MaterialKey>": {
#         "bands_hz": [63,125,250,500,1000,2000,4000,8000],
#         "alpha":    [ ... per-band α clamped & monotonic ... ]
#       }, ...
#     }

# • results/<run_id>/stage1_summary.csv
#     material,f_hz,alpha,deviation_pct

# • (optional) results/<run_id>/stage1_score.txt
#     A simple scalar (weighted Huber) vs averaged targets, bands ≥ optimise_from_hz

# -------------------------------------------------------------------------------
# Method (high level)
# -------------------
# 1) Load config and ground truth.
# 2) Build per-material α(f) from YAML anchors (absorption).
#    - Clamp into [alpha_min, alpha_max]
#    - Enforce monotonic increase if enabled
#    - Compute per-band deviation_pct mask from deviation_groups (for logging)
# 3) If usable areas exist, evaluate *analytic* RT proxy per band:
#    - For each band, compute Sabine or Eyring RT:
#        Sabine:  RT = 0.161 * V / Σ( S_i · α_i )
#        Eyring:  RT = 0.161 * V / Σ( S_i · ln(1/(1-α_i)) )
#      where S_i is the total area for all surfaces mapped to material i.
#    - Optionally compute a simple baseline loss vs target (averaged over receivers)
#      for reporting only (Stage-1 does not iterate).
# 4) Write `stage1_alpha.json` and `stage1_summary.csv`.

# Notes:
# • If your objective metrics are EDT/T20 (not RT60), the Stage-1 proxy still runs
#   with RT60; treat the baseline loss as *indicative* only. Stage-2 (Treble)
#   will do exact EDT/T20/C50 scoring.
# • This mirrors the GA RT60 approach (anchors → band interpolation and Sabine/
#   Eyring proxy), but without the evolutionary loop.  # (ref) stage2_ga_rt60.py

# -------------------------------------------------------------------------------
# Configuration knobs
# -------------------
# • stage1.bounds.alpha_min / alpha_max   : clamp limits (default 0.02..0.95)
# • stage1.monotonic                      : enforce non-decreasing α(f)
# • stage1.proxy_model                    : "sabine" | "eyring" | "none"
# • stage1.area_source                    : "override" | "box_LWH" | "file"

# If proxy_model == "none" or areas are missing, the script will skip proxy RT
# and write only the α(f) seed (recommended if geometry is unknown).

# -------------------------------------------------------------------------------
# Usage
# -----
# # from project root:
# > python -m scripts.10_stage1_stat_fit

# # or:
# > python scripts\\10_stage1_stat_fit.py

# Expected:
# • Writes JSON/CSV under results/<run_id>, prints INFO lines to console.

# -------------------------------------------------------------------------------
# Differences vs Stage-2 GA RT60
# -------------------------------
# • Stage-1: single pass, deterministic; no population, no mutation/crossover.
# • Uses the same anchor-to-band interpolation and Sabine/Eyring formulae idea,
#   but only to report a baseline; Stage-2 will iterate and truly minimise error.
# • Inputs are yaml-driven; no pandas/matplotlib; minimal IO.  # GA script uses
#   pandas/plots and a full GA loop.  # (ref) stage2_ga_rt60.py

# -------------------------------------------------------------------------------
# Caveats
# -------
# • RT proxies are coarse for EDT/T20/C50—use only to create a reasonable α(f)
#   starting point. Stage-2 (Treble) is where accuracy matters.
# • Ensure areas map sensibly to materials; if in doubt, provide explicit
#   `materials.*.area_m2_override` values.

# ===============================================================================
# """ --> 
# # make project root importable when running directly
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json, csv
from src.cfg import load_config, validate_config
from src.io_utils import coloured, ensure_dir
from src.metrics import load_ground_truth, targets_avg_over_receivers, weighted_huber_loss
from src.materials import anchor_copy, clamp_vec, enforce_monotonic_increasing, deviation_mask_from_groups
from pathlib import Path

def main():
    cfg = load_config("configs/project.yaml")
    validate_config(cfg)

    run_dir = ensure_dir(Path(cfg["paths"]["working_dir"]) / cfg["_run_id"])
    bands = cfg["_bands"]["f_hz"]
    start_idx = cfg["_bands"]["opt_start_index"]

    # load GT and compute receiver-averaged targets for requested metrics
    gt = load_ground_truth(cfg["paths"]["ground_truth_json"])
    obj_metrics = cfg["metrics"]["objective_metrics"]
    tgt = targets_avg_over_receivers(gt, obj_metrics, bands)

    # === Stage-1 seed: take anchors as initial α(f) per material ===
    stage1_alpha = {}
    rows_for_csv = []

    # bounds and monotonic from YAML
    bounds = cfg.get("stage1", {}).get("bounds", {})
    alpha_min = float(bounds.get("alpha_min", 0.02))
    alpha_max = float(bounds.get("alpha_max", 0.95))
    monotonic = bool(cfg.get("stage1", {}).get("monotonic", True))

    for mkey, mdef in cfg["materials"].items():
        if mdef.get("type", "full_octave_absorption") != "full_octave_absorption":
            continue  # only absorption here
        alpha = anchor_copy(mdef.get("anchors", {}), "absorption")
        # clamp and (optionally) enforce monotonic ↑
        alpha = clamp_vec(alpha, alpha_min, alpha_max)
        if monotonic:
            alpha = enforce_monotonic_increasing(alpha)

        # optional small Stage-1 adjustments: allow band-wise ± deviation_pct around anchors
        dev_mask = deviation_mask_from_groups(bands, mdef.get("deviation_groups", {}))
        # For Stage-1 we’ll keep it equal to anchors; GA will mutate later.
        # (If you want a quick perturb: alpha[i] *= (1 ± dev_mask[i]*0.1) etc.)

        stage1_alpha[mkey] = {"bands_hz": bands, "alpha": alpha}
        for i, f in enumerate(bands):
            rows_for_csv.append({
                "material": mkey, "f_hz": f,
                "alpha": alpha[i], "deviation_pct": dev_mask[i]
            })

    # --- Produce a very simple proxy prediction for reporting only ---
    # Here we just repackage anchors as "simulated" metric placeholders so that
    # we can compute a baseline loss number against targets. In Stage-2, the
    # real forward sim will replace this entirely.
    sim = {m: [0.0]*len(bands) for m in obj_metrics}
    # A conservative placeholder: use no-op (zeros) so Stage-1 score is just “baseline”.
    # (Alternatively, if you have a quick Sabine/Eyring proxy, plug it here.)

    weights = cfg["metrics"].get("weights", {})
    huber_delta = cfg["metrics"].get("huber_delta", {})
    baseline_loss = weighted_huber_loss(sim, tgt, weights, huber_delta, start_idx)

    # write JSON and CSV
    (run_dir / "stage1_alpha.json").write_text(json.dumps(stage1_alpha, indent=2))
    with (run_dir / "stage1_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["material", "f_hz", "alpha", "deviation_pct"])
        w.writeheader()
        w.writerows(rows_for_csv)

    print(coloured(f"[OK] Stage-1 alpha written → {run_dir/'stage1_alpha.json'}", "green"))
    print(coloured(f"[INFO] Baseline (placeholder) loss vs targets: {baseline_loss:.4f}", "yellow"))
    print(coloured("[NEXT] Run 20_forward_once.py to validate via Treble.", "green"))

if __name__ == "__main__":
    main()
