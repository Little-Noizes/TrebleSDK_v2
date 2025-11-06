import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG (edit as needed) --------------------------------------------------
TARGETS_FILE =    "C:/Users/usuario/Documents/TrebleSDK/v2/data/omni/targets_measured_omni.json"          # (optional, not required to plot)
ALPHA_SEED_FILE = "C:/Users/usuario/Documents/TrebleSDK/v2/configs/results/seed_run_001_20251106_145658/stage1_alpha.json"                # required for 1_0 plot
RESULTS_LOG_FILE ="C:/Users/usuario/Documents/TrebleSDK/v2/configs/results/seed_run_001_20251106_145658/detailed_results_log.json"       # required for comparisons

FINAL_LOSS = 2.863608
RCV_CODES = ["R_30x00y", "R_60x15y", "R_00x20y", "R_40x30y", "R_15x40y"]
BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]

# --- Helpers ------------------------------------------------------------------
def norm_metric(s: str) -> str:
    return "".join(ch for ch in str(s).strip().upper() if ch.isalnum())

def load_json(path: str | Path):
    p = Path(path)
    if not p.exists():
        print(f"[warn] file not found: {p}")
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def plot_alpha_coeffs(alpha_json: dict, title: str, out_png: str):
    if not isinstance(alpha_json, dict):
        print("[warn] alpha json invalid; skipping alpha plot")
        return
    rows = []
    for name, spec in alpha_json.items():
        if "air" in str(name).lower():
            continue
        bands = spec.get("bands_hz", [])
        alpha = spec.get("alpha", [])
        if not bands or not alpha:
            continue
        if len(bands) != len(alpha):
            continue
        for b, a in zip(bands, alpha):
            try:
                rows.append({
                    "Material": str(name).replace("My_", "").replace("_", " "),
                    "Frequency_Hz": int(b),
                    "Alpha": float(a),
                })
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if df.empty:
        print("[warn] no material rows to plot in alpha; skipping")
        return

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Absorption Coefficient (α)")

    for mat in sorted(df["Material"].unique()):
        subset = df[df["Material"] == mat].sort_values("Frequency_Hz")
        plt.plot(subset["Frequency_Hz"], subset["Alpha"], marker="o", linestyle="-", label=mat)

    plt.xscale("log")
    plt.xticks(sorted(df["Frequency_Hz"].unique()))
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
    plt.ylim(0, 1.0)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Material", loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(out_png)

def results_df_from_log(log_json: list, want_metric="EDT_s"):
    if not isinstance(log_json, list) or not log_json:
        return pd.DataFrame()
    m_norm = norm_metric(want_metric)
    rows = []
    for r in log_json:
        try:
            if norm_metric(r.get("metric", "")) != m_norm:
                continue
            rows.append({
                "receiver": r.get("rcv_code", ""),
                "frequency_hz": int(r.get("f_hz")),
                "target": float(r.get("target_val")),
                "prediction": float(r.get("predicted_val")),
                "error": float(r.get("error")),
            })
        except Exception:
            # skip malformed row
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep only requested bands & receivers (if present)
    df = df[df["frequency_hz"].isin(BANDS)]
    if RCV_CODES:
        df = df[df["receiver"].isin(RCV_CODES)]
    # drop non-finite values (log a quick note)
    init_len = len(df)
    df = df[np.isfinite(df["target"]) & np.isfinite(df["prediction"])]
    dropped = init_len - len(df)
    if dropped:
        print(f"[warn] dropped {dropped} non-finite row(s) before plotting")
    return df.sort_values(["receiver", "frequency_hz"])

def plot_avg_comparison(df: pd.DataFrame, loss: float, out_png: str):
    if df.empty:
        print("[warn] no rows for average comparison; skipping")
        return
    avg = df.groupby("frequency_hz", as_index=False).agg(
        target_avg=("target", "mean"),
        pred_avg=("prediction", "mean")
    )
    plt.figure(figsize=(8, 6))
    plt.title(f"Statistical Average EDT\u209B Comparison (Loss={loss:.3f})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Average EDT\u209B (s)")

    plt.plot(avg["frequency_hz"], avg["target_avg"], marker="o", linestyle="-", label="Target (Measured)")
    plt.plot(avg["frequency_hz"], avg["pred_avg"], marker="s", linestyle="--", label="Simulation (Stage 1 Seed)")

    plt.xscale("log")
    plt.xticks(BANDS)
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(out_png)

def plot_receiver_series(df: pd.DataFrame, loss: float, out_dir: Path):
    if df.empty:
        print("[warn] no rows for per-receiver plots; skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for rc in RCV_CODES:
        sub = df[df["receiver"] == rc].sort_values("frequency_hz")
        if sub.empty:
            print(f"[info] no rows for {rc}; skipping")
            continue
        # simple MAE for display
        mae = float((sub["target"] - sub["prediction"]).abs().mean())

        plt.figure(figsize=(8, 6))
        plt.title(f"{rc} — EDT\u209B vs Measured (MAE={mae:.3f} s; Total Loss={loss:.3f})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("EDT\u209B (s)")

        plt.plot(sub["frequency_hz"], sub["target"], marker="o", linestyle="-", label="Target (Measured)")
        plt.plot(sub["frequency_hz"], sub["prediction"], marker="s", linestyle="--", label="Simulation (Stage 1 Seed)")

        plt.xscale("log")
        plt.xticks(BANDS)
        plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(loc="best")
        plt.tight_layout()
        out_png = out_dir / f"2_0_{rc}_edt.png"
        plt.savefig(out_png, dpi=150)
        print(out_png)

# --- RUN ----------------------------------------------------------------------
# --- RUN ----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from pathlib import Path

    # Automatically detect latest results folder
    base_results_dir = Path(r"C:/Users/usuario/Documents/TrebleSDK/v2/configs/results")
    run_dirs = sorted(
        [d for d in base_results_dir.iterdir() if d.is_dir() and d.name.startswith("seed_run_001_")],
        key=os.path.getmtime,
    )
    if not run_dirs:
        raise RuntimeError(f"No results folders found in {base_results_dir}")
    run_dir = run_dirs[-1]  # latest
    print(f"[info] Using latest run folder: {run_dir}")

    # Define input/output paths inside the run folder
    RESULTS_LOG_FILE = run_dir / "detailed_results_log.json"
    ALPHA_SEED_FILE = run_dir / "stage1_alpha.json"
    TARGETS_FILE = run_dir / "targets_measured_omni.json"  # optional

    # Output PNGs will go here
    out_dir = run_dir
    per_rcv_dir = out_dir / "per_receiver_plots"
    per_rcv_dir.mkdir(exist_ok=True)

    # Load JSONs
    alpha_json = load_json(ALPHA_SEED_FILE)
    log_json = load_json(RESULTS_LOG_FILE)

    # --- 1) Alpha Coefficients (Stage 1 Seed) ---
    if alpha_json:
        plot_alpha_coeffs(alpha_json,
                          "1.0) Initial (Stage 1 Seed) Absorption Coefficients (α)",
                          out_dir / "1_0_initial_alpha.png")

    # --- 2) Results Comparisons ---
    df = results_df_from_log(log_json, want_metric="EDT_s")
    if not df.empty:
        plot_avg_comparison(df, FINAL_LOSS, out_dir / "1_1_avg_edt_comparison.png")
        plot_receiver_series(df, FINAL_LOSS, per_rcv_dir)
    else:
        print(f"[info] No valid rows found in {RESULTS_LOG_FILE}; skipping comparison plots.")

    print(f"\nAll plots saved under: {run_dir}")

