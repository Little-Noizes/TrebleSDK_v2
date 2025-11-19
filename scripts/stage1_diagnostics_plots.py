# #!/usr/bin/env python
# """
# scripts/stage1_diagnostics_plots.py

# Post-process a Stage 1 + Hybrid run directory and produce:
# 1) A plot of measured vs predicted metric at a single band (e.g. 125 Hz)
# 2) A plot of the Stage 1 alpha(f) curves for all materials

# Usage (from repo root):
#     (venv_treble) PS> python -m scripts.stage1_diagnostics_plots ^
#         --run_dir C:\Users\usuario\Documents\TrebleSDK\v2\results\seed_run_001_20251105_185545 ^
#         --metric t20 --band 125
# """

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_joined(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "joined.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"joined.csv not found in {run_dir}")
    df = pd.read_csv(csv_path)
    
    # Check if dataframe is empty
    if df.empty:
        raise ValueError(f"joined.csv in {run_dir} is empty")
    
    # Normalize metric names to lowercase just in case
    if "metric" in df.columns:
        df["metric"] = df["metric"].str.lower()
    return df


def plot_metric_band(df: pd.DataFrame, metric: str, band_hz: int):
    """Plot target vs pred for all receivers at a single band."""
    metric = metric.lower()
    mask = (df["metric"] == metric) & (df["band_hz"] == band_hz)
    sub = df.loc[mask].copy()

    if sub.empty:
        print(f"[WARN] No rows found for metric='{metric}' at {band_hz} Hz")
        print("       Available metrics:", sorted(df["metric"].unique()))
        print("       Available bands:  ", sorted(df["band_hz"].unique()))
        return

    sub = sub.sort_values("receiver")

    # Basic summary
    err = sub["prediction"] - sub["target"]
    rmse = (err**2).mean() ** 0.5
    print(f"\n=== {metric.upper()} at {band_hz} Hz ===")
    print(f"Receivers: {len(sub)}")
    print(f"Mean target:     {sub['target'].mean():.3f}")
    print(f"Mean prediction: {sub['prediction'].mean():.3f}")
    print(f"RMSE:            {rmse:.3f}\n")

    # Plot per receiver
    plt.figure()
    x = range(len(sub))
    plt.plot(x, sub["target"], marker="o", linestyle="-", label="target")
    plt.plot(x, sub["prediction"], marker="x", linestyle="--", label="prediction")

    plt.xticks(x, sub["receiver"], rotation=90)
    plt.xlabel("Receiver")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} at {band_hz} Hz - Target vs Predicted")
    plt.legend()
    plt.tight_layout()


def load_stage1_alpha(run_dir: Path) -> dict:
    """Load Stage 1 alpha dict from stage1_alpha.json if present."""
    path = run_dir / "stage1_alpha.json"
    if not path.exists():
        raise FileNotFoundError(f"stage1_alpha.json not found in {run_dir}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_alpha_curves(stage1_alpha: dict, band_highlight: int | None = None):
    """
    Plot alpha(f) for each material.
    Tries a few common key patterns: 'bands_hz' + 'alpha', etc.
    """
    plt.figure()

    any_plotted = False

    for mat_name, spec in stage1_alpha.items():
        if not isinstance(spec, dict):
            continue

        # Try to find frequency and alpha data with common key names
        freqs = None
        alpha = None
        
        # Look for frequency data
        for freq_key in ["bands_hz", "freqs", "f_hz", "frequencies"]:
            if freq_key in spec and spec[freq_key] is not None:
                freqs = spec[freq_key]
                break
        
        # Look for alpha data
        for alpha_key in ["alpha", "values", "base", "absorption"]:
            if alpha_key in spec and spec[alpha_key] is not None:
                alpha = spec[alpha_key]
                break

        if freqs is None or alpha is None:
            print(f"[WARN] Could not find bands/alpha keys for material '{mat_name}'")
            continue

        if len(freqs) != len(alpha):
            print(f"[WARN] Length mismatch for '{mat_name}' (bands={len(freqs)}, alpha={len(alpha)})")
            continue

        plt.semilogx(freqs, alpha, marker="o", linestyle="-", label=mat_name)
        any_plotted = True

    if not any_plotted:
        print("[WARN] No alpha curves plotted (check stage1_alpha.json structure).")
        return

    if band_highlight is not None:
        plt.axvline(band_highlight, linestyle="--", linewidth=1, color="gray", alpha=0.7)
        plt.text(
            band_highlight, 0.95,
            f"{band_highlight} Hz",
            rotation=90,
            va="top", ha="right"
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Absorption coefficient α")
    plt.title("Stage 1 α(f) per material")
    plt.grid(True, which="both", linestyle=":")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()


def _parse_args():
    ap = argparse.ArgumentParser(description="Stage 1 diagnostics plots")
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Path to the run directory (where joined.csv and stage1_alpha.json live)",
    )
    ap.add_argument(
        "--metric",
        default="t20",
        help="Metric name to plot (e.g. t20, edt). Case-insensitive.",
    )
    ap.add_argument(
        "--band",
        type=int,
        default=125,
        help="Octave/third-octave band centre frequency in Hz (default: 125)",
    )
    return ap.parse_args()


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)

    # 1) Metric vs band plot
    try:
        df = load_joined(run_dir)
        plot_metric_band(df, args.metric, args.band)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    # 2) Alpha curves
    try:
        stage1_alpha = load_stage1_alpha(run_dir)
        plot_alpha_curves(stage1_alpha, band_highlight=args.band)
    except FileNotFoundError as e:
        print(f"[WARN] {e}")

    # Show all figures at the end
    plt.show()


if __name__ == "__main__":
    main()