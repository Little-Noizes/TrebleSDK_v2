import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Show measured/simulated metrics as ASCII table.")
    parser.add_argument("--metric", required=True, help="Metric to use (e.g. t20, edt, c50, etc.)")
    parser.add_argument("--bands", required=True, help="Comma-separated frequency bands (e.g. 125,250,500,1000,2000,4000,8000)")
    parser.add_argument("--receivers", required=True, help="Comma-separated receiver codes (e.g. R_30x00y,R_60x15y,...)")
    parser.add_argument("--targets", required=True, help="Path to measured targets JSON")
    parser.add_argument("--run-dir", required=True, help="Path to run directory containing Hybrid JSON")
    return parser.parse_args()

def load_targets(path, metric, bands, receivers):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Try both 'points' and flat mapping
    if "points" in data:
        rows = []
        for p in data["points"]:
            rcv = p.get("rcv_code")
            if rcv not in receivers:
                continue
            metrics = p.get("metrics", {})
            vals = metrics.get(metric)
            if isinstance(vals, list):
                for b, v in zip(bands, vals):
                    rows.append((rcv, b, v))
            elif isinstance(vals, dict):
                for b, v in vals.items():
                    if int(b) in bands:
                        rows.append((rcv, int(b), v))
        df = pd.DataFrame(rows, columns=["receiver", "band", "measured"])
    else:
        # Flat mapping
        rows = []
        for rcv, block in data.items():
            if rcv not in receivers:
                continue
            metrics = block.get("metrics", {})
            vals = metrics.get(metric)
            if isinstance(vals, list):
                for b, v in zip(bands, vals):
                    rows.append((rcv, b, v))
            elif isinstance(vals, dict):
                for b, v in vals.items():
                    if int(b) in bands:
                        rows.append((rcv, int(b), v))
        df = pd.DataFrame(rows, columns=["receiver", "band", "measured"])
    return df

def load_hybrid(run_dir, metric, bands, receivers):
    run_dir = Path(run_dir)
    hyb_files = sorted(run_dir.glob("*_Hybrid.json"), key=lambda p: p.stat().st_mtime)
    if not hyb_files:
        raise FileNotFoundError("No Hybrid JSON found in run-dir")
    with open(hyb_files[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for rid, payload in (data.get("receivers") or {}).items():
        rcv = payload.get("label", rid)
        if rcv not in receivers:
            continue
        params = payload.get("parameters", {})
        vals = params.get(metric)
        if isinstance(vals, dict):
            for b, v in vals.items():
                if int(b) in bands:
                    rows.append((rcv, int(b), v))
        elif isinstance(vals, list):
            freq_vec = [int(f) for f in data.get("frequencies", [])]
            for b, v in zip(freq_vec, vals):
                if b in bands:
                    rows.append((rcv, b, v))
    df = pd.DataFrame(rows, columns=["receiver", "band", "simulated"])
    return df

def main():
    args = parse_args()
    metric = args.metric.strip()
    bands = [int(b) for b in args.bands.split(",") if b.strip()]
    receivers = [r.strip() for r in args.receivers.split(",") if r.strip()]
    df_meas = load_targets(args.targets, metric, bands, receivers)
    df_sim = load_hybrid(args.run_dir, metric, bands, receivers)
    # Merge
    df = pd.merge(df_meas, df_sim, on=["receiver", "band"], how="outer")
    df = df.set_index(["receiver", "band"]).sort_index()
    # Build table
    table = {}
    # Measured average per band
    avg_measured = df.groupby("band")["measured"].mean()
    table["measured_average"] = avg_measured
    for rcv in receivers:
        row_meas = df.loc[rcv]["measured"] if rcv in df.index.get_level_values(0) else pd.Series([np.nan]*len(bands), index=bands)
        row_sim = df.loc[rcv]["simulated"] if rcv in df.index.get_level_values(0) else pd.Series([np.nan]*len(bands), index=bands)
        # If only one band, convert to Series
        if not isinstance(row_meas, pd.Series):
            row_meas = pd.Series([row_meas], index=bands)
        if not isinstance(row_sim, pd.Series):
            row_sim = pd.Series([row_sim], index=bands)
        abs_diff = row_sim - row_meas
        norm_err = (row_sim - row_meas) / row_meas
        table[f"meas_{rcv}"] = row_meas
        table[f"sim_{rcv}"] = row_sim
        table[f"abs_diff_{rcv}"] = abs_diff
        table[f"norm_error_{rcv}"] = norm_err
    # Print table
    print(f"\n{'':<18}", end="")
    for b in bands:
        print(f"{b:>10}", end="")
    print("\n" + "-" * (18 + 10*len(bands)))
    for rowname, series in table.items():
        print(f"{rowname:<18}", end="")
        for b in bands:
            val = series.get(b, np.nan)
            if pd.isna(val):
                print(f"{'nan':>10}", end="")
            else:
                print(f"{val:10.3f}", end="")
        print()
    print("-" * (18 + 10*len(bands)))

if __name__ == "__main__":
    main()