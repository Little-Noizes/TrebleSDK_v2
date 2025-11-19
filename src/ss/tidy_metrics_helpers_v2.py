import pandas as pd
import json
import os
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

# --- Static Configuration ---
# Full list of bands present in the JSON files (used for indexing)
FULL_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

# --- Helper function copied from run_stage1_and_forward.py ---
def _normalise_metric_key(name: str) -> str:
    """
    Normalise metric names and map common aliases so YAML does not need changes.
    Examples: EDT_s -> edt, C50_dB -> c50.
    The result is always lowercase.
    """
    raw = "".join(ch for ch in str(name).strip().upper() if ch.isalnum())
    aliases = {
        # Time constants
        "EDTS": "edt", "T20S": "t20", "T30S": "t30", "T60S": "t60", "RT60S": "t60",
        # Clarity / definition / centre time
        "C50DB": "c50", "C80DB": "c80", "D50": "d50", "TS": "ts",
    }
    # The default return (if not in aliases) is raw.lower()
    return aliases.get(raw, raw.lower())

def load_json(path: Path) -> Optional[Dict]:
    """Load JSON data from a file path."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

# --- Core Helper Functions required by run_stage1_and_forward.py ---

def load_targets(gt_json_path: str, default_bands: List[int]) -> pd.DataFrame:
    """Load ground-truth metrics from a JSON file into a tidy DataFrame."""
    gt_path = Path(gt_json_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_json_path}")

    data = load_json(gt_path)
    if not data:
        return pd.DataFrame()

    freq_vec = [int(round(float(f))) for f in data.get("frequency_bands") or data.get("bands") or default_bands]
    
    records = []
    points = data.get("points") if isinstance(data.get("points"), list) else (data if isinstance(data, dict) and not data.get("points") else [])

    for entry in points:
        if not isinstance(entry, dict):
            continue
        label = entry.get("rcv_code") or entry.get("label") or entry.get("name")
        if not label:
            continue
        metrics_map = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        
        for metric_name, values in metrics_map.items():
            metric = _normalise_metric_key(str(metric_name))
            
            # Case 1: Values is a list (assumed to be in frequency_bands order)
            if isinstance(values, list):
                for idx, freq_hz in enumerate(freq_vec):
                    if idx < len(values):
                        val = values[idx]
                        if isinstance(val, (int, float)) and math.isfinite(val):
                            records.append({
                                "receiver": str(label), 
                                "metric": metric, 
                                "band_hz": int(freq_hz), 
                                "target": val
                            })
            
            # Case 2: Values is a dict (frequency as string key)
            elif isinstance(values, dict):
                for key2, val2 in values.items():
                    try:
                        freq_hz = int(round(float(key2)))
                        val = float(val2)
                        if math.isfinite(val):
                            records.append({
                                "receiver": str(label), 
                                "metric": metric, 
                                "band_hz": int(freq_hz), 
                                "target": val
                            })
                    except Exception:
                        continue

    return pd.DataFrame(records)

def load_hybrid_pred(run_dir: Path) -> pd.DataFrame:
    """
    Scan run_dir for latest *_Hybrid.json and return results in a tidy DataFrame.
    This is robust to band-data being a list (ordered) or a dict (f_hz mapped).
    """
    candidates = sorted(run_dir.glob("*_Hybrid.json"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"No *_Hybrid.json found in {run_dir}")
    hybrid_path = candidates[-1]
    
    data = load_json(hybrid_path)
    if not data:
        return pd.DataFrame()

    recvs = data.get("receivers") or {}
    records = []

    for r_id, payload in recvs.items():
        label = payload.get("label") or r_id
        params = payload.get("parameters") or {}
        
        for m_key, band_map in params.items():
            metric = _normalise_metric_key(m_key)
            
            # Case 1 (The Fix): band_map is a list of values (assumed ordered by FULL_BANDS)
            if isinstance(band_map, list):
                if len(band_map) != len(FULL_BANDS):
                    continue # Skip if length doesn't match standard bands
                
                for i, freq_hz in enumerate(FULL_BANDS):
                    val = band_map[i]
                    if isinstance(val, (int, float)) and math.isfinite(val):
                        records.append({
                            "receiver": str(label), 
                            "metric": metric, 
                            "band_hz": int(freq_hz), 
                            "prediction": val
                        })

            # Case 2: band_map is a dict (frequency as string key)
            elif isinstance(band_map, dict):
                for fk, vv in band_map.items():
                    try:
                        freq_hz = int(round(float(fk)))
                        val = float(vv)
                        if math.isfinite(val):
                            records.append({
                                "receiver": str(label), 
                                "metric": metric, 
                                "band_hz": int(freq_hz), 
                                "prediction": val
                            })
                    except Exception:
                        continue
                        
    return pd.DataFrame(records)


def join_metrics(
    df_tgt: pd.DataFrame, 
    df_pred: pd.DataFrame, 
    keep_receivers: List[str], 
    keep_metrics: List[str], 
    keep_bands: List[int]
) -> pd.DataFrame:
    """Inner-join target and prediction DataFrames on (receiver, metric, band)."""
    
    # Filter by requested attributes
    df_tgt = df_tgt[df_tgt["receiver"].isin(keep_receivers)]
    df_pred = df_pred[df_pred["receiver"].isin(keep_receivers)]
    
    df_tgt = df_tgt[df_tgt["metric"].isin(keep_metrics)]
    df_pred = df_pred[df_pred["metric"].isin(keep_metrics)]
    
    df_tgt = df_tgt[df_tgt["band_hz"].isin(keep_bands)]
    df_pred = df_pred[df_pred["band_hz"].isin(keep_bands)]

    # Inner join
    df_join = pd.merge(
        df_tgt, df_pred, 
        on=["receiver", "metric", "band_hz"], 
        how="inner"
    )

    # Convert columns to float and drop NaNs (should be minimal after load)
    df_join["target"] = pd.to_numeric(df_join["target"], errors='coerce')
    df_join["prediction"] = pd.to_numeric(df_join["prediction"], errors='coerce')
    df_join.dropna(subset=["target", "prediction"], inplace=True)

    return df_join


def compute_weighted_huber_loss(
    df_join: pd.DataFrame,
    weights: Dict[str, Any],
    huber_delta: Dict[str, Any],
    reduction: str = "mean",
) -> float:
    """Computes the weighted Huber loss for the joined DataFrame."""

    if df_join.empty:
        return float('nan')

    # Get raw errors
    errors = df_join["prediction"] - df_join["target"]

    # --- Weights and Deltas ---
    # Normalise keys (using the same logic as run_stage1_and_forward.py)
    norm_weights = {
        _normalise_metric_key(k): v for k, v in (weights or {}).items()
    }
    norm_deltas = {
        _normalise_metric_key(k): v for k, v in (huber_delta or {}).items()
    }
    
    # Add 'all' for metrics/receivers not explicitly weighted/deltas
    norm_weights['all'] = norm_weights.get('all', 1.0)
    norm_deltas['all'] = norm_deltas.get('all', 1.0)
    
    # Apply weights and deltas per row
    final_weights = pd.Series(1.0, index=df_join.index)
    final_deltas = pd.Series(float(norm_deltas['all']), index=df_join.index)

    # Weight/Delta by Metric
    for metric, w in norm_weights.items():
        if metric in df_join["metric"].unique():
            final_weights.loc[df_join["metric"] == metric] = float(w)
            
    for metric, d in norm_deltas.items():
        if metric in df_join["metric"].unique():
            final_deltas.loc[df_join["metric"] == metric] = float(d)

    # Weight/Delta by Receiver
    # NOTE: The weights/deltas in YAML can also be by rcv_code, but the common
    # practice in the script seems to be metric-specific weighting. 
    # For robustness, we'll only apply per-metric weights/deltas.
    
    # --- Huber Loss Calculation ---
    abs_errors = errors.abs()
    
    # Quadratic region (abs_error <= delta)
    quad_mask = abs_errors <= final_deltas
    loss_quad = 0.5 * (errors[quad_mask] ** 2)

    # Linear region (abs_error > delta)
    linear_mask = abs_errors > final_deltas
    loss_linear = final_deltas[linear_mask] * (abs_errors[linear_mask] - 0.5 * final_deltas[linear_mask])

    # Total weighted loss (sum of both regions)
    total_loss_sum = (final_weights[quad_mask] * loss_quad).sum() + \
                     (final_weights[linear_mask] * loss_linear).sum()
                     
    if reduction == "mean":
        sum_weights = final_weights.sum()
        return total_loss_sum / sum_weights if sum_weights != 0 else float('nan')
    elif reduction == "sum":
        return total_loss_sum
    else:
        # Default to mean reduction if not specified or invalid
        sum_weights = final_weights.sum()
        return total_loss_sum / sum_weights if sum_weights != 0 else float('nan')