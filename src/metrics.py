from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np # Standard dependency for array/NaN handling
from typing import Dict, List, Optional, Any, Tuple

# ---------- helpers for NaN-safe averaging ----------
def _isfinite(x: Any) -> bool:
    """Check if x is a finite number, handling non-numeric types safely."""
    try:
        # Use np.isfinite for robust checking, especially with arrays
        if isinstance(x, (int, float)):
            return math.isfinite(x)
        # Use numpy for array-like safety if needed elsewhere
        return bool(np.isfinite(x))
    except (TypeError, ValueError):
        return False

def _normalise_metric_key(key: str) -> str:
    """Normalises metric names for consistent dictionary lookups (e.g., T20 -> T20)."""
    # NOTE: The patch uses this helper internally, so it must be defined.
    # We will normalize to standard ALL CAPS form as used in the patch:
    return key.strip().upper().replace('.', '').replace('_', '')

# --- Loss Helpers ---
def _huber_scalar(diff: float, delta: float) -> float:
    """Computes the scalar Huber loss for a single difference."""
    a = abs(diff)
    return 0.5 * a * a / delta if a <= delta else a - 0.5 * delta

# --- Metric Helpers ---
def _normalise_metric_name_list(metrics_list: List[str]) -> List[str]:
    """Normalises a list of metric names."""
    return [_normalise_metric_key(m) for m in metrics_list]


# -----------------------------------------------------------------------------
# 1. Metric Computation from Impulse Response (IR)
# -----------------------------------------------------------------------------
def compute_metrics_from_ir(
    ir_object: Any,
    bands_hz: List[int],
    metrics_list: List[str]
) -> Dict[str, Dict[int, float]]:
    """
    Compute acoustic metrics from a Treble IR-like object, normalising to:
      { 'EDT': {125: v, 250: v, ...}, 'T20': {...}, 'C50': {...}, ... }

    Tries several Treble APIs depending on SDK version. If none match, returns a
    shaped zeros dict so the rest of the pipeline can run.
    """
    metrics_norm = _normalise_metric_name_list(metrics_list)
    bands_int = [int(round(f)) for f in bands_hz]

    def _shape_empty() -> Dict[str, Dict[int, float]]:
        return {m: {f: 0.0 for f in bands_int} for m in metrics_norm}

    def _shape_from_raw(raw: Any) -> Dict[str, Dict[int, float]]:
        """
        Accept a few possible raw shapes and coerce to our dict-of-dicts.
        """
        if raw is None:
            return _shape_empty()

        # Case 1: dict with 'metrics' sub-dict
        if isinstance(raw, dict) and "metrics" in raw:
            met_map = raw["metrics"]
            out: Dict[str, Dict[int, float]] = {}
            for m in metrics_norm:
                vec = met_map.get(m) or met_map.get(m.upper())
                if isinstance(vec, list) and len(vec) == len(bands_int):
                    out[m] = {bands_int[i]: float(vec[i]) for i in range(len(bands_int))}
                else:
                    out[m] = {f: 0.0 for f in bands_int}
            return out

        # Case 2: dict of metric -> vector
        if isinstance(raw, dict):
            out: Dict[str, Dict[int, float]] = {}
            for m in metrics_norm:
                vec = raw.get(m) or raw.get(m.upper())
                if isinstance(vec, list) and len(vec) == len(bands_int):
                    out[m] = {bands_int[i]: float(vec[i]) for i in range(len(bands_int))}
                else:
                    out[m] = {f: 0.0 for f in bands_int}
            return out

        # Case 3: list of metric items
        if isinstance(raw, list):
            mm: Dict[str, List[float]] = {}
            for item in raw:
                try:
                    k = _normalise_metric_key(str(item.get("metric")))
                    values = item.get("values", [])
                except Exception:
                    continue
                if isinstance(values, list) and len(values) == len(bands_int):
                    mm[k] = [float(x) for x in values]
            out: Dict[str, Dict[int, float]] = {}
            for m in metrics_norm:
                vec = mm.get(m, None)
                if vec is not None:
                    out[m] = {bands_int[i]: vec[i] for i in range(len(bands_int))}
                else:
                    out[m] = {f: 0.0 for f in bands_int}
            return out

        return _shape_empty()

    # Try direct call on IR
    if hasattr(ir_object, "get_acoustic_parameters"):
        try:
            raw = ir_object.get_acoustic_parameters(metrics=metrics_norm, bands=bands_int)
            return _shape_from_raw(raw)
        except Exception:
            pass

    # Try a parent/owner that can compute metrics
    for attr in ("parent_results", "results", "parent", "owner", "to_results"):
        if hasattr(ir_object, attr):
            try:
                holder = getattr(ir_object, attr)
                # if it's a callable like to_results()
                holder = holder() if callable(holder) else holder
                if hasattr(holder, "get_acoustic_parameters"):
                    raw = holder.get_acoustic_parameters(metrics=metrics_norm, bands=bands_int)
                    return _shape_from_raw(raw)
            except Exception:
                pass

    # If nothing matched, return a safe shaped zero map
    return _shape_empty()


# -----------------------------------------------------------------------------
# 2. Loss Functions
# -----------------------------------------------------------------------------

def weighted_huber_loss_aggregated(sim: Dict[str, List[float]],
                                   tgt: Dict[str, List[float]],
                                   weights: Dict[str, float],
                                   huber_delta: Dict[str, float],
                                   start_idx: int) -> float:
    """
    Original implementation (unchanged) kept for backwards compatibility.
    Computes loss from dicts, weighted by metric.
    """
    total, count = 0.0, 0
    for m, tvec in tgt.items():
        # Ensure 'sim' has the metric, if not, skip it
        if m not in sim:
            continue
            
        svec = sim[m]
        w = float(weights.get(m, 1.0))
        d = float(huber_delta.get(m, 0.5))
        
        # Ensure lists are the same length before iterating
        vec_len = min(len(tvec), len(svec))
        
        for i in range(start_idx, vec_len):
            s = svec[i]; t = tvec[i]
            if not (_isfinite(s) and _isfinite(t)):
                continue
            total += w * _huber_scalar(s - t, d)
            count += 1
    return total / max(count, 1)


def weighted_huber_loss(
    y_true: List[float],
    y_pred: List[float],
    y_metric: List[str],
    y_band: List[int],
    y_rcv: List[str],
    weights_cfg: Dict[str, Any]
) -> float:
    """
    Vectorised weighted Huber loss that matches 30_run_stage1_and_forward.py.
    This is the function required by the wrapper script.
    """
    n = len(y_true)
    if n == 0:
        return 1e4  # penalise empty comparisons

    delta_cfg = weights_cfg.get("huber_delta", 0.1)
    # The 'metrics' key in weights_cfg often contains the weights *and* delta config.
    # We prioritize specific weight keys if available.
    metric_w = dict(weights_cfg.get("metric_weights", {}))
    band_w   = dict(weights_cfg.get("band_weights", {}))
    rcv_w    = dict(weights_cfg.get("receiver_weights", {}))
    
    # Handle the structure where metric weights are nested under 'metrics: weights'
    if not metric_w:
        metric_w = dict(weights_cfg.get("weights", {}))
    
    total = 0.0
    wsum  = 0.0

    for i in range(n):
        t = float(y_true[i]); p = float(y_pred[i])
        m = _normalise_metric_key(str(y_metric[i]))
        f = int(y_band[i]);  r = str(y_rcv[i])

        diff = p - t

        # Metric-specific delta allowed, else use scalar
        if isinstance(delta_cfg, dict):
            delta = float(delta_cfg.get(m, 0.1))
        else:
            delta = float(delta_cfg)

        hub = _huber_scalar(diff, delta)

        # Look up weights: default to 1.0 if not found
        wm = float(metric_w.get(m, 1.0))
        wf = float(band_w.get(str(f), 1.0))
        wr = float(rcv_w.get(r, 1.0))
        w  = wm * wf * wr

        total += w * hub
        wsum  += w

    return float(total / wsum) if wsum > 0 else 0.0