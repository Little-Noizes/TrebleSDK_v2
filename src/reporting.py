"""
===============================================================================
src/reporting.py
===============================================================================
Purpose
-------
Small, dependency-free CLI printing utilities for forward runs:
- echo configuration (obj, materials, source, receivers)
- tabular per-band comparisons (target vs predicted)
- summary norms per receiver and overall

No external dependencies; pure stdlib.

===============================================================================
"""

from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
from pathlib import Path
import math

def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        # compact but precise enough
        return f"{x:.3f}"
    return str(x)

def echo_header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def echo_kv(label: str, value):
    print(f"{label:<24} {value}")

def echo_list(label: str, items: Iterable[str]):
    items = list(items)
    print(f"{label:<24} {len(items)}")
    for s in items:
        print(f"  - {s}")

def echo_source(kind: str, pos: List[float], orient: List[float] | None, cf2_path: str | None):
    echo_header("SOURCE")
    echo_kv("Type", kind)
    echo_kv("Position [m]", pos)
    if orient is not None:
        echo_kv("Orientation [deg xyz]", orient)
    if kind.lower() == "directive":
        echo_kv("Directivity (CF2)", cf2_path or "(none)")

def echo_receivers(rcvs: List[Dict]):
    echo_header("RECEIVERS")
    for r in rcvs:
        echo_kv(f"{r['rcv_code']}", r["xyz_m"])

def table_compare_per_band(
    rcv_code: str,
    metric_name: str,
    bands_hz: List[int],
    target_vec: List[float],
    predicted_vec: List[float],
):
    echo_header(f"COMPARE   rcv={rcv_code}   metric={metric_name}")
    print(f"{'Band[Hz]':>9}  {'Target':>10}  {'Pred':>10}  {'|Diff|':>10}")
    print("-" * 48)
    diffs = []
    for f, t, p in zip(bands_hz, target_vec, predicted_vec):
        if t is None or p is None or (isinstance(t, float) and not math.isfinite(t)) or (isinstance(p, float) and not math.isfinite(p)):
            print(f"{f:>9}  {(_fmt(t)):>10}  {(_fmt(p)):>10}  {(_fmt(None)):>10}")
            continue
        d = abs(p - t)
        diffs.append(d)
        print(f"{f:>9}  {t:>10.3f}  {p:>10.3f}  {d:>10.3f}")
    # Norms
    if diffs:
        mae = sum(diffs) / len(diffs)
        rmse = (sum(d*d for d in diffs) / len(diffs)) ** 0.5
        print("-" * 48)
        print(f"{'MAE':>9}  {'':>10}  {'':>10}  {mae:>10.3f}")
        print(f"{'RMSE':>9}  {'':>10}  {'':>10}  {rmse:>10.3f}")

def summarise_receiver_errors(
    per_metric_diffs: Dict[str, List[float]]
) -> Tuple[float, float]:
    """
    Return (MAE, RMSE) aggregating all metrics/bands for a receiver.
    """
    flat = [abs(x) for arr in per_metric_diffs.values() for x in arr if isinstance(x, (int, float))]
    if not flat:
        return float("nan"), float("nan")
    mae = sum(flat) / len(flat)
    rmse = (sum(x*x for x in flat) / len(flat)) ** 0.5
    return mae, rmse
