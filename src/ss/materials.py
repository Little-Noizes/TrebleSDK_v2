# <!-- """
# ===============================================================================
# src/materials.py
# ===============================================================================
# Purpose
# -------
# Utility module for managing material acoustic parameters, primarily absorption
# and scattering coefficients, during optimisation or statistical fitting stages.

# Responsibilities:
# 1. Provide helper functions for safe manipulation of per-band absorption vectors.
# 2. Enforce physical and numerical constraints (e.g., 0 ≤ α ≤ 1, monotonic ↑).
# 3. Generate deviation masks based on YAML "deviation_groups" definitions.
# 4. Support Stage-1 statistical fitting and Stage-2 GA mutation bounds.

# Used by:
# - scripts/10_stage1_stat_fit.py  → building initial (Stage-1) absorption dataset
# - src/eval_forward.py / scripts/30_ga_loop.py → mutation / GA candidate update

# -------------------------------------------------------------------------------
# Dependencies
# ------------
# • Python standard library only:
#   - typing (for List, Dict)

# No external dependencies, no file I/O.

# -------------------------------------------------------------------------------
# Expected Inputs
# ---------------
# Typical caller data originates from `project.yaml`:

# materials:
#   My_Plasterboard_Ceiling:
#     type: "full_octave_absorption"
#     anchors:
#       absorption: [0.15, 0.12, 0.10, 0.09, 0.08, 0.08, 0.08, 0.08]
#       scattering: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
#     deviation_groups:
#       low:  { bands: [63, 125], deviation_pct: 0.20 }
#       mid:  { bands: [250, 500, 1000], deviation_pct: 0.30 }
#       high: { bands: [2000, 4000, 8000], deviation_pct: 0.25 }

# Functions in this module accept these subsections (e.g., `anchors` dict,
# `deviation_groups` dict) and return simple Python lists ready for computation.

# -------------------------------------------------------------------------------
# Functions
# ----------
# clamp_vec(v, lo, hi)
#     → List[float]
#     Clamp each element of vector `v` between lower and upper bounds.
#     Used to maintain physical limits (0 ≤ α ≤ 1).

# enforce_monotonic_increasing(v)
#     → List[float]
#     Enforces monotonic non-decreasing behaviour in absorption spectra:
#     each band cannot have a smaller α than the previous one.

# anchor_copy(anchors, which="absorption")
#     → List[float]
#     Safely returns a copy of the "absorption" (or "scattering") anchor list
#     from the material's `anchors` dictionary.

# deviation_mask_from_groups(f_bands, groups_cfg)
#     → List[float]
#     Generates a per-band vector of allowed deviation percentages based on
#     `deviation_groups` definitions.
#     Each band’s value corresponds to the maximum deviation % permitted for GA.

# -------------------------------------------------------------------------------
# Inputs
# ------
# - v : List[float]                  → numeric absorption/scattering data
# - lo, hi : float                   → lower/upper bounds for clamping
# - anchors : Dict[str, List[float]] → typically from `material.anchors` in YAML
# - which : str                      → "absorption" or "scattering"
# - f_bands : List[int]              → band centres, e.g., [63,125,250,...]
# - groups_cfg : Dict                → YAML-style structure from deviation_groups

# -- -->

from __future__ import annotations
from typing import List, Dict

def clamp_vec(v: List[float], lo: float, hi: float) -> List[float]:
    return [min(max(x, lo), hi) for x in v]

def enforce_monotonic_increasing(v: List[float]) -> List[float]:
    out = []
    cur = v[0]
    out.append(cur)
    for x in v[1:]:
        cur = max(cur, x)
        out.append(cur)
    return out

def anchor_copy(anchors: Dict[str, List[float]], which="absorption") -> List[float]:
    arr = anchors.get(which)
    if not arr:
        raise ValueError(f"anchors.{which} missing or empty")
    return list(arr)

def deviation_mask_from_groups(f_bands: List[int], groups_cfg: Dict) -> List[float]:
    # default 0.0 (no deviation) unless specified
    mask = [0.0] * len(f_bands)
    if not groups_cfg:
        return mask
    # groups like: {low: {bands:[63,125], deviation_pct:0.2}, ...}
    for _gname, g in groups_cfg.items():
        bset = set(int(b) for b in g.get("bands", []))
        dev = float(g.get("deviation_pct", 0.2))
        for i, f in enumerate(f_bands):
            if int(f) in bset:
                mask[i] = max(mask[i], dev)
    return mask
