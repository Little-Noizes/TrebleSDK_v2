# src/materials/materials.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Sequence, Optional

# -----------------------------------------------------------------------------
# LOW-LEVEL HELPERS
# -----------------------------------------------------------------------------

def clamp_vec(v: Sequence[float], lo: float, hi: float) -> List[float]:
    """Clamp each value of v to [lo, hi]."""
    return [hi if x > hi else lo if x < lo else float(x) for x in v]

def enforce_monotonic_increasing(v: Sequence[float]) -> List[float]:
    """
    Enforce non-decreasing (monotonic increasing) behaviour across bands.
    Useful for 'physically reasonable' trends when anchors are noisy.
    """
    if not v:
        return []
    out: List[float] = [float(v[0])]
    for x in v[1:]:
        xv = float(x)
        if xv < out[-1]:
            out.append(out[-1])
        else:
            out.append(xv)
    return out

def _linear_interp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x1 == x0:
        return float(y0)
    t = (x - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))

def _interp_to_target_bands(
    anchor_bands_hz: Sequence[float],
    anchor_vals: Sequence[float],
    target_bands_hz: Sequence[int | float],
) -> List[float]:
    """
    Simple, robust linear interpolation from anchor bands to the target band list.
    Extrapolates flat at the ends.
    """
    if len(anchor_bands_hz) != len(anchor_vals):
        raise ValueError("Anchor bands and values must have the same length.")
    if not anchor_bands_hz:
        return [0.0 for _ in target_bands_hz]

    # Ensure ascending by band
    pairs = sorted(
        zip([float(b) for b in anchor_bands_hz], [float(a) for a in anchor_vals]),
        key=lambda t: t[0]
    )
    ab = [p[0] for p in pairs]
    av = [p[1] for p in pairs]

    out: List[float] = []
    for f in target_bands_hz:
        x = float(f)
        if x <= ab[0]:
            out.append(av[0])
            continue
        if x >= ab[-1]:
            out.append(av[-1])
            continue
        # find segment
        for i in range(1, len(ab)):
            if ab[i] >= x:
                out.append(_linear_interp(x, ab[i-1], av[i-1], ab[i], av[i]))
                break
    return out

def anchor_copy(anchors: Dict[str, Any], which: str = "absorption") -> Tuple[List[int], List[float]]:
    """
    Extracts anchors.{bands_hz, which}. Returns (bands_hz, values).
    If no bands are specified in anchors, returns ([], values) and the caller must align.
    """
    vals = anchors.get(which)
    if not isinstance(vals, (list, tuple)) or not vals:
        raise ValueError(f"anchors.{which} missing or empty.")
    bands = anchors.get("bands_hz") or []
    if bands and len(bands) != len(vals):
        raise ValueError("anchors.bands_hz and anchors.absorption length mismatch.")
    return list(bands), [float(x) for x in vals]

# -----------------------------------------------------------------------------
# HIGH-LEVEL CONTROL FUNCTION (Stage 1 seed)
# -----------------------------------------------------------------------------

def compute_stage1_alpha(
    cfg: Dict[str, Any],
    bands: List[int],
) -> Dict[str, Dict[str, Any]]:
    """
    Deterministic Stage-1 statistical fit to generate the initial alpha(f) seed.
    - Reads materials.* from project.yaml
    - Pulls per-material anchors (absorption, optional anchors.bands_hz)
    - Interpolates to the provided 'bands' if needed
    - Applies constraints: clamp to [0,1] and enforce monotonic increasing (configurable)
    Returns:
      { material_key: { "bands_hz": [..], "alpha": [..] }, ... }
    """
    if not isinstance(bands, list) or not bands:
        raise ValueError("compute_stage1_alpha: 'bands' must be a non-empty list of frequencies (Hz).")

    stage1_alpha: Dict[str, Dict[str, Any]] = {}
    material_cfgs = cfg.get("materials", {}) or {}

    # Global/default constraints (override per material if present)
    default_constraints = (cfg.get("constraints") or {})
    default_enforce_mono = bool(default_constraints.get("enforce_monotonic_increasing", True))
    default_clamp_lo = float(default_constraints.get("alpha_min", 0.0))
    default_clamp_hi = float(default_constraints.get("alpha_max", 1.0))

    for mat_key, mat_spec in material_cfgs.items():
        anchors = (mat_spec.get("anchors") or {})
        if "absorption" not in anchors:
            # No seed for this material – skip politely
            continue

        # Per-material overrides
        m_constraints = (mat_spec.get("constraints") or {})
        enforce_mono = bool(m_constraints.get("enforce_monotonic_increasing", default_enforce_mono))
        clamp_lo = float(m_constraints.get("alpha_min", default_clamp_lo))
        clamp_hi = float(m_constraints.get("alpha_max", default_clamp_hi))

        # Extract anchors
        anchor_bands, anchor_vals = anchor_copy(anchors, which="absorption")

        # Align to target bands (interpolate if anchor bands provided; else assume already aligned)
        if anchor_bands:
            alpha_base = _interp_to_target_bands(anchor_bands, anchor_vals, bands)
        else:
            if len(anchor_vals) != len(bands):
                raise ValueError(
                    f"Material '{mat_key}': anchors list length ({len(anchor_vals)}) "
                    f"does not match target bands length ({len(bands)}). "
                    f"Provide anchors.bands_hz or align lengths."
                )
            alpha_base = list(anchor_vals)

        # Clamp + monotonicity
        alpha_clamped = clamp_vec(alpha_base, lo=clamp_lo, hi=clamp_hi)
        alpha_final = enforce_monotonic_increasing(alpha_clamped) if enforce_mono else alpha_clamped

        stage1_alpha[mat_key] = {
            "bands_hz": list(bands),
            "alpha": alpha_final,
        }

    if not stage1_alpha:
        raise RuntimeError(
            "Stage 1 produced no alpha spectra. Ensure project.yaml contains materials.* with anchors.absorption."
        )

    return stage1_alpha
