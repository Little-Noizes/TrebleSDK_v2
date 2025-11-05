## `src/cfg.py`

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
from .io_utils import read_yaml, ensure_dir, new_run_id

REQUIRED_TOP = ["project", "paths", "bands", "receivers", "source", "materials", "tag_to_material", "metrics", "calculation", "optimisation"]
OCTAVE_FREQS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]


def load_config(path: Path | str) -> dict:
    cfg = read_yaml(path)
    cfg["_cfg_path"] = str(Path(path).resolve())
    cfg["_run_id"] = new_run_id()
    _resolve_at_refs(cfg)   # expand @paths.* to strings
    _normalise_paths(cfg)
    _derive_band_masks(cfg)
    return cfg


def _require_keys(d: dict, keys: list[str], ctx: str):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing keys in {ctx}: {missing}")


def _normalise_paths(cfg: dict):
    # Ensure working/temp directories exist early
    ensure_dir(Path(cfg["paths"]["working_dir"]))
    ensure_dir(Path(cfg["paths"]["temp_dir"]))


def _resolve_at_refs(cfg: dict):
    # resolve values like "@paths.directivity_cf2"
    def resolve_val(v: Any):
        if isinstance(v, str) and v.startswith("@"):
            parts = v[1:].split(".")
            node = cfg
            for p in parts:
                node = node[p]
            return node
        return v
    # walk minimal nodes that may contain refs
    src = cfg.get("source", {})
    if "directivity_file" in src:
        src["directivity_file"] = resolve_val(src["directivity_file"]) 


def _validate_paths(cfg: dict):
    p = cfg["paths"]
    for k in ("model_obj", "ground_truth_json"):
        fp = Path(p[k]).resolve()
        if not fp.exists():
            raise ValueError(f"Path not found for paths.{k}: {fp}")
    if cfg.get("source", {}).get("kind") == "directive":
        cf2 = Path(cfg["source"]["directivity_file"]).resolve()
        if not cf2.exists():
            raise ValueError(f"CF2 not found: {cf2}")


def _validate_receivers(cfg: dict):
    recs = cfg.get("receivers", [])
    if not recs:
        raise ValueError("Receivers list empty")
    seen_codes = set()
    for r in recs:
        for key in ("id", "name", "rcv_code", "xyz_m"):
            if key not in r:
                raise ValueError(f"Receiver missing '{key}': {r}")
        if len(r["xyz_m"]) != 3:
            raise ValueError(f"Receiver xyz_m must be length 3: {r}")
        code = r["rcv_code"]
        if code in seen_codes:
            raise ValueError(f"Duplicate receiver code: {code}")
        seen_codes.add(code)


def _validate_materials(cfg: dict):
    mats: Dict[str, Any] = cfg.get("materials", {})
    if not mats:
        raise ValueError("No materials defined")
    # check tag_to_materials all map to a defined key
    mapping = cfg.get("tag_to_material", {})
    for tag, key in mapping.items():
        if key not in mats:
            raise ValueError(f"tag_to_material maps '{tag}' → '{key}' which is not defined in materials")
    # each material must have anchors with 8 octave bands if type is full_octave_absorption
    for key, m in mats.items():
        t = m.get("type", "full_octave_absorption")
        anchors = m.get("anchors", {})
        a_abs = anchors.get("absorption")
        a_scat = anchors.get("scattering")
        if t == "full_octave_absorption":
            for arr, nm in ((a_abs, "anchors.absorption"), (a_scat, "anchors.scattering")):
                if arr is None:
                    raise ValueError(f"Material '{key}' missing {nm}")
                if len(arr) != 8:
                    raise ValueError(f"Material '{key}' {nm} must have 8 values for octave bands {OCTAVE_FREQS}")
        # optimise flags sanity
        for flag in ("optimise", "optimise_absorption", "optimise_scattering"):
            if flag in m and not isinstance(m[flag], bool):
                raise ValueError(f"Material '{key}' field '{flag}' must be boolean")


def _validate_metrics(cfg: dict):
    m = cfg.get("metrics", {})
    _require_keys(m, ["objective_metrics"], "metrics")
    if not m["objective_metrics"]:
        raise ValueError("metrics.objective_metrics must have at least one metric")
    # Optional weights/huber_delta
    for opt in ("weights", "huber_delta"):
        if opt in m and not isinstance(m[opt], dict):
            raise ValueError(f"metrics.{opt} must be a mapping of metric name -> value")


def _derive_band_masks(cfg: dict):
    b = cfg.get("bands", {})
    kind = b.get("kind", "third")
    f_hz: List[int] = b.get("f_hz", [])
    if kind not in ("third", "octave"):
        raise ValueError("bands.kind must be 'third' or 'octave'")
    if not f_hz:
        raise ValueError("bands.f_hz must be provided (list of centre frequencies in Hz)")
    min_hz = b.get("optimise_from_hz", 63)
    idx = [i for i, f in enumerate(f_hz) if f >= min_hz]
    if not idx:
        raise ValueError(f"optimise_from_hz {min_hz} is higher than highest band {max(f_hz)}")
    cfg["_bands"] = {
        "kind": kind,
        "f_hz": f_hz,
        "optimise_from_hz": min_hz,
        "opt_start_index": idx[0]
    }


def validate_config(cfg: dict) -> None:
    _require_keys(cfg, REQUIRED_TOP, "root")
    _validate_paths(cfg)
    _validate_receivers(cfg)
    _validate_materials(cfg)
    _validate_metrics(cfg)
    # Prepare run directory early
    ensure_dir(Path(cfg["paths"]["working_dir"]) / cfg["_run_id"])
