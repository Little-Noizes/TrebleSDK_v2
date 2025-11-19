# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# r"""
# run_stage1_and_forward.py — Hybrid metrics path

# Stage 1: generate constrained alpha(f) seed (src.materials.compute_stage1_alpha)
# Stage 2: apply alpha(f), run a Treble HYBRID simulation, then read metrics
#          from the latest *_Hybrid.json (NOT from IRs), compute a weighted
#          Huber loss, and write receipts/logs.

# CLI:
#   (venv_treble) PS> python -m src.run_stage1_and_forward --config .\configs\project.yaml --run_label seed_run_001
# """

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
#from src.utils import load_yaml
from src.tidy_metrics_helpers_v2 import (
    # --- FIX 1: ADD load_json to resolve NameError ---
    load_json, 
    load_targets, load_hybrid_pred, join_metrics, compute_weighted_huber_loss
)
import yaml
def load_yaml(path: Path) -> Dict[str, Any]:
    """Helper to safely load a YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ---- Treble import (robust pattern) ------------------------------------------
try:
    from treble_tsdk import treble
    print("Using treble_tsdk")
except ImportError:
    from treble_tsdk import tsdk_namespace as treble
    print("Using treble_tsdk2")

# ---- Stage 1 + Loss helper ---------------------------------------------------
from src.materials import compute_stage1_alpha
from src.metrics import weighted_huber_loss  # NOTE: predictions come from Hybrid JSON, not IRs


# =============================================================================
# Utilities
# =============================================================================

def _cfg_load(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    data = yaml.safe_load(txt)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return data


def _resolve(cfg_dir: Path, p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    pp = Path(p)
    return (cfg_dir / pp).resolve() if not pp.is_absolute() else pp.resolve()


def _get_bands(cfg: Dict[str, Any]) -> List[int]:
    bands = (cfg.get("bands", {}) or {}).get("list") \
        or (cfg.get("bands", {}) or {}).get("hz") \
        or (cfg.get("bands", {}) or {}).get("f_hz") \
        or []
    if not bands:
        raise ValueError("No bands found. Provide bands.list (preferred) or bands.hz/f_hz in YAML.")
    return [int(round(float(b))) for b in bands]


def _normalise_metric_key(name: str) -> str:
    """
    Normalise metric names and map common aliases so YAML does not need changes.
    Examples: EDT_s → edt, C50_dB → c50.
    The result is always lowercase.
    """
    raw = "".join(ch for ch in str(name).strip().upper() if ch.isalnum())
    aliases = {
        # Time constants
        "EDTS": "edt", "T20S": "t20", "T30S": "t30", "T60S": "t60", "RT60S": "t60",
        # Clarity / definition / centre time
        "C50DB": "c50", "C80DB": "c80", "D50": "d50", "TS": "ts",
    }
    return aliases.get(raw, raw.lower())


def _extract_receivers(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise receivers list from YAML."""
    rcvs = cfg.get("receivers")
    if rcvs is None:
        return []
    if isinstance(rcvs, dict):
        rcvs = rcvs.get("items", [])
    if not isinstance(rcvs, list):
        raise ValueError("'receivers' must be a list (or a dict containing 'items').")

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rcvs):
        if not isinstance(r, dict):
            raise ValueError(f"Receiver at index {i} must be an object.")
        label = r.get("label") or r.get("rcv_code") or r.get("name") or f"R{i+1}"
        pos = r.get("xyz_m")
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        else:
            try:
                x = float(r.get("x")); y = float(r.get("y")); z = float(r.get("z"))
            except (TypeError, ValueError):
                raise ValueError(f"Receiver '{label}': position must be 'xyz_m: [x,y,z]' or separate x/y/z.")
        rtype = (r.get("type") or "mono").strip().lower()
        if rtype not in ("mono", "spatial"):
            rtype = "mono"
        tags = r.get("tags") if isinstance(r.get("tags"), list) else []
        out.append({"label": str(label), "x": x, "y": y, "z": z, "type": rtype, "tags": tags})
    return out


def _get_or_import_model(project: treble.Project, name: str, obj_path: Path) -> Any:
    # Reuse by name (varies by SDK)
    for lister in ("get_model_by_name", "get_model"):
        if hasattr(project, lister):
            try:
                m = getattr(project, lister)(name) or getattr(project, lister)(model_name=name)
                if m is not None:
                    return m
            except Exception:
                pass
    for lister in ("list_models", "get_models"):
        if hasattr(project, lister):
            try:
                for m in getattr(project, lister)():
                    if getattr(m, "name", None) == name:
                        return m
            except Exception:
                pass
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")
    model = project.add_model(model_name=name, model_file_path=str(obj_path))
    if model is None:
        for m in getattr(project, "get_models", lambda: [])():
            if getattr(m, "name", None) == name:
                return m
        model = project.add_model(
            model_name=f"{name}_{datetime.now():%Y%m%d_%H%M%S}",
            model_file_path=str(obj_path)
        )
    if hasattr(model, "wait_for_model_processing"):
        model.wait_for_model_processing()
    return model


def _upload_or_reuse_cf2(tsdk: treble.TSDK, cf2_path: Path):
    """Create-or-reuse a CF2 directivity via source_directivity_library."""
    cf2_path = cf2_path.expanduser().resolve()
    if not cf2_path.exists():
        raise FileNotFoundError(f"CF2 not found: {cf2_path}")
    base_name = cf2_path.stem

    lib = getattr(tsdk, "source_directivity_library", None)
    if lib is None:
        raise RuntimeError("No source_directivity_library available in this SDK build.")

    # --- FIX: Attempt to find/reuse it FIRST ---
    try:
        # 1. Try common 'get by name' methods
        if hasattr(lib, "get_by_name"):
            d = lib.get_by_name(base_name)
            if d is not None:
                print(f"[debug] Reused existing CF2: {base_name}")
                return d, "directivity"
    except Exception:
        pass
        
    try:
        # 2. Try iterating through the list
        for d in lib.get_organization_directivities():
            if getattr(d, "name", None) == base_name:
                print(f"[debug] Reused existing CF2 (via list): {base_name}")
                return d, "directivity"
    except Exception:
        pass

    # --- Only if NOT found: Attempt to create it ---
    print(f"[debug] Uploading new CF2: {base_name}")
    try:
        sd_obj = lib.create_source_directivity(
            name=base_name,
            source_directivity_file_path=str(cf2_path),
            category=treble.SourceDirectivityCategory.amplified,
            sub_category=treble.SourceDirectivityAmplified.studio_and_broadcast_monitor,
            description="Uploaded by run_stage1_and_forward",
            manufacturer=None,
            correct_ir_by_on_axis_spl_default=True,
        )
        if sd_obj is not None:
            return sd_obj, "directivity"
    except Exception as e:
        # If creation fails, we must raise
        raise RuntimeError(f"Failed to create CF2 directivity named '{base_name}': {e}") from e

    raise RuntimeError(f"Failed to create or find CF2 directivity named '{base_name}'.")

def _build_source_from_cf2(cf2_obj, cf2_type: str, pos: treble.Point3d) -> treble.Source:
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in getattr(cf2_obj, "name", "CF2_Source"))
    try:
        return treble.Source.make_directive(location=pos, label=safe_label, source_directivity=cf2_obj)
    except Exception:
        src = treble.Source.make_omni(label=safe_label, position=pos)
        if hasattr(cf2_obj, "id"):
            try:
                src.directivity_id = cf2_obj.id
            except Exception:
                try:
                    sp = treble.SourceProperties(directivity_id=cf2_obj.id)
                    src.source_properties = sp
                except Exception:
                    pass
        return src


def _apply_alpha_to_materials(
    tsdk: treble.TSDK,
    stage1_alpha: Dict[str, Dict[str, Any]],
    *,
    allow_create: bool = True
) -> Dict[str, str]:
    """
    Apply alpha(f) per material to the org material library.
    Returns a redirect map original_name -> effective_name_used.
    """
    lib = getattr(tsdk, "material_library", None)
    if lib is None:
        raise RuntimeError("TSDK.material_library not available.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_redirect: Dict[str, str] = {}

    for mat_name, spec in stage1_alpha.items():
        bands = spec.get("bands_hz")
        alpha = spec.get("alpha")
        if not bands or not alpha or len(bands) != len(alpha):
            raise ValueError(f"Invalid alpha spec for '{mat_name}': bands/alpha mismatch.")

        # 1) Try update-style APIs (varies by SDK)
        updated = False
        for fn in ("update", "update_absorption", "set_absorption", "update_material"):
            if hasattr(lib, fn):
                try:
                    getattr(lib, fn)(
                        material_id=mat_name,   # adjust to id if required on your build
                        absorption_coefficients=alpha,
                        band_frequencies=bands
                    )
                    name_redirect[mat_name] = mat_name
                    updated = True
                    break
                except Exception:
                    pass
        if updated:
            continue

        # 2) Create a NEW uniquely named material
        new_name = f"{mat_name}__S1_{timestamp}"
        md = treble.MaterialDefinition(
            name=new_name,
            description=f"alpha(f) by Stage1 seed @ {timestamp}",
            category=getattr(treble.MaterialCategory, "other"),
            default_scattering=0.1,
            material_type=treble.MaterialRequestType.full_octave_absorption,
            coefficients=alpha,
        )
        fitted = lib.perform_material_fitting(md)
        created = lib.create(fitted)
        effective_name = getattr(created, "name", new_name)
        name_redirect[mat_name] = effective_name

    return name_redirect

def _apply_ga_params_to_materials(cfg: dict, ga_params: dict) -> dict:
    alpha_map = ga_params.get("alpha", {}) or {}
    scatter_map = ga_params.get("scatter", {}) or {}
    mats = cfg.get("materials", {}) or {}

    # apply α(f)
    for m, spec in alpha_map.items():
        if m not in mats: 
            continue
        node = mats[m] or {}
        anchors = (node.get("anchors") or {})
        anchors["absorption"] = list(map(float, spec["alpha"]))
        node["anchors"] = anchors
        # lock α if desired
        node["optimise_absorption"] = False
        node["deviation_groups"] = {
            "low":  {"bands": [63, 125], "deviation_pct": 0.0},
            "mid":  {"bands": [250, 500, 1000], "deviation_pct": 0.0},
            "high": {"bands": [2000, 4000, 8000], "deviation_pct": 0.0},
        }
        mats[m] = node

    # apply scattering
    for m, sval in scatter_map.items():
        if m not in mats:
            continue
        node = mats[m] or {}
        anchors = (node.get("anchors") or {})
        anchors["scattering"] = float(sval)
        node["anchors"] = anchors
        node["optimise_scattering"] = True
        mats[m] = node

    cfg["materials"] = mats
    return cfg

# --- NO GLOBAL CODE HERE ---

def _load_hybrid_pred_lookup(run_dir: Path) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Scan run_dir for latest *_Hybrid.json and return:
      { rcv_label: { METRIC_CANON: { band_hz:int -> value:float } } }
    """
    import os
    candidates = sorted(run_dir.glob("*_Hybrid.json"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"No *_Hybrid.json found in {run_dir}")
    hybrid_path = candidates[-1]
    print(f"[debug] using Hybrid JSON: {hybrid_path.name}")

    data = json.loads(hybrid_path.read_text(encoding="utf-8"))
    recvs = data.get("receivers") or {}
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    for r_id, payload in recvs.items():
        label = payload.get("label") or r_id
        params = payload.get("parameters") or {}
        mblock: Dict[str, Dict[int, float]] = {}
        for m_key, band_map in params.items():
            if not isinstance(band_map, dict):
                continue
            m_norm = _normalise_metric_key(m_key)
            per_band: Dict[int, float] = {}
            for fk, vv in band_map.items():
                try:
                    per_band[int(round(float(fk)))] = float(vv)
                except Exception:
                    continue
            if per_band:
                mblock[m_norm] = per_band
        out[str(label)] = mblock
    if not out:
        raise RuntimeError(f"Hybrid JSON parsed but contained no receiver parameters: {hybrid_path.name}")
    return out


# =============================================================================
# Core pipeline
# =============================================================================


def _build_targets_lookup(raw: Dict[str, Any], default_bands: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Convert the ground-truth JSON structure into a lookup by receiver label."""

    # Some datasets expose a flat mapping already. Honour that first.
    if raw and all(isinstance(v, dict) for v in raw.values()):
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for key, val in raw.items():
            metrics_map = val.get("metrics") if isinstance(val, dict) else None
            if not isinstance(metrics_map, dict):
                continue
            mm: Dict[str, Dict[str, float]] = {}
            for m_name, values in metrics_map.items():
                norm = _normalise_metric_key(m_name)
                vec: Dict[str, float] = {}
                if isinstance(values, list):
                    for i in range(min(len(default_bands), len(values))):
                        try:
                            vec[str(int(default_bands[i]))] = float(values[i])
                        except (TypeError, ValueError):
                            continue
                elif isinstance(values, dict):
                    for key2, val2 in values.items():
                        try:
                            vec[str(int(round(float(key2))))] = float(val2)
                        except (TypeError, ValueError):
                            continue
                if vec:
                    mm[norm] = vec
            if mm:
                out[key] = mm
        if out:
            return out

    points = raw.get("points") if isinstance(raw.get("points"), list) else []
    freq = raw.get("frequency_bands") or raw.get("bands") or []
    if isinstance(freq, list) and freq:
        freq_vec = [int(round(float(f))) for f in freq]
    else:
        freq_vec = list(default_bands)

    lookup: Dict[str, Dict[str, Dict[str, float]]] = {}
    for entry in points:
        if not isinstance(entry, dict):
            continue
        label = entry.get("rcv_code") or entry.get("label") or entry.get("name")
        if not label:
            continue
        metrics_map = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        per_metric: Dict[str, Dict[str, float]] = {}
        for metric_name, values in metrics_map.items():
            norm = _normalise_metric_key(str(metric_name))
            vec: Dict[str, float] = {}
            if isinstance(values, list):
                for idx, freq_hz in enumerate(freq_vec):
                    if idx >= len(values):
                        break
                    try:
                        vec[str(int(freq_hz))] = float(values[idx])
                    except (TypeError, ValueError):
                        continue
            elif isinstance(values, dict):
                for key2, val2 in values.items():
                    try:
                        freq_hz = int(round(float(key2)))
                        vec[str(int(freq_hz))] = float(val2)
                    except (TypeError, ValueError):
                        continue
            if vec:
                per_metric[norm] = vec
        if per_metric:
            lookup[str(label)] = per_metric

    return lookup


# --- 1. CORRECTED FUNCTION SIGNATURE AND FILE SAVING LOGIC ---
def run_stage1_and_forward_sim(
    tsdk: Any,
    cfg: Dict[str, Any],        # Accepts the loaded config dictionary
    run_label: str,
    cfg_path: Path,              # Accepts the path for resolving relatives
    params_to_save: Optional[Dict[str, Any]] = None # <--- ADDED ARGUMENT
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Stage-1 seed → apply materials → Stage-2 forward sim → read Hybrid JSON → vector loss.
    Returns: (scalar_loss, detailed_rows_log)
    """
    # NOTE: The YAML configuration is already loaded and modified by _main()

    cfg_dir = cfg_path.parent
    bands = _get_bands(cfg)

    # ---- Paths (resolve RELATIVE to YAML) -----------------------------------
    obj_path = _resolve(cfg_dir, (cfg.get("paths", {}) or {}).get("model_obj"))
    cf2_path = _resolve(cfg_dir, (cfg.get("paths", {}) or {}).get("directivity_cf2"))
    workdir = _resolve(cfg_dir, (cfg.get("paths", {}) or {}).get("working_dir")) or (cfg_dir / "results")
    gt_json = _resolve(cfg_dir, (cfg.get("paths", {}) or {}).get("ground_truth_json"))

    if not obj_path or not obj_path.exists():
        raise FileNotFoundError(f"OBJ path invalid: {obj_path}")
    if not cf2_path or not cf2_path.exists():
        raise FileNotFoundError(f"CF2 path invalid: {cf2_path}")
    if not gt_json or not gt_json.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_json}")

    workdir.mkdir(parents=True, exist_ok=True)
    run_dir = workdir / f"{run_label}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Output will be saved to: {run_dir.resolve()} ---")

    # --- ADDED LOGIC TO SAVE STAGE 1 ALPHA ---
    if params_to_save:
        (run_dir / "stage1_alpha.json").write_text(
            json.dumps(params_to_save, indent=2), encoding="utf-8"
        )
        print(" Saved initial Stage 1 alpha parameters to stage1_alpha.json")
    # -----------------------------------------

    # ---- Stage 1: compute seed alpha(f) -------------------------------------
    print("\n--- 1) Stage 1: Statistical alpha(f) seed ---")
    
    # We remove the old Stage 1 logic from run_stage1_and_forward_sim
    # to enforce that material modification (GA or seed) happens ONLY in _main().
    # The simulation will now use the materials present in the 'cfg' passed from _main().
    
    # ---- Apply materials to org library -------------------------------------
    print("--- 2) Materials applied to Treble library ---")
    
    # CRITICAL: Actually apply the alpha(f) values to the Treble material library
    # This ensures GA-generated or Stage1 values are used in the simulation
    if params_to_save:
        print("[debug] Applying alpha(f) to Treble material library...")
        name_redirect = _apply_alpha_to_materials(tsdk, params_to_save, allow_create=True)
        print(f"[debug] Applied {len(name_redirect)} materials to library")
    else:
        # Fallback: just use material names as-is
        material_names_to_check = list(cfg.get("materials", {}).keys())
        name_redirect: Dict[str, str] = {name: name for name in material_names_to_check}
    
    # ---- Project & Model -----------------------------------------------------
    project_name = (cfg.get("project", {}) or {}).get("name") or "Treble_Project"
    model_name = (cfg.get("model", {}) or {}).get("name") or Path(obj_path).stem

    print("[debug] creating project and model…")
    proj = tsdk.get_or_create_project(name=project_name)
    model = _get_or_import_model(proj, model_name, obj_path)
    print(f"Project: {proj.name} | Model: {getattr(model, 'name', model_name)}")

    # ---- Source & Receivers --------------------------------------------------
    print("--- 3) Building source & receivers ---")
    src_cfg = (cfg.get("source", {}) or {})
    s_pos = src_cfg.get("position_m") or src_cfg.get("xyz_m")
    if not s_pos or len(s_pos) != 3:
        raise ValueError("source.position_m (or xyz_m) must be [x,y,z] in metres.")
    src_pos = treble.Point3d(float(s_pos[0]), float(s_pos[1]), float(s_pos[2]))

    print("[debug] uploading CF2…")
    cf2_obj, cf2_type = _upload_or_reuse_cf2(tsdk, cf2_path)
    print("[debug] CF2 uploaded.")

    src = _build_source_from_cf2(cf2_obj, cf2_type, src_pos)

    rcv_cfgs = _extract_receivers(cfg)
    rcv_list: List[treble.Receiver] = []
    for r in rcv_cfgs:
        p = treble.Point3d(r["x"], r["y"], r["z"])
        if r["type"] == "spatial" and hasattr(treble.Receiver, "make_spatial"):
            rcv_list.append(treble.Receiver.make_spatial(position=p, label=r["label"]))
        else:
            rcv_list.append(treble.Receiver.make_mono(position=p, label=r["label"]))
    if not rcv_list:
        raise RuntimeError("No receivers defined in YAML.")
    print(f" Source '{src.label}' and {len(rcv_list)} receiver(s) ready.")

    # ---- Material assignments (layer/tag → material OBJECT) ------------------
    tag_map = (cfg.get("tag_to_material") or {})
    assignments: List[treble.MaterialAssignment] = []

    matlib = getattr(tsdk, "material_library", None)
    if matlib is None:
        raise RuntimeError("TSDK.material_library is not available in this SDK build.")

    def _get_material_obj_by_name(name: str):
        if hasattr(matlib, "get_by_name"):
            try:
                obj = matlib.get_by_name(name)
                if obj is not None:
                    return obj
            except Exception:
                pass
        for list_fn in ("list_materials", "get_materials", "list_all", "list"):
            if hasattr(matlib, list_fn):
                try:
                    for m in getattr(matlib, list_fn)():
                        if getattr(m, "name", None) == name:
                            return m
                except Exception:
                    pass
        return None

    for layer_name, mat_name in tag_map.items():
        if isinstance(mat_name, dict):
            mat_name = mat_name.get("name")
        effective_name = name_redirect.get(mat_name, mat_name)
        mat_obj = _get_material_obj_by_name(effective_name)
        if mat_obj is None:
            raise RuntimeError(
                f"Material '{effective_name}' not found in library. (Original YAML name: '{mat_name}')"
            )
        assignments.append(treble.MaterialAssignment(layer_name=str(layer_name), material=mat_obj))

    # ---- Simulation definition ------------------------------------------------
    calc = (cfg.get("calculation", {}) or {})
    term = (calc.get("termination", {}) or {})
    energy_db = float(term.get("energy_decay_threshold_db", 35.0))
    crossover = calc.get("crossover_frequency_hz", 720)

    sim_settings = treble.SimulationSettings(speed_of_sound=343.0, ambisonics_order=0)

    sim_def = treble.SimulationDefinition(
        name=run_label,
        simulation_type=treble.SimulationType.hybrid,
        model=model,
        receiver_list=rcv_list,
        source_list=[src],
        material_assignment=assignments,
        energy_decay_threshold=energy_db,
        crossover_frequency=crossover,
        simulation_settings=sim_settings,
        tags=["stage1_seed", "forward_eval"],
    )

    # ---- Run simulation ------------------------------------------------------
    print("--- 4) Starting simulation ---")
    sim_def.name = f"{run_label}_{datetime.now():%Y%m%d_%H%M%S}"

    def _safe_add_and_start(defn):
        try:
            s = proj.add_simulation(definition=defn)
            proj.start_simulations()
            return s
        except Exception as e:
            msg = str(e)
            if "Duplicate simulation names" in msg or "duplicate simulation" in msg.lower():
                alt_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[-8:]
                defn.name = f"{run_label}_{alt_suffix}"
                print(f"[sim] duplicate name; retrying with '{defn.name}'")
                s = proj.add_simulation(definition=defn)
                proj.start_simulations()
                return s
            raise

    sim = _safe_add_and_start(sim_def)

    MAX_WAIT_SECONDS = int(calc.get("timeout_seconds", 1800))
    poll_interval = 30
    print(f"[sim] waiting up to {MAX_WAIT_SECONDS}s for completion…")

    sim_id = getattr(sim, "id", None) or getattr(sim, "simulation_id", None) or sim_def.name

    def _reload_sim():
        for fn in ("get_simulation_by_id", "get_simulation", "get_simulation_by_name"):
            if hasattr(proj, fn):
                try:
                    s = getattr(proj, fn)(sim_id)
                    if s is not None:
                        return s
                except Exception:
                    pass
        for lister in ("get_simulations", "list_simulations"):
            if hasattr(proj, lister):
                try:
                    for s in getattr(proj, lister)():
                        sid = getattr(s, "id", None) or getattr(s, "simulation_id", None) or getattr(s, "name", None)
                        if sid == sim_id or sid == sim_def.name:
                            return s
                except Exception:
                    pass
        return sim

    for fn in ("wait_for_estimate", "wait_for_token_cost"):
        if hasattr(proj, fn):
            try:
                print(f"[sim] {fn}…")
                getattr(proj, fn)()
            except Exception:
                pass

    start_time = time.time()
    last_status = None
    empty_status_count = 0

    while True:
        elapsed = int(time.time() - start_time)
        sim = _reload_sim()
        if hasattr(sim, "refresh"):
            try:
                sim.refresh()
            except Exception:
                pass

        try:
            status = sim.get_status()
        except AttributeError:
            status = getattr(sim, "status", "")
        s = (str(status) or "").lower()

        if status != last_status:
            print(f"[{elapsed:5d}s] status  {status}")
            last_status = status
        else:
            print(f"[{elapsed:5d}s] still {status or '<no-status>'}…")

        if not str(status):
            empty_status_count += 1
            if empty_status_count in (3, 6):
                try:
                    print(f"[{elapsed:5d}s] nudging: start_simulations()")
                    proj.start_simulations()
                except Exception:
                    pass
        else:
            empty_status_count = 0

        if any(k in s for k in ("complete", "finished", "success")):
            print(f"[{elapsed:5d}s]  Simulation finished.")
            break
        if any(k in s for k in ("failed", "error", "canceled", "cancelled")):
            raise RuntimeError(f" Simulation failed with status: {status}")
        if elapsed > MAX_WAIT_SECONDS:
            raise TimeoutError(f"Simulation timed out after {MAX_WAIT_SECONDS}s (last status={status})")

        time.sleep(poll_interval)

    # ---- Results & metrics ---------------------------------------------------
        # ---- Results & metrics (Hybrid JSON path) -------------------------------
    print("--- 5) Reading results & computing metrics ---")
    # This ensures the *_Hybrid.json is downloaded into run_dir
    _ = sim.get_results_object(results_directory=str(run_dir))

    # Load targets and predictions as tidy tables
    df_tgt  = load_targets(str(gt_json), default_bands=bands)
    df_pred = load_hybrid_pred(run_dir)

    # Config → metric list + bands to keep
    metrics_req: List[str] = (cfg.get("metrics", {}) or {}).get("objective_metrics") or []
    # FIX for case-sensitivity: Canonicalize requested metrics to lowercase 
    # to match the lowercase metric names stored in the DataFrames.
    metrics_req = [_normalise_metric_key(m) for m in metrics_req]
    
    band_min = (cfg.get("bands", {}) or {}).get("optimise_from_hz")
    band_max = (cfg.get("bands", {}) or {}).get("optimise_to_hz")
    band_set = [f for f in bands if (band_min is None or f >= int(band_min))
                              and (band_max is None or f <= int(band_max))]

    # The receivers we actually simulated (labels)
    keep_receivers = [getattr(r, "label", None) or getattr(r, "name", "") for r in rcv_list]

    # Inner-join (receiver, metric, band)
    df_join = join_metrics(
        df_tgt, df_pred,
        keep_receivers=keep_receivers,
        keep_metrics=metrics_req,
        keep_bands=band_set,
    )

    # Debug if no overlap
    if df_join.empty:
        diag = {
            "requested_metrics": metrics_req,
            "requested_bands_hz": band_set,
            "requested_receivers": keep_receivers,
            "targets_keys_summary": {
                "receivers": sorted(df_tgt["receiver"].unique().tolist()),
                "metrics": sorted(df_tgt["metric"].unique().tolist()),
                "bands": sorted(map(int, df_tgt["band_hz"].unique().tolist())),
            },
            "preds_keys_summary": {
                "receivers": sorted(df_pred["receiver"].unique().tolist()),
                "metrics": sorted(df_pred["metric"].unique().tolist()),
                "bands": sorted(map(int, df_pred["band_hz"].unique().tolist())),
            },
            "hint": "Receiver labels, canonical metric keys (all lowercase), and bands must overlap.",
        }
        (run_dir / "debug_zero_join.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        raise RuntimeError(
            "No overlapping rows after join. See debug_zero_join.json for details."
        )

    # Write tidy CSVs for transparency
    df_tgt.to_csv(run_dir / "targets_tidy.csv", index=False)
    df_pred.to_csv(run_dir / "pred_tidy.csv", index=False)
    df_join.to_csv(run_dir / "joined.csv", index=False)

    # Weights & Huber delta from YAML (any casing is fine; helpers canonicalise)
    weights_cfg = (cfg.get("metrics", {}) or {}).get("weights") or {}
    deltas_cfg  = (cfg.get("metrics", {}) or {}).get("huber_delta") or {}

    print("--- 6) Computing weighted Huber loss (Hybrid) ---")
    scalar_loss = compute_weighted_huber_loss(
        df_join,
        weights=weights_cfg,
        huber_delta=deltas_cfg,
        reduction="mean",
    )

    # If the helper produced a debug_nan_rows.csv in CWD, move it into run_dir
    from pathlib import Path as _P
    dbg = _P("debug_nan_rows.csv")
    if dbg.exists():
        try:
            dbg.rename(run_dir / "debug_nan_rows.csv")
        except Exception:
            pass

    # Build detailed rows (nice for plotting later)
    detailed_rows = [
        {
            "rcv_code": str(r),
            "metric":  str(m),
            "f_hz":    int(f),
            "target_val": float(t),
            "predicted_val": float(p),
            "error":   float(p - t),
        }
        for (r, m, f, t, p) in zip(
            df_join["receiver"].tolist(),
            df_join["metric"].tolist(),
            df_join["band_hz"].tolist(),
            df_join["target"].astype(float).tolist(),
            df_join["prediction"].astype(float).tolist(),
        )
    ]
    
    # ---- Write receipts ------------------------------------------------------
    receipt = {
        "project": project_name,
        "model": getattr(model, "name", model_name),
        "run_label": sim_def.name,  # actual unique sim name
        "bands_hz": bands,
        "metrics": metrics_req,
        "termination": {"energy_decay_threshold_dB": energy_db, "crossover_frequency_Hz": crossover},
        "receivers": [getattr(r, "label", "") for r in rcv_list],
        # Removed reference to stage1_alpha.json because it's only generated for initial seed run.
        "results_dir": str(run_dir.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "loss": float(scalar_loss),
    }
    (run_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (run_dir / "detailed_results_log.json").write_text(json.dumps(detailed_rows, indent=2), encoding="utf-8")

    print(f" Done. Loss={scalar_loss:.6f} | Receipt: {run_dir.name}/run_receipt.json")
    return float(scalar_loss), detailed_rows


# =============================================================================
# CLI
# =============================================================================
# --- 2. CORRECTED _main() FUNCTION ---

def _main():
    # 1. Argument Parsing 
    ap = argparse.ArgumentParser(description="Run Stage 1 seed + forward HYBRID sim; read Hybrid JSON; compute scalar loss.")
    ap.add_argument("--config", required=True, help="Path to project.yaml")
    ap.add_argument("--run_label", required=False, default=f"stage1_2_{datetime.now():%Y%m%d_%H%M%S}", help="Run label/name")
    ap.add_argument("--params_in_file", default=None, help="JSON with {'alpha': {...}, 'scatter': {...}}")
    args = ap.parse_args()
    
    # 2. CONFIG LOADING & PARAMETER OVERRIDE (ALL LOGIC MOVED INSIDE _main)
    cfg_path = Path(args.config) # Define path here
    cfg = load_yaml(cfg_path) 

    # --- NEW: Variable to capture alpha parameters for saving ---
    stage1_params_to_save = None
    
    if args.params_in_file:
        print(f"Reading parameters from GA file: {args.params_in_file}")
        ga_params = load_json(Path(args.params_in_file)) 
        cfg = _apply_ga_params_to_materials(cfg, ga_params)
        # CRITICAL FIX: Extract alpha params so they get applied to Treble library
        stage1_params_to_save = ga_params.get("alpha", {}) 
    else:
        print("Running Stage 1 statistical seed.")
        # When running a standard seed, run_stage1_and_forward_sim will 
        # use the existing logic to compute the seed alpha if not supplied by GA.
        # Since the GA requires the config to be modified *before* the sim call, 
        # we handle the initial seed here for a clean split:
        bands = _get_bands(cfg)
        stage1_alpha = compute_stage1_alpha(cfg, bands)
        
        # Apply the seed to the config dictionary
        cfg = _apply_ga_params_to_materials(cfg, {"alpha": stage1_alpha, "scatter": {}})
        
        # --- NEW: Capture alpha parameters here for saving later ---
        stage1_params_to_save = stage1_alpha
        
    # 3. Simulation Call (Single, Corrected Call)
    tsdk = treble.TSDK()
    # Note: Passing the dictionary 'cfg' and the path 'cfg_path'
    loss, log = run_stage1_and_forward_sim(
        tsdk, cfg, args.run_label, 
        cfg_path=cfg_path,
        params_to_save=stage1_params_to_save # <--- PASS THIS NEW ARGUMENT
    ) 
    
    print("\n=== SUMMARY ===")
    print(f"Loss: {loss:.6f}")
    print(f"Rows: {len(log)}")


if __name__ == "__main__":
    _main()