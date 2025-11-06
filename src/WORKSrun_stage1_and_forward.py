#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
run_stage1_and_forward.py
- Stage 1: generate constrained alpha(f) seed for all materials (src.materials.compute_stage1_alpha)
- Stage 2: apply alpha(f), run Treble hybrid sim, compute ONLY metrics listed in YAML,
           produce a single scalar loss via vector weighted Huber, and save logs.

CLI:
  (venv_treble) PS> python -m src.run_stage1_and_forward --config .\configs\project.yaml --run_label seed_run_001
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import yaml

# ---- Treble import (robust pattern, matches your working scripts) ------------
try:
    from treble_tsdk import treble
    print("Using treble_tsdk")
except ImportError:
    from treble_tsdk import tsdk_namespace as treble
    print("Using treble_tsdk2")

# ---- Stage 1 + Metrics helpers (the real modules you created) ----------------
from src.materials import compute_stage1_alpha
from src.metrics import compute_metrics_from_ir, weighted_huber_loss


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
    """Normalise metric names (e.g. ``EDT_s`` -> ``EDT``) for lookups."""
    return "".join(ch for ch in name.strip().upper() if ch.isalnum())


def _extract_receivers(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accepts receivers under cfg['receivers']; supports [{'xyz_m':[x,y,z], 'label':..., 'type':...}, ...]
        and also {'x':..,'y':..,'z':..}. Returns normalized list."""
    rcvs = cfg.get("receivers")
    if rcvs is None:
        return []
    if isinstance(rcvs, dict):  # allow {"items":[...]}
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
        # duplicate name fallback
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
    """
    Create-or-reuse via source_directivity_library.
    If create fails due to duplicate name, scan and reuse the existing one.
    """
    cf2_path = cf2_path.expanduser().resolve()
    if not cf2_path.exists():
        raise FileNotFoundError(f"CF2 not found: {cf2_path}")
    base_name = cf2_path.stem

    lib = getattr(tsdk, "source_directivity_library", None)
    if lib is None:
        raise RuntimeError("No source_directivity_library available in this SDK build.")

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
    except Exception:
        # likely duplicate name; reuse existing by name
        pass

    # Reuse by name if exists
    try:
        for d in lib.get_organization_directivities():
            if getattr(d, "name", None) == base_name:
                return d, "directivity"
    except Exception:
        pass

    # If we got here, we could not create nor find by name.
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
    Apply alpha(f) per material to the org material library:
      stage1_alpha: { material_key: {"bands_hz":[...], "alpha":[...]} }

    Strategy:
      1) Try in-place update if this SDK exposes it.
      2) Else, create a NEW uniquely named material and return a name-redirect map:
         { original_name: new_material_name }
      The caller can use this to redirect MaterialAssignments to the new names.

    Returns:
      name_redirect: Dict[original_name -> effective_name_used]
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
                        material_id=mat_name,   # if your build needs an ID: swap to get_by_name->id
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

        # 2) No update path — create a NEW uniquely named material
        #    Avoid duplicate-name errors by checking existence first.
        existing = None
        if hasattr(lib, "get_by_name"):
            try:
                existing = lib.get_by_name(mat_name)
            except Exception:
                existing = None

        if existing is not None and not allow_create:
            # We refuse to create; just reuse existing name.
            name_redirect[mat_name] = mat_name
            continue

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
                    for key, val in values.items():
                        try:
                            vec[str(int(round(float(key))))] = float(val)
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
                for key, val in values.items():
                    try:
                        freq_hz = int(round(float(key)))
                        vec[str(freq_hz)] = float(val)
                    except (TypeError, ValueError):
                        continue
            if vec:
                per_metric[norm] = vec
        if per_metric:
            lookup[str(label)] = per_metric

    return lookup


def run_stage1_and_forward_sim(
    tsdk: Any,
    config_path: Path,
    run_label: str
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Stage-1 seed → apply materials → Stage-2 forward sim → per-IR metrics → vector loss.
    Returns:
        (scalar_loss, detailed_rows_log)
    """
    cfg_path = Path(config_path).expanduser().resolve()
    cfg = _cfg_load(cfg_path)
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

    # ---- Stage 1: compute seed alpha(f) -------------------------------------
    print("\n--- 1) Stage 1: Statistical alpha(f) seed ---")
    stage1_alpha = compute_stage1_alpha(cfg, bands)
    (run_dir / "stage1_alpha.json").write_text(json.dumps(stage1_alpha, indent=2), encoding="utf-8")
    print("Stage 1 alpha seed saved.")

    # ---- Apply materials to org library -------------------------------------
    print("--- 2) Applying materials to Treble library ---")
    name_redirect = _apply_alpha_to_materials(tsdk, stage1_alpha, allow_create=True)

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
    print(f"✔ Source '{src.label}' and {len(rcv_list)} receiver(s) ready.")

    # ---- Material assignments (layer/tag → material OBJECT, using redirect) -----
    tag_map = (cfg.get("tag_to_material") or {})
    assignments: List[treble.MaterialAssignment] = []

    # Access material library once
    matlib = getattr(tsdk, "material_library", None)
    if matlib is None:
        raise RuntimeError("TSDK.material_library is not available in this SDK build.")

    def _get_material_obj_by_name(name: str):
        """Return a material DTO/object from the library by name."""
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
                f"Material '{effective_name}' not found in library. "
                f"(Original YAML name: '{mat_name}')"
            )

        assignments.append(
            treble.MaterialAssignment(layer_name=str(layer_name), material=mat_obj)
        )

    # ---- Simulation definition ------------------------------------------------
    calc = (cfg.get("calculation", {}) or {})
    term = (calc.get("termination", {}) or {})
    energy_db = float(term.get("energy_decay_threshold_db", 35.0))
    crossover = calc.get("crossover_frequency_hz", 720)

    sim_settings = treble.SimulationSettings(
        speed_of_sound=343.0,
        ambisonics_order=0,
    )

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

    # Always uniquify the sim name to avoid duplicate-name errors
    unique_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_def.name = f"{run_label}_{unique_suffix}"

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
    poll_interval = 30  # seconds
    print(f"[sim] waiting up to {MAX_WAIT_SECONDS}s for completion…")

    # Try to capture a stable identifier for reloading the sim each poll
    sim_id = getattr(sim, "id", None) or getattr(sim, "simulation_id", None) or sim_def.name

    def _reload_sim():
        """Return a fresh sim object from the project, if possible."""
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

    # Optional: try to sync cost info if available
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
            print(f"[{elapsed:5d}s] status → {status}")
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
            print(f"[{elapsed:5d}s] ✅ Simulation finished.")
            break
        if any(k in s for k in ("failed", "error", "canceled", "cancelled")):
            raise RuntimeError(f"❌ Simulation failed with status: {status}")

        if elapsed > MAX_WAIT_SECONDS:
            raise TimeoutError(f"Simulation timed out after {MAX_WAIT_SECONDS}s (last status={status})")

        time.sleep(poll_interval)

    # ---- Results & metrics ---------------------------------------------------
    print("--- 5) Reading results & computing metrics ---")
    res = sim.get_results_object(results_directory=str(run_dir))

    # Prepare targets
    targets_raw = json.loads(
        _resolve(cfg_dir, (cfg.get("paths", {}) or {}).get("ground_truth_json")).read_text(encoding="utf-8")
    )
    if not isinstance(targets_raw, dict):
        raise ValueError("Ground-truth JSON must be an object mapping.")
    targets_lookup = _build_targets_lookup(targets_raw, bands)

    # Define metrics & band window *before* any checks/use
    metrics_req: List[str] = (cfg.get("metrics", {}) or {}).get("objective_metrics") or []
    band_min = (cfg.get("bands", {}) or {}).get("optimise_from_hz")
    band_max = (cfg.get("bands", {}) or {}).get("optimise_to_hz")
    band_set = [f for f in bands if (band_min is None or f >= int(band_min)) and (band_max is None or f <= int(band_max))]

    # Fail-fast diagnostics
    if not metrics_req:
        raise RuntimeError("No objective metrics configured under cfg['metrics']['objective_metrics'].")
    if not targets_lookup:
        raise RuntimeError("Ground-truth JSON parsed but produced an empty lookup (no receiver metrics).")
    print(f"[debug] objective_metrics={metrics_req} | band_set={band_set} | yaml_rcvs={len(rcv_list)}")

    y_true: List[float] = []
    y_pred: List[float] = []
    y_metric: List[str] = []
    y_band: List[int] = []
    y_rcv: List[str] = []

    # Robust source label (first source)
    sim_sources = getattr(sim, "sources", None) or sim_def.source_list
    source_label = getattr(sim_sources[0], "label", None) or getattr(sim_sources[0], "name", "S")

    # Loop receivers, pull IRs, compute requested metrics
        # --- Normalise metric names on both sides ---------------------------------
    def N(x: str) -> str:
        return "".join(ch for ch in x.strip().upper() if ch.isalnum())

    metrics_req_raw: List[str] = metrics_req
    metrics_req_norm: List[str] = [N(m) for m in metrics_req_raw]

    # Loop receivers, pull IRs, compute requested metrics
    for rcv in rcv_list:
        r_label = getattr(rcv, "label", None) or getattr(rcv, "name", "R")
        ir = res.get_mono_ir(source=source_label, receiver=r_label)

        # compute -> raw map (likely uses names like "EDT_s")
        pred_map_raw = compute_metrics_from_ir(ir, band_set, metrics_req_raw)  # {metric_name: {f_hz: val}}
        # normalise keys on the prediction map
        pred_map = {N(k): v for k, v in (pred_map_raw or {}).items()}

        # targets_lookup already uses normalised keys (via _build_targets_lookup)
        tgt_map = targets_lookup.get(r_label) or targets_lookup.get(str(r_label)) or {}

        for m_norm in metrics_req_norm:
            pred_by_f = pred_map.get(m_norm, {})
            tgt_by_f  = tgt_map.get(m_norm, {})
            for f in band_set:
                f_key = str(int(f))  # targets use string keys
                if f_key not in tgt_by_f or f not in pred_by_f:
                    continue
                y_true.append(float(tgt_by_f[f_key]))
                y_pred.append(float(pred_by_f[f]))
                y_metric.append(m_norm)
                y_band.append(f)
                y_rcv.append(r_label)

    
    if len(y_true) == 0:
        # Emit helpful debug & stop instead of silently returning a huge default loss
        sim_rcv_labels = [getattr(r, "label", None) or getattr(r, "name", "") for r in rcv_list]
        diag = {
            "metrics_requested": metrics_req,
            "bands_requested_hz": band_set,
            "sim_receivers": sim_rcv_labels,
            "target_receivers": sorted(list(targets_lookup.keys())),
            "hint": "Ensure YAML receiver labels match ground_truth_json labels, and metrics/bands overlap."
        }
        (run_dir / "zero_rows_debug.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        raise RuntimeError(
            "No overlapping rows between predictions and targets. "
            f"See {run_dir.name}/zero_rows_debug.json for details."
        )

    # ---- Loss (vector Huber) -------------------------------------------------
    print("--- 6) Computing weighted Huber loss ---")
    weights_cfg = (cfg.get("metrics", {}) or {})
    scalar_loss = weighted_huber_loss(y_true, y_pred, y_metric, y_band, y_rcv, weights_cfg)

    # ---- Detailed per-row log (optional) ------------------------------------
    detailed_rows = [
        {
            "rcv_code": r, "metric": m, "f_hz": int(f),
            "target_val": float(t), "predicted_val": float(p), "error": float(p - t),
        }
        for t, p, m, f, r in zip(y_true, y_pred, y_metric, y_band, y_rcv)
    ]

    # ---- Write receipts ------------------------------------------------------
    receipt = {
        "project": project_name,
        "model": getattr(model, "name", model_name),
        "run_label": sim_def.name,  # store the actual unique sim name
        "bands_hz": bands,
        "metrics": metrics_req,
        "termination": {"energy_decay_threshold_dB": energy_db, "crossover_frequency_Hz": crossover},
        "receivers": [getattr(r, "label", "") for r in rcv_list],
        "stage1_alpha_json": str((run_dir / "stage1_alpha.json").resolve()),
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

def _main():
    ap = argparse.ArgumentParser(description="Run Stage 1 seed + Stage 2 forward sim and return scalar loss.")
    ap.add_argument("--config", required=True, help="Path to project.yaml")
    ap.add_argument("--run_label", required=False, default=f"stage1_2_{datetime.now():%Y%m%d_%H%M%S}", help="Run label/name")
    args = ap.parse_args()

    tsdk = treble.TSDK()
    loss, log = run_stage1_and_forward_sim(tsdk, Path(args.config), args.run_label)
    print("\n=== SUMMARY ===")
    print(f"Loss: {loss:.6f}")
    print(f"Rows: {len(log)}")


if __name__ == "__main__":
    _main()
