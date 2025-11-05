# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# 30_run_stage1_and_forward.py
# - Stage 1: generate constrained alpha(f) seed for all materials (using src.materials.compute_stage1_alpha)
# - Stage 2: apply alpha(f) to Treble, run hybrid sim, extract ONLY the metrics listed in YAML,
#            compute a single scalar loss, and return a detailed per-(rcv,metric,band) log.

# CLI:
#   (venv_treble) PS> python -m src.30_run_stage1_and_forward --config .\configs\project.yaml --run_label seed_run_001
# """

from __future__ import annotations
import argparse
import time
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

# --- Treble import (same pattern as your working scripts) --------------------
try:
    from treble_tsdk import treble
    print("Using treble_tsdk")
except ImportError:  # legacy alias in some environments
    from treble_tsdk import tsdk_namespace as treble
    print("Using treble_tsdk2")

# --- Stage 1 + metrics helpers (you already have these modules) --------------
from src.materials import compute_stage1_alpha
from src.metrics import compute_metrics_from_ir, weighted_huber_loss


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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

def _extract_receivers(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accepts receivers as [{'xyz_m':[x,y,z], 'label':..., 'type':'mono'|'spatial'}, ...].
       Also supports separate x/y/z keys. Returns a normalized list."""
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
            raise ValueError(f"Receiver at index {i} must be a mapping/object.")
        label = r.get("label") or r.get("code") or f"R{i+1}"
        pos = r.get("xyz_m")
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        else:
            try:
                x = float(r.get("x"))
                y = float(r.get("y"))
                z = float(r.get("z"))
            except (TypeError, ValueError):
                raise ValueError(f"Receiver '{label}': position must be 'xyz_m: [x,y,z]' or separate x/y/z keys.")
        rtype = (r.get("type") or "mono").strip().lower()
        if rtype not in ("mono", "spatial"):
            rtype = "mono"
        tags = r.get("tags") if isinstance(r.get("tags"), list) else []
        out.append({"label": label, "x": x, "y": y, "z": z, "type": rtype, "tags": tags})
    return out

def _get_or_import_model(project: treble.Project, name: str, obj_path: Path) -> treble.ModelObj:
    # Reuse by name if present
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
        # Duplicate name edge case; re-list and reuse
        if hasattr(project, "get_models"):
            for m in project.get_models():
                if getattr(m, "name", None) == name:
                    return m
        # Fallback to timestamped name
        model = project.add_model(
            model_name=f"{name}_{datetime.now():%Y%m%d_%H%M%S}",
            model_file_path=str(obj_path)
        )
    if hasattr(model, "wait_for_model_processing"):
        model.wait_for_model_processing()
    return model

def _upload_or_reuse_cf2(tsdk: treble.TSDK, cf2_path: Path):
    """Create-or-reuse from source_directivity_library; fallback to device_library."""
    cf2_path = cf2_path.expanduser().resolve()
    if not cf2_path.exists():
        raise FileNotFoundError(f"CF2 file not found: {cf2_path}")
    base_name = cf2_path.stem

    lib = getattr(tsdk, "source_directivity_library", None)
    if lib is not None:
        sd_obj = lib.create_source_directivity(
            name=base_name,
            source_directivity_file_path=str(cf2_path),
            category=treble.SourceDirectivityCategory.amplified,
            sub_category=treble.SourceDirectivityAmplified.studio_and_broadcast_monitor,
            description="Uploaded via Stage1+2 wrapper",
            manufacturer=None,
            correct_ir_by_on_axis_spl_default=True,
        )
        if sd_obj is None:
            # Reuse by name
            try:
                org_list = lib.get_organization_directivities()
                for d in org_list:
                    if getattr(d, "name", None) == base_name:
                        return d, "directivity"
            except Exception:
                pass
            raise RuntimeError("Directivity creation returned None and reuse by name failed.")
        return sd_obj, "directivity"

    dev_lib = getattr(tsdk, "device_library", None)
    if dev_lib is not None:
        try:
            existing = dev_lib.get_device_by_name(base_name)
            if existing:
                return existing, "device"
        except Exception:
            try:
                for d in dev_lib.list_devices():
                    if getattr(d, "name", "") == base_name:
                        return d, "device"
            except Exception:
                pass
        device_obj = dev_lib.import_device(str(cf2_path))
        return device_obj, "device"

    raise RuntimeError("This SDK build exposes neither source_directivity_library nor device_library.")

def _build_source_from_cf2(cf2_obj, cf2_type: str, pos: treble.Point3d) -> treble.Source:
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in getattr(cf2_obj, "name", "CF2_Source"))
    if cf2_type == "device":
        try:
            return treble.Source(
                label=safe_label,
                position=pos,
                device=cf2_obj,
                source_type=treble.SourceType.directive,
            )
        except Exception:
            pass
    # Preferred directive object
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
) -> None:
    """
    Applies alpha(f) per material to the Treble org library.
    Expected stage1_alpha format:
      material_key -> {"bands_hz": [..], "alpha": [..]}
    Tries 'update' path; if unsupported, creates or re-fits via perform_material_fitting+create.
    """
    lib = getattr(tsdk, "material_library", None)
    if lib is None:
        raise RuntimeError("TSDK.material_library is not available in this SDK build.")

    for mat_name, spec in stage1_alpha.items():
        bands = spec.get("bands_hz")
        alpha = spec.get("alpha")
        if not bands or not alpha or len(bands) != len(alpha):
            raise ValueError(f"Invalid alpha spec for material '{mat_name}': bands/alpha mismatch")

        # Try an 'update' API if available in your build
        for candidate in ("update", "update_absorption", "set_absorption"):
            if hasattr(lib, candidate):
                try:
                    getattr(lib, candidate)(
                        material_id=mat_name,  # assuming names as IDs; swap to id lookup if needed
                        absorption_coefficients=alpha,
                        band_frequencies=bands
                    )
                    break
                except Exception:
                    # fall through to fitting/creation
                    pass
        else:
            # No direct update; try reuse-by-name or create via fit
            mobj = None
            if hasattr(lib, "get_by_name"):
                try:
                    mobj = lib.get_by_name(mat_name)
                except Exception:
                    mobj = None

            md = treble.MaterialDefinition(
                name=mat_name,
                description=f"alpha(f) updated by Stage1 seed @ {datetime.now():%Y%m%d_%H%M%S}",
                category=getattr(treble.MaterialCategory, "other"),
                default_scattering=0.1,
                material_type=treble.MaterialRequestType.full_octave_absorption,
                coefficients=alpha,
            )
            fitted = lib.perform_material_fitting(md)
            if mobj is None and not allow_create:
                raise RuntimeError(f"Material '{mat_name}' not found and creation is disabled.")
            lib.create(fitted)  # new/updated entry depending on build

def _build_receivers(cfg: Dict[str, Any]) -> List[treble.Receiver]:
    out: List[treble.Receiver] = []
    for r in _extract_receivers(cfg):
        p = treble.Point3d(r["x"], r["y"], r["z"])
        if r["type"] == "spatial" and hasattr(treble.Receiver, "make_spatial"):
            out.append(treble.Receiver.make_spatial(position=p, label=r["label"]))
        else:
            out.append(treble.Receiver.make_mono(position=p, label=r["label"]))
    if not out:
        raise RuntimeError("No receivers defined in YAML.")
    return out

def _build_source(cfg: Dict[str, Any], tsdk: treble.TSDK) -> treble.Source:
    src_cfg = cfg.get("source", {}) or {}
    pos = src_cfg.get("position_m") or src_cfg.get("xyz_m")
    if not pos or len(pos) != 3:
        raise ValueError("source.position_m (or xyz_m) must be [x, y, z] in metres.")
    pos3 = treble.Point3d(float(pos[0]), float(pos[1]), float(pos[2]))

    cfg_dir = Path(cfg.get("__cfg_dir__"))  # injected in main loader
    cf2_path = _resolve(cfg_dir, (cfg.get("paths", {}) or {}).get("directivity_cf2"))
    if not cf2_path:
        raise RuntimeError("paths.directivity_cf2 is required in YAML.")
    cf2_obj, cf2_type = _upload_or_reuse_cf2(tsdk, cf2_path)
    return _build_source_from_cf2(cf2_obj, cf2_type, pos3)

def _collect_metric_targets(cfg: Dict[str, Any], cfg_dir: Path) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[int], List[str], Dict[str, Any]]:
    """
    Returns:
      targets: targets[rcv_label][metric][str(f_hz)] = value
      bands:   [ ... ints ... ]
      metrics: [ 'EDT', 'T20', 'C50', ... ]  (exactly as in YAML)
      weights_cfg: dict from cfg['metrics'] (for loss)
    """
    paths = cfg.get("paths", {}) or {}
    gt_path = _resolve(cfg_dir, paths.get("ground_truth_json"))
    if not gt_path or not gt_path.exists():
        raise FileNotFoundError(f"Ground truth JSON not found: {gt_path}")

    targets = json.loads(gt_path.read_text(encoding="utf-8"))
    if not isinstance(targets, dict):
        raise ValueError("Ground truth JSON must be an object mapping.")

    #bands = cfg.get("bands", {}).get("list") or cfg.get("bands", {}).get("hz") or []
    bands = (
    cfg.get("bands", {}).get("list") 
    or cfg.get("bands", {}).get("hz") 
    or cfg.get("bands", {}).get("f_hz") # <--- ADDED LINE
    or []
    )
    if not bands:
        raise ValueError("Bands list is required in cfg['bands']['list'] (or ['hz']).")
    bands = [int(round(float(b))) for b in bands]

    metrics = cfg.get("metrics", {}).get("objective_metrics") or []
    if not metrics:
        raise ValueError("cfg['metrics']['objective_metrics'] must list the metrics to use (e.g., ['EDT','T20','C50']).")

    weights_cfg = cfg.get("metrics", {}) or {}
    return targets, bands, metrics, weights_cfg

def _compute_loss_and_log(
    *,
    cfg: Dict[str, Any],
    bands: List[int],
    metrics: List[str],
    targets: Dict[str, Dict[str, Dict[str, float]]],
    results_reader: Any,   # expects res.get_mono_ir(...)
    source_label: str,
    receiver_list: List[treble.Receiver],
    weights_cfg: Dict[str, Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Pull mono IRs, compute only the requested metrics, build detailed rows, then compute weighted scalar loss.
    """
    rows: List[Dict[str, Any]] = []

    y_true: List[float] = []
    y_pred: List[float] = []
    y_metric: List[str] = []
    y_band: List[int] = []
    y_rcv: List[str] = []

    band_min = cfg.get("bands", {}).get("optimise_from_hz")
    band_max = cfg.get("bands", {}).get("optimise_to_hz")
    band_set = [f for f in bands if (band_min is None or f >= int(band_min)) and (band_max is None or f <= int(band_max))]

    for rcv in receiver_list:
        rcv_label = getattr(rcv, "label", None) or getattr(rcv, "name", "R")
        ir = results_reader.get_mono_ir(source=source_label, receiver=rcv_label)
        metric_map = compute_metrics_from_ir(ir, band_set, metrics)  # dict: metric -> {f_hz -> val}

        for m in metrics:
            pred_by_f = metric_map.get(m, {})
            tgt_by_f = (targets.get(rcv_label, {}).get(m, {})) if isinstance(targets.get(rcv_label, {}), dict) else {}
            for f in band_set:
                f_key = str(int(f))
                if f_key not in tgt_by_f or f not in pred_by_f:
                    continue
                tval = float(tgt_by_f[f_key])
                pval = float(pred_by_f[f])
                err = pval - tval
                rows.append({
                    "rcv_code": rcv_label,
                    "metric": m,
                    "f_hz": f,
                    "target_val": tval,
                    "predicted_val": pval,
                    "error": err,
                })
                y_true.append(tval)
                y_pred.append(pval)
                y_metric.append(m)
                y_band.append(f)
                y_rcv.append(rcv_label)

    loss = weighted_huber_loss(y_true, y_pred, y_metric, y_band, y_rcv, weights_cfg)
    return float(loss), rows


# -----------------------------------------------------------------------------
# Primary API
# -----------------------------------------------------------------------------

def run_stage1_and_forward_sim(
    tsdk: Any,
    config_path: Path,
    run_label: str
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Runs the Stage 1 statistical fit to generate the alpha seed, updates Treble
    materials, runs the Stage 2 forward simulation, and returns the objective loss.

    Returns:
        (scalar_objective_loss, detailed_results_log)
    """
    cfg_path = Path(config_path).expanduser().resolve()
    cfg = _cfg_load(cfg_path)
    cfg_dir = cfg_path.parent
    cfg["__cfg_dir__"] = str(cfg_dir)

    # Step 1: config & targets
    targets, bands, metrics, weights_cfg = _collect_metric_targets(cfg, cfg_dir)

    # Step 2: Stage 1 seed
    stage1_alpha: Dict[str, Dict[str, Any]] = compute_stage1_alpha(cfg, bands)

    # Persist Stage 1 output
    workdir = _resolve(cfg_dir, (cfg.get("paths", {}) or {}).get("working_dir")) or (cfg_dir / "results")
    workdir.mkdir(parents=True, exist_ok=True)
    run_dir = workdir / f"{run_label}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage1_alpha.json").write_text(json.dumps(stage1_alpha, indent=2), encoding="utf-8")

    # Step 3: apply materials
    _apply_alpha_to_materials(tsdk, stage1_alpha, allow_create=True)

    # Step 4: build sim
    project_name = (cfg.get("project", {}) or {}).get("name") or "Treble_Project"
    proj = tsdk.get_or_create_project(name=project_name)

    obj_path = _resolve(cfg_dir, (cfg.get("paths", {}) or {}).get("model_obj"))
    if not obj_path:
        raise RuntimeError("paths.model_obj is required in YAML.")
    model_name = (cfg.get("model", {}) or {}).get("name") or Path(obj_path).stem
    model = _get_or_import_model(proj, model_name, obj_path)

    src = _build_source(cfg, tsdk)
    rcv_list = _build_receivers(cfg)

    tag_map = (cfg.get("tag_to_material") or {})
    assignments: List[treble.MaterialAssignment] = []
    for layer_name, mat_name in tag_map.items():
        if isinstance(mat_name, dict):
            mat_name = mat_name.get("name")
        #assignments.append(treble.MaterialAssignment(layer_name=str(layer_name), material=mat_name))
        # ------------------ REPLACEMENT ------------------
        # Get the Treble Material object by its name (required for MaterialAssignment)
        # We use get_by_name, which works even if the material was just created or already existed.
        try:
            material_obj = tsdk.material_library.get_by_name(mat_name)
        except Exception as e:
            raise RuntimeError(f"Failed to find or retrieve material '{mat_name}' in the library.") from e

        # Pass the Material object (which has the .id attribute) to MaterialAssignment
        assignments.append(treble.MaterialAssignment(layer_name=str(layer_name), material=material_obj))
# -------------------------------------------------
    calc = (cfg.get("calculation") or {})
    term = (calc.get("termination") or {})
    energy_db = float(term.get("energy_decay_threshold_db", 35.0))
    crossover = calc.get("crossover_frequency_hz", 720)

    sim_def = treble.SimulationDefinition(
        name=run_label,
        simulation_type=treble.SimulationType.hybrid,
        model=model,
        receiver_list=rcv_list,
        source_list=[src],
        material_assignment=assignments,
        energy_decay_threshold=energy_db,
        crossover_frequency=crossover,
        simulation_settings={"ambisonicsOrder": 2, "speedOfSound": 343.0},
        tags=["stage1_seed", "forward_eval"],
    )

    # Step 5: run sim
    sim = proj.add_simulation(definition=sim_def)
    proj.start_simulations()

    MAX_WAIT_SECONDS = int(calc.get("timeout_seconds", 1800))
    start_time = time.time()
    time.sleep(10)
    last = None
    while time.time() - start_time < MAX_WAIT_SECONDS:
        time.sleep(30)
        try:
            status = sim.get_status()
        except AttributeError:
            status = getattr(sim, "status", "")
        if status != last:
            print(f"[sim] status: {status}")
            last = status
        s = str(status).lower()
        if "complete" in s or "finished" in s:
            break
        if "failed" in s or "error" in s or "canceled" in s:
            raise RuntimeError(f"Simulation failed with status: {status}")
    else:
        raise TimeoutError(f"Simulation timed out after {MAX_WAIT_SECONDS} s.")

    # Step 6–7: results → metrics → loss/log
    res = sim.get_results_object(results_directory=str(run_dir))
    sim_sources = getattr(sim, "sources", None) or sim_def.source_list
    source_label = getattr(sim_sources[0], "label", None) or getattr(sim_sources[0], "name", "S")

    scalar_loss, detailed_log = _compute_loss_and_log(
        cfg=cfg,
        bands=bands,
        metrics=metrics,
        targets=targets,
        results_reader=res,
        source_label=source_label,
        receiver_list=rcv_list,
        weights_cfg=weights_cfg,
    )

    # Receipt
    receipt = {
        "project": project_name,
        "model": getattr(model, "name", model_name),
        "run_label": run_label,
        "bands_hz": bands,
        "metrics": metrics,
        "termination": {"energy_decay_threshold_dB": energy_db, "crossover_frequency_Hz": crossover},
        "receivers": [getattr(r, "label", "") for r in rcv_list],
        "stage1_alpha_json": str((run_dir / "stage1_alpha.json").resolve()),
        "results_dir": str(run_dir.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "loss": float(scalar_loss),
    }
    (run_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    return float(scalar_loss), detailed_log


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------

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
