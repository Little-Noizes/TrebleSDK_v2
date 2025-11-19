#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Import OBJ, ensure materials, upload CF2 as directivity, run sim, export IRs.
- Uses treble_tsdk -> treble.TSDK()
- Uses source_directivity_library (with reuse if name exists)
- Falls back to device_library if needed
- Builds ALL receivers from your YAML (supports xyz_m or x/y/z), no YAML changes
- Exports WAV IRs for every (source, receiver) pair

Usage:
  (venv_treble) PS> python .\src\20_forward_once.py --config .\configs\project.yaml
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys
import time
import argparse
import json

# ─────────────────────────────────────────────────────────────────────
# 0) Third-party
# ─────────────────────────────────────────────────────────────────────
try:
    import yaml
except Exception:
    print("[FATAL] PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# 1) IMPORT TREBLE (same pattern you’ve been using)
# ─────────────────────────────────────────────────────────────────────
try:
    from treble_tsdk import treble
    print("Using treble_tsdk")
except ImportError:
    from treble_tsdk import tsdk_namespace as treble
    print("Using treble_tsdk2")

# ─────────────────────────────────────────────────────────────────────
# 2) CONSTANTS / PATHS  (adjust if you move folders)
# ─────────────────────────────────────────────────────────────────────
OBJ_PATH = Path(r"C:\Users\usuario\Documents\TrebleSDK\Models\classroom1.obj")
CF2_PATH = Path(r"C:\Users\usuario\Documents\TrebleSDK\data\001_classroom\GenelecOy-8030.cf2")

PROJECT  = "Classroom_OBJ"
MODEL    = "classroom1_obj"
SIM_NAME = f"classroom1_quickcheck_{datetime.now():%Y%m%d_%H%M%S}"  # timestamp to avoid clashes

# layer → material name (matching your SketchUp/OBJ layer names)
LAYER_TO_TREBLE = {
    "walls_plasterboard": "My_Painted_Plasterboard",
    "floor_linoleum":     "My_Linoleum_on_Slab",
    "window":             "My_Glass_10mm",
    "platform":           "My_Timber_Platform_Dense",
    "door_timber":        "My_Solid_Core_Door",
    "whiteboard":         "My_Whiteboard_Gloss",
    "ceiling_plaster":    "My_Plasterboard_Ceiling",
}

# Material definitions (same style you’ve been using)
MATERIAL_DEFS = {
    "My_Painted_Plasterboard": {
        "description": "Painted 13 mm plasterboard on studs (example)",
        "category": treble.MaterialCategory.gypsum,
        "default_scattering": 0.08,
        "material_type": treble.MaterialRequestType.full_octave_absorption,
        "coefficients": [0.12, 0.10, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06],
    },
    "My_Linoleum_on_Slab": {
        "description": "Linoleum on concrete slab (example)",
        "category": treble.MaterialCategory.rigid,
        "default_scattering": 0.05,
        "material_type": treble.MaterialRequestType.full_octave_absorption,
        "coefficients": [0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03],
    },
    "My_Glass_10mm": {
        "description": "Monolithic glass 10 mm (example)",
        "category": treble.MaterialCategory.windows,
        "default_scattering": 0.02,
        "material_type": treble.MaterialRequestType.full_octave_absorption,
        "coefficients": [0.10, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02],
    },
    "My_Timber_Platform_Dense": {
        "description": "Dense timber platform (example)",
        "category": treble.MaterialCategory.wood,
        "default_scattering": 0.10,
        "material_type": treble.MaterialRequestType.full_octave_absorption,
        "coefficients": [0.12, 0.09, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06],
    },
    "My_Solid_Core_Door": {
        "description": "Timber solid-core door (example)",
        "category": treble.MaterialCategory.wood,
        "default_scattering": 0.10,
        "material_type": treble.MaterialRequestType.full_octave_absorption,
        "coefficients": [0.15, 0.12, 0.10, 0.08, 0.07, 0.07, 0.07, 0.07],
    },
    "My_Whiteboard_Gloss": {
        "description": "Gloss laminate whiteboard (example)",
        "category": treble.MaterialCategory.other,
        "default_scattering": 0.05,
        "material_type": treble.MaterialRequestType.full_octave_absorption,
        "coefficients": [0.10, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04],
    },
    "My_Plasterboard_Ceiling": {
        "description": "Plain plasterboard ceiling (example)",
        "category": treble.MaterialCategory.gypsum,
        "default_scattering": 0.10,
        "material_type": treble.MaterialRequestType.full_octave_absorption,
        "coefficients": [0.15, 0.12, 0.10, 0.09, 0.08, 0.08, 0.08, 0.08],
    },
}

# ─────────────────────────────────────────────────────────────────────
# 3) SMALL HELPERS (model, materials, directivity, receivers)
# ─────────────────────────────────────────────────────────────────────

def get_model_by_name(project, name: str):
    for fn in ("get_model_by_name", "get_model"):
        if hasattr(project, fn):
            try:
                m = getattr(project, fn)(name) or getattr(project, fn)(model_name=name)
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
    return None

def get_or_import_model(project, name: str, path: Path):
    existing = get_model_by_name(project, name)
    if existing is not None:
        return existing
    if not path.exists():
        raise FileNotFoundError(f"OBJ not found: {path}")
    model = project.add_model(model_name=name, model_file_path=str(path))
    if model is None:
        # Name exists: reuse existing
        m2 = get_model_by_name(project, name)
        if m2 is not None:
            return m2
        # Else pick a timestamped name
        model = project.add_model(
            model_name=f"{name}_{datetime.now():%Y%m%d_%H%M%S}",
            model_file_path=str(path)
        )
    if hasattr(model, "wait_for_model_processing"):
        model.wait_for_model_processing()
    return model

def ensure_material(tsdk, name: str, spec: dict):
    lib = getattr(tsdk, "material_library", None)
    if lib is None:
        raise RuntimeError("This SDK build does not expose tsdk.material_library")

    # Reuse if exists
    if hasattr(lib, "get_by_name"):
        m = lib.get_by_name(name)
        if m is not None:
            return m

    category = spec["category"]
    if isinstance(category, str):
        category_str = category.replace(" ", "_").lower()
        category = getattr(treble.MaterialCategory, category_str, treble.MaterialCategory.other)

    md = treble.MaterialDefinition(
        name=name,
        description=spec.get("description", ""),
        category=category,
        default_scattering=spec.get("default_scattering", 0.05),
        material_type=spec.get("material_type", treble.MaterialRequestType.full_octave_absorption),
        coefficients=spec["coefficients"],
    )
    fitted = lib.perform_material_fitting(md)
    created = lib.create(fitted)
    return created

def upload_cf2(tsdk, cf2_path: Path):
    """Upload CF2 to directivity library (reuse if name exists), or fall back to device library."""
    cf2_path = cf2_path.expanduser().resolve()
    if not cf2_path.exists():
        raise FileNotFoundError(f"CF2 file not found: {cf2_path}")

    base_name = cf2_path.stem
    lib = getattr(tsdk, "source_directivity_library", None)
    if lib is not None:
        print("-> Using source_directivity_library")
        # Try creating with a deterministic name; if it exists, reuse it
        sd_obj = lib.create_source_directivity(
            name=base_name,  # stable name to allow reuse
            source_directivity_file_path=str(cf2_path),
            category=treble.SourceDirectivityCategory.amplified,
            sub_category=treble.SourceDirectivityAmplified.studio_and_broadcast_monitor,
            description="Uploaded via OBJ+CF2+sim script",
            manufacturer="Genelec Oy",
            correct_ir_by_on_axis_spl_default=True,
        )
        if sd_obj is None:
            # Name exists: reuse
            try:
                org_list = lib.get_organization_directivities()
                for d in org_list:
                    if getattr(d, "name", None) == base_name:
                        print(f"✔ Reusing existing directivity: {base_name}")
                        return d, "directivity"
            except Exception:
                pass
            raise RuntimeError("Directivity creation returned None and reuse by name failed.")
        return sd_obj, "directivity"

    # Fallback to device library
    dev_lib = getattr(tsdk, "device_library", None)
    if dev_lib is not None:
        print(" -> source_directivity_library not found, using device_library")
        device_name = base_name
        try:
            existing = dev_lib.get_device_by_name(device_name)
            if existing:
                print(f"✔ CF2 Device already exists: {existing.name}")
                return existing, "device"
        except Exception:
            try:
                for d in dev_lib.list_devices():
                    if getattr(d, "name", "") == device_name:
                        print(f"✔ CF2 Device already exists: {d.name}")
                        return d, "device"
            except Exception:
                pass

        device_obj = dev_lib.import_device(str(cf2_path))
        print(f"[OK] Imported CF2 as Device: {device_obj.name}")
        return device_obj, "device"

    raise RuntimeError("This SDK build exposes neither source_directivity_library nor device_library.")

def build_source_from_cf2(cf2_obj, cf2_type: str, position: treble.Point3d):
    """Create a Treble source (directive) from an uploaded CF2 object."""
    base_name = getattr(cf2_obj, "name", "CF2_Directivity")
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in base_name)

    if cf2_type == "device":
        # Some builds allow direct device on Source ctor
        try:
            return treble.Source(
                label=safe_label,
                position=position,
                device=cf2_obj,
                source_type=treble.SourceType.directive,
            )
        except Exception:
            pass

    # Preferred: directivity object id
    try:
        return treble.Source.make_directive(location=position, label=safe_label, source_directivity=cf2_obj)
    except Exception:
        # Fallback: omni + attach directivity id via SourceProperties
        src = treble.Source.make_omni(label=safe_label, position=position)
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

# ── Receivers from YAML (standalone; no YAML changes required) ───────────────
def extract_receivers_from_yaml(cfg: dict) -> list[dict]:
    """
    Returns list of receivers as dicts: {'label','x','y','z','type','tags'}
    Accepts either:
      - receivers: [{xyz_m:[x,y,z], label:...}]  (preferred)
      - receivers: [{x:..., y:..., z:..., label:...}]
    """
    rcvs = cfg.get("receivers")
    if rcvs is None:
        return []
    if isinstance(rcvs, dict):
        rcvs = rcvs.get("items", [])
    if not isinstance(rcvs, list):
        raise ValueError("'receivers' must be a list (or a dict containing 'items').")

    out: list[dict] = []
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

# ─────────────────────────────────────────────────────────────────────
# 4) MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    # CLI: only used to read YAML (for receivers)
    ap = argparse.ArgumentParser(description="Treble forward simulation with YAML receivers.")
    ap.add_argument("--config", required=True, help="Path to project.yaml (used to read receivers).")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    tsdk = treble.TSDK()
    proj = tsdk.get_or_create_project(PROJECT)
    print("[OK] Using project:", proj.name)

    model = get_or_import_model(proj, MODEL, OBJ_PATH)
    print("Using model:", model.name)

    # Ensure materials (Option B: create/reuse in org library)
    materials_by_name = {}
    for mat_name, spec in MATERIAL_DEFS.items():
        mobj = ensure_material(tsdk, mat_name, spec)
        materials_by_name[mat_name] = mobj
        print("[OK] Material ready:", mat_name)

    assignments = [
        treble.MaterialAssignment(layer_name=layer, material=materials_by_name[mat_name])
        for layer, mat_name in LAYER_TO_TREBLE.items()
    ]

    # Source position (as you had it)
    src_pos = treble.Point3d(4.5, 0.53, 1.5)

    # Receivers from YAML (this is the new bit)
    receivers_cfg = extract_receivers_from_yaml(cfg)
    if not receivers_cfg:
        raise RuntimeError("No receivers defined in YAML file.")
    rcv_objs = []
    for r in receivers_cfg:
        rpos = treble.Point3d(r["x"], r["y"], r["z"])
        if r["type"] == "spatial" and hasattr(treble.Receiver, "make_spatial"):
            rcv = treble.Receiver.make_spatial(position=rpos, label=r["label"])
        else:
            rcv = treble.Receiver.make_mono(position=rpos, label=r["label"])
        rcv_objs.append(rcv)
    print(f"✔ {len(rcv_objs)} receivers created from YAML: {[r.label for r in rcv_objs]}")

    # 1) Upload or reuse CF2
    cf2_obj, cf2_type = upload_cf2(tsdk, CF2_PATH)
    print(f"[OK] CF2 ready as {cf2_type}: {getattr(cf2_obj, 'name', 'unknown')}")

    # 2) Build source from CF2
    src = build_source_from_cf2(cf2_obj, cf2_type, src_pos)
    print("[OK] Source built:", src.label)

    # 3) Simulation definition
    sim_def = treble.SimulationDefinition(
        name=SIM_NAME,
        model=model,
        material_assignment=assignments,
        source_list=[src],
        receiver_list=rcv_objs,  # ← all receivers from YAML
        simulation_type=treble.SimulationType.hybrid,
        crossover_frequency=720,     # keep both, your env tolerates both
        energy_decay_threshold=35,   # per your preference
    )

    sim = proj.add_simulation(definition=sim_def)
    print("Starting simulation…")
    proj.start_simulations()
    print(f"Simulation {sim.name} started. Waiting for completion...")

    # Wait loop
    MAX_WAIT_SECONDS = 1800
    start_time = time.time()
    time.sleep(10)
    while time.time() - start_time < MAX_WAIT_SECONDS:
        time.sleep(30)
        try:
            sim_status = sim.get_status().lower()
        except AttributeError:
            sim_status = getattr(sim, "status", "").lower()

        if "complete" in sim_status or "finished" in sim_status:
            print(f"Simulation completed successfully after {int(time.time() - start_time)} seconds.")
            break
        if "failed" in sim_status or "error" in sim_status:
            raise RuntimeError(f"Simulation failed with status: {sim_status}")
        print(f"Status: {sim_status}. Waiting 30 seconds...")
    else:
        raise TimeoutError("Simulation timed out after 30 minutes.")

    # Results
    out = Path("results") / sim.name
    out.mkdir(parents=True, exist_ok=True)
    res = sim.get_results_object(results_directory=str(out))
    exported_count = 0
    # Be tolerant: different builds expose sources/receivers on sim or on sim_def
    sim_sources = getattr(sim, "sources", None) or getattr(sim_def, "source_list", [])
    sim_receivers = getattr(sim, "receivers", None) or getattr(sim_def, "receiver_list", [])

    for source in sim_sources:
        s_label = getattr(source, "label", None) or getattr(source, "name", "S")
        for receiver in sim_receivers:
            r_label = getattr(receiver, "label", None) or getattr(receiver, "name", "R")
            try:
                mono_ir = res.get_mono_ir(source=s_label, receiver=r_label)
                wav_path = out / f"{s_label}-{r_label}.wav"
                mono_ir.write_to_wav(path_to_file=str(wav_path))
                exported_count += 1
            except Exception as e:
                print(f"Error exporting IR for {s_label} → {r_label}: {e}")

    if exported_count:
        print(f"IRs exported to {exported_count} WAV file(s) in: {out.resolve()}")
    else:
        print("No WAVs exported; raw results saved to:", out.resolve())

    # Also drop a small run receipt
    receipt = {
        "project": PROJECT,
        "model": getattr(model, "name", MODEL),
        "sources": [s_label],
        "receivers": [getattr(r, "label", None) or getattr(r, "name", "R") for r in sim_receivers],
        "termination": {"energy_decay_threshold_dB": 35, "crossover_frequency_Hz": 720},
        "results_dir": str(out.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(" Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
