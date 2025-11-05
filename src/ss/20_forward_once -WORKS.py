#!/usr/bin/env python
"""
Import OBJ, ensure materials, upload CF2 as directivity, run sim, export IRs.
Single, unified script based on your two working ones.

- uses treble_tsdk -> treble.TSDK()
- uses source_directivity_library first
- falls back to device_library if needed
- creates a Treble-valid source label (alnum + underscore)
"""

from pathlib import Path
from datetime import datetime
import sys
import time

# ─────────────────────────────────────────────────────────────────────
# 1) IMPORT TREBLE (same pattern as your working script)
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
SIM_NAME = f"classroom1_quickcheck_{datetime.now():%Y%m%d_%H%M%S}"  # we add a timestamp to avoid name clashes

# layer → material name; this has to be thea same as in your sketchup model exported to OBJ
# remember to swith the chekcbox to ensure z is upwards when exporting from SketchUp --> OPTIONS
LAYER_TO_TREBLE = {
    "walls_plasterboard": "My_Painted_Plasterboard",
    "floor_linoleum":     "My_Linoleum_on_Slab",
    "window":             "My_Glass_10mm",
    "platform":           "My_Timber_Platform_Dense",
    "door_timber":        "My_Solid_Core_Door",
    "whiteboard":         "My_Whiteboard_Gloss",
    "ceiling_plaster":    "My_Plasterboard_Ceiling",
}

# material definitions (same as working script)
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
# 3) HELPERS FROM YOUR WORKING SCRIPT (unchanged)
# ─────────────────────────────────────────────────────────────────────

def list_names(m):
    for g in ("list_layer_names", "get_layer_names", "get_mesh_material_names", "list_mesh_materials"):
        if hasattr(m, g):
            try:
                names = sorted(getattr(m, g)())
                return names
            except Exception:
                pass
    return []

def get_model_by_name(project, name):
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

def get_or_import_model(project, name, path: Path):
    existing = get_model_by_name(project, name)
    if existing is not None:
        return existing
    model = project.add_model(model_name=name, model_file_path=str(path))
    if model is None:
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

    if hasattr(lib, "get_by_name"):
        m = lib.get_by_name(name)
        if m is not None:
            return m

    # convert string-ish category to enum
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

# ─────────────────────────────────────────────────────────────────────
# 4) CF2 UPLOAD (your newer script) — folded in
# ─────────────────────────────────────────────────────────────────────

def upload_cf2(tsdk, cf2_path: Path):
    """Upload CF2 to directivity library, or fall back to device library."""
    cf2_path = cf2_path.expanduser().resolve()
    if not cf2_path.exists():
        raise FileNotFoundError(f"CF2 file not found: {cf2_path}")

    name = f"{cf2_path.stem} (uploaded {datetime.now():%Y%m%d_%H%M%S})"

    # 1) try directivity library
    lib = getattr(tsdk, "source_directivity_library", None)
    if lib is not None:
        print("-> Using source_directivity_library")
        uploaded = lib.create_source_directivity(
            name=name,
            source_directivity_file_path=str(cf2_path),
            category=treble.SourceDirectivityCategory.amplified,
            sub_category=treble.SourceDirectivityAmplified.studio_and_broadcast_monitor,
            description="Uploaded via OBJ+CF2+sim script",
            manufacturer="Genelec Oy",
        )
        return uploaded, "directivity"

    # 2) fallback: device library (older style)
    dev_lib = getattr(tsdk, "device_library", None)
    if dev_lib is not None:
        print(" -> source_directivity_library not found, using device_library")
        device_name = cf2_path.stem
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
        print(f"[Ok] Imported CF2 as Device: {device_obj.name}")
        return device_obj, "device"

    raise RuntimeError("This SDK build exposes neither source_directivity_library nor device_library.")

def build_source_from_cf2(cf2_obj, cf2_type: str, position: treble.Point3d):
    """Create a Treble source (directive) from an uploaded CF2 object."""
    base_name = getattr(cf2_obj, "name", "CF2_Directivity")
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in base_name)

    if cf2_type == "device":
        src = treble.Source(
            label=safe_label,
            position=position,
            device=cf2_obj,
            source_type=treble.SourceType.directive,
        )
    else:
        # directivity in library
        src = treble.Source.make_omni(
            label=safe_label,
            position=position,
        )
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

# ─────────────────────────────────────────────────────────────────────
# 5) MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    tsdk = treble.TSDK()
    proj = tsdk.get_or_create_project(PROJECT)
    print("[Ok] Using project:", proj.name)

    model = get_or_import_model(proj, MODEL, OBJ_PATH)
    print("Using model:", model.name)

    # materials
    materials_by_name = {}
    for mat_name, spec in MATERIAL_DEFS.items():
        mobj = ensure_material(tsdk, mat_name, spec)
        materials_by_name[mat_name] = mobj
        print("[Ok] Material ready:", mat_name)

    assignments = [
        treble.MaterialAssignment(layer_name=layer, material=materials_by_name[mat_name])
        for layer, mat_name in LAYER_TO_TREBLE.items()
    ]

    # positions
    src_pos = treble.Point3d(4.5, 0.53, 1.5)
    rcv_pos = treble.Point3d(7.5, 3.4, 1.5)

    # 1) upload or get CF2
    cf2_obj, cf2_type = upload_cf2(tsdk, CF2_PATH)
    print(f"[Ok] CF2 uploaded as {cf2_type}: {getattr(cf2_obj, 'name', 'unknown')}")

    # 2) build source from that CF2
    src = build_source_from_cf2(cf2_obj, cf2_type, src_pos)
    print("[Ok] Source built from CF2:", src.label)

    # receiver
    rcv = treble.Receiver.make_mono(
        position=rcv_pos,
        label="Listener",
    )

    # sim de
    sim_def = treble.SimulationDefinition(
        name=SIM_NAME,
        model=model,
        material_assignment=assignments,
        source_list=[src],
        receiver_list=[rcv],
        simulation_type=treble.SimulationType.hybrid,
        crossover_frequency=720,
        energy_decay_threshold=35,
    )

    sim = proj.add_simulation(definition=sim_def)
    print("Starting simulation…")
    proj.start_simulations()
    print(f"Simulation {sim.name} started. Waiting for completion...")

    # wait loop (same as yours)
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

    # results
    out = Path("results") / sim.name
    out.mkdir(parents=True, exist_ok=True)
    res = sim.get_results_object(results_directory=str(out))
    exported_count = 0
    for source in sim.sources:
        for receiver in sim.receivers:
            try:
                mono_ir = res.get_mono_ir(source=source.label, receiver=receiver.label)
                wav_path = out / f"{source.label}-{receiver.label}.wav"
                mono_ir.write_to_wav(path_to_file=str(wav_path))
                exported_count += 1
            except Exception as e:
                print(f"Error exporting IR for {source.label} → {receiver.label}: {e}")

    if exported_count:
        print(f"IRs exported to {exported_count} WAV file(s) in: {out.resolve()}")
    else:
        print("No WAVs exported; raw results saved to:", out.resolve())

    print(" Done.")

if __name__ == "__main__":
    main()
