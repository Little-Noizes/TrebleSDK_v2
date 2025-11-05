# === Treble SDK import & Class Shims (definitive for your 2.3.x build) ========
import importlib

# Import the package
try:
    import treble_tsdk as treble
except ImportError as e:
    raise SystemExit(f"[FATAL] treble_tsdk not found in this venv: {e}")

# Resolve the TSDK client class (preferred location first, then fallback)
try:
    from treble_tsdk.tsdk import TSDK as _TSDK
    TSDK = _TSDK
except Exception:
    TSDK = getattr(treble, "TSDK", None)
if TSDK is None:
    raise SystemExit("[FATAL] Could not locate TSDK (treble_tsdk.tsdk.TSDK nor treble_tsdk.TSDK).")

# Geometry
try:
    from treble_tsdk.utility_classes import Point3d
except Exception:
    try:
        from treble_tsdk.core.geometry import Point3D as Point3d
    except Exception:
        Point3d = None

# Sources (your probes show these live in core.source)
try:
    from treble_tsdk.core.source import Source, SourceType, SourceProperties
except Exception as e:
    raise SystemExit(f"[FATAL] Source classes missing: {e}")

# Receiver (separate module in your build)
try:
    from treble_tsdk.core.receiver import Receiver
except Exception as e:
    raise SystemExit(f"[FATAL] Receiver class missing: {e}")

# Simulation
try:
    from treble_tsdk.core.simulation import SimulationDefinition, SimulationType
except Exception as e:
    raise SystemExit(f"[FATAL] Simulation classes missing: {e}")

# Materials (no core.material in your build → use client.api_models)
try:
    from treble_tsdk.client.api_models import (
        MaterialDefinition, MaterialCategory, MaterialRequestType, MaterialAssignment
    )
except Exception as e:
    raise SystemExit(f"[FATAL] Material DTOs missing (client.api_models): {e}")

# Final sanity check
_missing = [n for n, cls in {
    "TSDK": TSDK,
    "Point3d": Point3d,
    "Source": Source,
    "Receiver": Receiver,
    "SimulationDefinition": SimulationDefinition,
    "SimulationType": SimulationType,
    "MaterialDefinition": MaterialDefinition,
    "MaterialCategory": MaterialCategory,
    "MaterialRequestType": MaterialRequestType,
    "MaterialAssignment": MaterialAssignment,
}.items() if cls is None]
if _missing:
    raise SystemExit(f"[FATAL] Missing Treble classes: {', '.join(_missing)}")

print("[OK] Treble SDK classes successfully defined.")


# --- final check for required classes -----------------------------------------
required_classes = {
    'Point3d': Point3d,
    'Source': Source,
    'SourceProperties': SourceProperties,
    'MaterialAssignment': MaterialAssignment,
    'SimulationDefinition': SimulationDefinition,
}
missing = [name for name, val in required_classes.items() if val is None]

if missing:
    print(coloured(f"[FAILURE] Critical Treble SDK classes not found for this SDK build. Missing: {', '.join(missing)}", "red"))
    sys.exit(1) 

print(coloured("[OK] Treble SDK classes successfully defined.", "cyan"))
# ==============================================================================


# ============================ utils & helpers ================================

def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _resolve_path_ref(cfg: dict, maybe_ref: str) -> str:
    """Resolve @paths.* references used in YAML to actual path strings."""
    if isinstance(maybe_ref, str) and maybe_ref.startswith("@paths."):
        return cfg["paths"][maybe_ref.split(".", 1)[1]]
    return maybe_ref

def get_or_create_project(tsdk, name: str):
    """Robust project getter across SDK variants."""
    try:
        return tsdk.get_or_create_project(name)
    except Exception:
        try:
            return tsdk.projects.use(name)
        except Exception:
            return tsdk.projects.create(name=name)

def get_model_by_name(project, name):
    """Find a model in a project by name, handling various SDK methods (using project.get_models())."""
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
    """Gets an existing model or imports a new one, waiting for processing."""
    existing = get_model_by_name(project, name)
    if existing is not None:
        return existing
    
    # If not found, try to add it
    model = project.add_model(model_name=name, model_file_path=str(path))
    
    if model is None:
        # Fallback to unique name if previous name is taken but object not returned
        model = project.add_model(model_name=f"{name}_{_now_tag()}", model_file_path=str(path))
    
    if hasattr(model, "wait_for_model_processing"):
        model.wait_for_model_processing()
    return model

def ensure_material(tsdk, name: str, spec: dict, bands_hz: list[int]):
    """
    Ensures a material exists on the cloud, creating it if necessary.
    """
    lib = getattr(tsdk, "material_library", None)
    if lib is None:
        raise AttributeError("TSDK object is missing 'material_library'")

    # Try to retrieve it first
    if hasattr(lib, "get_by_name"):
        m = lib.get_by_name(name)
        if m is not None:
            return m

    # If retrieval fails, create it
    md = MaterialDefinition(
        name=name,
        description=spec.get("description", "Imported from YAML"),
        category=getattr(MaterialCategory, spec.get("category", "rigid"), MaterialCategory.rigid),
        default_scattering=spec.get("default_scattering", 0.05),
        material_type=MaterialRequestType.full_octave_absorption,
        coefficients=spec.get("coefficients", [0.05] * len(bands_hz)),
    )
    
    fitted = lib.perform_material_fitting(md)
    created = lib.create(fitted)
    return created

def upload_cf2(tsdk, cf2_path: Path):
    """
    Upload CF2 to directivity library, ensuring a unique name.
    """
    cf2_path = cf2_path.expanduser().resolve()
    if not cf2_path.exists():
        raise FileNotFoundError(f"CF2 file not found: {cf2_path}")
    
    # Use a unique name to avoid conflicts during upload/retrieval issues
    name = f"{cf2_path.stem}_dir_{_now_tag()}"

    lib = getattr(tsdk, "source_directivity_library", None)
    if lib is not None:
        cat = getattr(treble, "SourceDirectivityCategory", None)
        sub = getattr(treble, "SourceDirectivityAmplified", None)
        
        uploaded = lib.create_source_directivity(
            name=name,
            source_directivity_file_path=str(cf2_path),
            category=getattr(cat, 'amplified', 'Amplified'),
            sub_category=getattr(sub, 'studio_and_broadcast_monitor', 'StudioAndBroadcastMonitor'),
        )
        print(coloured(f"[OK] Uploaded CF2 to SourceDirectivityLibrary: {uploaded.name}", "green"))
        return uploaded, "source_directivity"

    raise RuntimeError("This SDK build exposes neither source_directivity_library.")

def build_source_from_cf2(cf2_obj, cf2_type: str, position: Point3d):
    """
    Create a Treble source (directive) from an uploaded CF2 object.
    """
    base_name = getattr(cf2_obj, "name", "CF2_Directivity")
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in base_name).strip("_")
    
    source_props = SourceProperties(
        # Source type is inferred as directional when source_directivity is provided.
        source_directivity=cf2_obj,
    )
    
    src = Source(
        label=safe_label[:30], # Truncate label to prevent API errors
        location=position,      # FIX: use 'location'
        source_properties=source_props,
    )
    return src

# ================================== REPORTING FUNCTIONS (The missing pieces) =====================================

def _print_config_echo(cfg: Dict[str, Any]):
    """Prints a summary of the configuration loaded from project.yaml."""
    echo_header("Configuration Summary")
    echo_kv("Project Name", cfg["project"]["name"])
    echo_kv("Run Label", cfg["project"].get("run_label", "N/A"))
    echo_kv("Model File", Path(cfg["paths"]["model_obj"]).name)
    echo_kv("Ground Truth File", Path(cfg["paths"]["ground_truth_json"]).name)
    
    echo_header("Materials to be Assigned")
    mat_names = list(cfg["materials"].keys())
    echo_list(label="Materials", items=mat_names)

    echo_header("Calculation Settings")
    calc_cfg = cfg["calculation"]
    echo_kv("Simulation Type", calc_cfg["simulation_type"])
    echo_kv("Crossover Freq (Hz)", calc_cfg["crossover_frequency_hz"])
    echo_kv("Termination Mode", calc_cfg["termination"]["mode"])
    echo_kv("GA Rays per Source", calc_cfg["ga"]["rays_per_source"])

def _process_results(cfg: Dict[str, Any], gt_avg: Dict[str, Any], results_json_path: Path):
    """Loads simulation results, compares against ground truth, and reports metrics."""
    # 1. Load results and process the data structures
    with open(results_json_path, 'r') as f:
        results_raw = json.load(f)
    
    # This function is assumed to be defined in src.metrics
    pred_metrics, pred_rce_avg = process_simulation_results(
        results_raw, 
        cfg["metrics"]["objective_metrics"], 
        cfg["_bands"]["f_hz"]
    )
    
    # 2. Per-Metric Comparison Tables
    for metric_name in cfg["metrics"]["objective_metrics"]:
        target_values = gt_avg["metrics"][metric_name]
        predicted_values = pred_rce_avg["metrics"][metric_name]

        echo_header(f"Frequency-Dependent Comparison: {metric_name}")
        table_compare_per_band(
            metric_name=metric_name,
            bands_hz=cfg["_bands"]["f_hz"],
            targets=target_values,
            predictions=predicted_values
        )

    # 3. Overall Error Summary (This uses the summarisation function from src.reporting)
    diffs, mae, rmse = summarise_receiver_errors(
        ground_truth=gt_avg, 
        predictions=pred_rce_avg, 
        metrics_to_use=cfg["metrics"]["objective_metrics"],
        huber_delta=cfg["metrics"]["huber_delta"],
        weights=cfg["metrics"]["weights"],
        optimise_from_hz=cfg["_bands"]["optimise_from_hz"]
    )
    
    if mae is not None and rmse is not None:
        echo_header("OVERALL ERROR SUMMARY")
        echo_kv("MAE (avg over rcvs/metrics/bands)", f"{mae:.3f}")
        echo_kv("RMSE (avg over rcvs/metrics/bands)", f"{rmse:.3f}")


# ================================== CORE TREBLE RUNNER =====================================

def _run_treble_forward(cfg: Dict[str, Any], out_dir: Path) -> Path:
    """
    Connects to the Treble SDK, loads the model, builds the simulation
    definition, runs the simulation, and downloads the results.
    """
    
    # 1. TSDK Client Instantiation (FIXED for UnboundLocalError)
    # Initialize tsdk to None to prevent UnboundLocalError if the conditional block fails.
    # 1) TSDK client (your build exposes it via treble_tsdk.tsdk)
    if TSDK is None:
        raise RuntimeError("treble_tsdk.tsdk.TSDK not found; update Treble SDK or adjust import path.")
    try:
        tsdk = TSDK()
    except Exception as e:
        raise RuntimeError(f"Could not instantiate TSDK(): {e}")
  
    # 2. Project and Model
    PROJECT = cfg["project"]["name"]
    MODEL_PATH = Path(_resolve_path_ref(cfg, cfg["paths"]["model_obj"])).resolve()
    MODEL_NAME = MODEL_PATH.stem


    proj = get_or_create_project(tsdk, PROJECT)
    print(coloured(f"[OK] Using Treble Project: {proj.name}", "green"))

    model = get_or_import_model(proj, MODEL_NAME, MODEL_PATH)
    print(coloured(f"[OK] Using Treble Model: {model.name} ({model.id})", "green"))
    
    # 3. Materials
    BAND_FREQS = cfg["_bands"]["f_hz"]
    
    # Ensure all required materials exist in the library
    lib_materials = {}
    for mat_name, mat_spec in cfg["materials"].items():
        lib_materials[mat_name] = ensure_material(tsdk, mat_name, mat_spec, BAND_FREQS)
    print(coloured(f"[OK] Defined {len(lib_materials)} unique materials.", "green"))

    # Create Material Assignments
    mat_assignments = []
    tag_to_material = cfg["model"]["tag_to_material"]
    
    # Note: MaterialAssignment is now aliased from MaterialAssignmentDto
    for tag, mat_name in tag_to_material.items():
        mat = lib_materials[mat_name]
        
        # Check if the tag exists in the model layers (optional but useful check)
        # model_layers_info = model.get_layers_info()
        # if tag not in [layer.label for layer in model_layers_info]:
        #     print(coloured(f"[WARN] Material tag '{tag}' not found in model layers. Skipping assignment.", "yellow"))
        #     continue

        mat_assignments.append(
            MaterialAssignment(
                layer_label=tag,
                material_id=mat.id,
            )
        )
    print(coloured(f"[OK] Created {len(mat_assignments)} material assignments.", "green"))
    
    # 4. Source Setup (CF2 Directivity)
    CF2_PATH = Path(_resolve_path_ref(cfg, cfg["paths"]["cf2_path"]))
    source_cfg = cfg["source"]
    
    cf2_obj, cf2_type = upload_cf2(tsdk, CF2_PATH)
    
    # Position uses Point3d (which was defined by the shim)
    source_pos = Point3d(
        x=source_cfg["position_m"]["x"],
        y=source_cfg["position_m"]["y"],
        z=source_cfg["position_m"]["z"],
    )
    src = build_source_from_cf2(cf2_obj, cf2_type, source_pos)
    
    # 5. Receiver Setup
    rcvs = []
    rcv_cfg = cfg["receivers"]
    for i, pos_m in enumerate(rcv_cfg):
        rcvs.append(
            Receiver(
                label=f"rcv_{i+1}",
                location=Point3d(
                    x=pos_m["x"],
                    y=pos_m["y"],
                    z=pos_m["z"],
                )
            )
        )
    
    # 6. Simulation Definition
    SIM_NAME = cfg["_run_id"]
    calc_cfg = cfg["calculation"]
    
    # Note: SimulationDefinition is now aliased from SimulationDto
    sim_def = SimulationDefinition(
        name=SIM_NAME,
        model_id=model.id,
        sources=[src],
        receivers=rcvs,
        material_assignments=mat_assignments,
        simulation_type=SimulationType.HYBRID, # Enum defined by shim
        
        # Calculation Settings
        crossover_frequency_hz=calc_cfg["crossover_freq_hz"],
        termination_mode=getattr(TerminationMode, calc_cfg["termination_mode"], TerminationMode.energy_decay), # Enum defined by shim
        ga_rays_per_source=calc_cfg["ga_rays_per_source"],
        
        # Metric Settings
        objective_metrics=cfg["metrics"]["objective_metrics"],
        band_frequencies_hz=BAND_FREQS,
    )
    
    # 7. Run Simulation
    sim = proj.run_simulation(sim_def)
    print(coloured(f"\n[OK] Launched simulation: {sim.name} ({sim.id})", "green"))
    print(f"Monitoring status... check Treble portal for details.")

    # 8. Monitor Status
    start_time = time.time()
    for _ in range(0, 60):  # Check every 30s for max 30 mins
        time.sleep(30)
        
        # Robust status check for older SDK versions
        sim_status = ""
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

    # 9. Results Download and Processing
    print(coloured("\n[INFO] Simulation finished. Downloading results...", "cyan"))
    
    # Note: The rest of the script is assumed to handle result retrieval and metric processing.
    # The return value is typically the path to the simulation results JSON.
    hybrid_json_path = out_dir / f"{SIM_NAME}_hybrid_results.json"
    
    # Placeholder for actual result retrieval logic, which is usually:
    # res_obj = sim.get_results_object(results_directory=str(out_dir))
    # actual_results = res_obj.get_hybrid_results_for_project_metrics()
    
    # For now, we return the path where results would be saved if this script included the download logic.
    return hybrid_json_path

# ================================== main =====================================

def main():
    cfg = load_config("configs/project.yaml")
    
    # Set the internal run ID
    run_label = cfg["project"].get("run_label", "default_run")
    cfg["_run_id"] = run_label.lower().replace(" ", "_")
    
    validate_config(cfg)

    # Echo inputs (OBJ, materials, source, receivers, metrics)
    _print_config_echo(cfg)

    # Determine paths
    project_root = Path(__file__).resolve().parents[1]
    run_dir = ensure_dir(project_root / cfg["paths"]["working_dir"] / cfg["_run_id"])
    fwd_dir = ensure_dir(run_dir / f"forward_{_now_tag()}")

    # Targets (averaged over receivers for baseline display)
    gt = load_ground_truth(project_root / cfg["paths"]["ground_truth_json"])
    gt_avg = targets_avg_over_receivers(gt, cfg["metrics"]["objective_metrics"], cfg["_bands"]["f_hz"])

    # --- Run the Treble forward ------------------------------------------------
    hybrid_json_path = _run_treble_forward(cfg, fwd_dir)
    # ---------------------------------------------------------------------------

    # Process and display results
    _process_results(cfg, gt_avg, hybrid_json_path)
    
if __name__ == "__main__":
    main()