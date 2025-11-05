#!/usr/bin/env python
"""
===============================================================================
scripts/06_debug_model_layers.py
===============================================================================
Purpose:
-------
Loads the OBJ model and prints the definitive list of layer names
extracted by the Treble SDK to solve the Material Assignment ValueError.
===============================================================================
"""

from pathlib import Path
import sys
import yaml
from typing import Any, Dict, List

# --- Minimal Helpers ---
def coloured(text: str, color: str) -> str:
    """Simple wrapper to simulate the coloured function for output."""
    colors = {'green': '\033[92m', 'cyan': '\033[96m', 'yellow': '\033[93m', 'red': '\033[91m', 'end': '\033[0m'}
    return colors.get(color, '') + str(text) + colors.get('end', '')

def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """Basic YAML loader to get config structure for comparison."""
    # CHANGE THIS LINE: Add encoding='utf-8'
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ─────────────────────────────────────────────────────────────────────
# 1) IMPORT TREBLE SDK
# ─────────────────────────────────────────────────────────────────────
try:
    # Use the import pattern successful in your other scripts
    from treble_tsdk import treble
    print(coloured("Using treble_tsdk", "green"))
except ImportError as e:
    raise SystemExit(coloured("TrebleSDK is not installed/active in this venv.", "red")) from e

# ─────────────────────────────────────────────────────────────────────
# 2) PATHS AND CONSTANTS
# ─────────────────────────────────────────────────────────────────────
# # Assumed paths based on your project structure:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = PROJECT_ROOT / "configs/project.yaml"
# # We read the model path from the config, but assume a default name
# MODEL_NAME = "classroom1" 


def main():
    # --- (Inside the main() function of src/06_debug_model_layers.py) ---

    try:
        # CRITICAL FIX 1: Correct Model Name
        MODEL_NAME = "classroom1_obj" 
        
        # Load config...
        cfg = load_yaml_config(CFG_PATH)
        yaml_assignments: List[str] = list(cfg["tag_to_material"].keys())
        PROJECT_NAME = cfg["project"]["name"]
        
        # 1. Initialize TSDK client
        tsdk = treble.TSDK()
        
        # 1.5: Get or Create Project (Works)
        print(coloured(f"\nAttempting to get or create project: {PROJECT_NAME}", "cyan"))
        project = tsdk.get_or_create_project(name=PROJECT_NAME)

        # 2. Get the existing Model
        print(coloured(f"Attempting to retrieve existing model '{MODEL_NAME}' from project '{PROJECT_NAME}'...", "cyan"))
        
        # ⚠️ CRITICAL FIX 2: Use project.get_models() and filter by name.
        # This is the most reliable way to get an existing item from a known project object.
        all_models_in_project = project.get_models() 
        model_list = [m for m in all_models_in_project if m.name == MODEL_NAME]
        
        model = model_list[0] if model_list else None
        
        if model is None:
            # Model is truly missing, proceed with import/add
            model_obj_path = PROJECT_ROOT / cfg["paths"]["model_obj"]
            print(coloured(f"[WARNING] Model not found in project. Attempting import via project.add_model...", "yellow"))
            
            # This is expected to fail with the "already exists" error on an existing model, 
            # but if the model was manually deleted, this would succeed.
            model = project.add_model(
                model_name=MODEL_NAME, 
                model_file_path=str(model_obj_path)
            )
            
            if model is None:
                # If the script reaches this point, the only solution is manual intervention.
                print(coloured("AUTOMATIC RETRIEVAL FAILED. THE MODEL IS ON THE CLOUD BUT INACCESSIBLE TO SDK.", "red"))
                raise RuntimeError("Model import failed and returned None. Please manually delete 'classroom1_obj' from your Treble project dashboard and run again.")
                
            print("Waiting for model processing...")
            model.wait_for_model_processing()
        else:
            print(coloured(f"[OK] Model '{MODEL_NAME}' retrieved successfully.", "green"))

        # 3. DEBUG OUTPUT
        model_layers = model.layer_names
        
        # ... (rest of the DEBUG OUTPUT code) ...
        
        print(coloured("\n===================================================================", "yellow"))
        print(coloured("== DEBUG: MODEL LAYER NAMES (Required Keys for Material Assignment) ==", "yellow"))
        print(coloured("===================================================================", "yellow"))
        
        print(f"Total layers found by Treble in model '{model.name}': {len(model_layers)}")
        print(f"List of required layer names (must match keys in project.yaml exactly):")
        
        # Print the required list clearly
        for name in model_layers:
            print(coloured(f"  - {name}", "green"))
            
        print(coloured("\n===================================================================", "yellow"))
        print(coloured("== DEBUG: YOUR ASSIGNMENT KEYS (Keys currently in project.yaml) ==", "yellow"))
        print(coloured("===================================================================", "yellow"))
        
        print(f"Total keys found in project.yaml: {len(yaml_assignments)}")
        print(f"List of keys currently being used (Mismatched if any keys above are missing):")
        
        # Print the user's assignment keys for comparison
        for name in yaml_assignments:
            # Highlight keys that are missing from the model for better debugging
            color = "red" if name not in model_layers else "cyan"
            print(coloured(f"  - {name}", color))
            
        # Final comparison summary
        missing_keys = set(model_layers) - set(yaml_assignments)
        
        if missing_keys:
            print(coloured(f"\n[FAILURE] The following required model layers are missing from your project.yaml tag_to_material block:", "red"))
            for key in missing_keys:
                print(coloured(f"  - {key}", "red"))
            print(coloured("\nACTION: Copy the 'List of required layer names' (in green) and replace the keys in your project.yaml's 'tag_to_material' section.", "red"))
        else:
            print(coloured("\n[SUCCESS] Your project.yaml layer names match the model layers.", "green"))


    except Exception as e:
        print(coloured(f"\n[FATAL] Debug script failed: {e}", "red"))
        sys.exit(1)


if __name__ == "__main__":
    main()