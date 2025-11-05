#!/usr/bin/env python
"""
===============================================================================
scripts/05_test_cf2_upload.py
===============================================================================
Purpose:
-------
Tests ONLY the Treble SDK login and CF2 directivity file upload, isolating 
the logic that was causing recent errors. This confirms the compatibility 
fix for SourceDirectivityCategory enums works.
===============================================================================
"""

from pathlib import Path
from datetime import datetime
import sys
import time
from typing import Optional, Tuple, Any

# --- Minimal Helpers (for print formatting) ---
def coloured(text: str, color: str) -> str:
    """Simple wrapper to simulate the coloured function for output."""
    colors = {'green': '\033[92m', 'cyan': '\033[96m', 'yellow': '\033[93m', 'red': '\033[91m', 'end': '\033[0m'}
    return colors.get(color, '') + str(text) + colors.get('end', '')

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────────────────────────────
# 1) IMPORT TREBLE SDK (Using the robust standard/namespace fallback)
# ─────────────────────────────────────────────────────────────────────
try:
    from treble_tsdk.tsdk import TSDK  # preferred in 2.3.x
except Exception:
    TSDK = getattr(treble, "TSDK", None)  # fallback if exposed at top-level

# ─────────────────────────────────────────────────────────────────────
# 2) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
# The path as defined in your project.yaml, relative to the v2 directory.
CF2_PATH = Path(r"input/directivity/GenelecOy-8030.cf2")

# ─────────────────────────────────────────────────────────────────────
# 3) CORE UPLOAD FUNCTION (The one that was fixed)
# ─────────────────────────────────────────────────────────────────────

def upload_cf2(tsdk, cf2_path: Path) -> Tuple[Any, str]:
    """Upload CF2 to directivity library, or fall back to device library."""
    if tsdk is None:
         raise RuntimeError("TSDK client is not initialized.")
        
    cf2_path = cf2_path.expanduser().resolve()
    if not cf2_path.exists():
        # Helpful absolute path check
        print(f"[DEBUG] Looking for CF2 at: {cf2_path}")
        raise FileNotFoundError(f"CF2 file not found: {cf2_path}")

    name = f"{cf2_path.stem} (uploaded {_now_tag()})"
    lib = getattr(tsdk, "source_directivity_library", None)
    
    if lib is not None:
        print(coloured("-> Using source_directivity_library", "cyan"))

        # FIX: Access enums from the main 'treble' namespace
        # This resolves the ImportError and ensures the required arguments are present.
        try:
            cat = treble.SourceDirectivityCategory.amplified
            sub = treble.SourceDirectivityAmplified.studio_and_broadcast_monitor

            uploaded = lib.create_source_directivity(
                name=name,
                source_directivity_file_path=str(cf2_path),
                category=cat,
                sub_category=sub,
                description="Test upload via test script",
                manufacturer="Genelec Oy",
            )
            print(coloured(f"[OK] Uploaded Directivity: {getattr(uploaded, 'name', 'N/A')}", "green"))
            print(f"Directivity ID: {getattr(uploaded, 'id', 'N/A')}")
            return uploaded, "directivity"
            
        except Exception as e:
            # Catch all exceptions, including unexpected AttributeErrors if enums are truly missing
            print(coloured(f"[ERR] Failed to upload to source_directivity_library: {e}", "red"))
            # Re-raise to stop execution if the primary upload method fails
            raise

    # Fallback: device library (for very old SDKs)
    dev_lib = getattr(tsdk, "device_library", None)
    if dev_lib is not None:
        print(coloured("-> source_directivity_library not found, using device_library", "cyan"))
        device_obj = dev_lib.import_device(str(cf2_path))
        print(coloured(f"[OK] Imported CF2 as Device: {getattr(device_obj, 'name', 'device')}", "green"))
        return device_obj, "device"

    raise RuntimeError(coloured("This SDK build exposes neither source_directivity_library nor device_library.", "red"))

# ─────────────────────────────────────────────────────────────────────
# 4) MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────

def main():
    try:
        # Initialize TSDK client
        tsdk = treble.TSDK()
                
        # Run the test
        upload_cf2(tsdk, CF2_PATH)
        
    except SystemExit:
        raise
    except Exception as e:
        print(coloured(f"[FATAL] Test failed: {e}", "red"))
        sys.exit(1)


if __name__ == "__main__":
    main()