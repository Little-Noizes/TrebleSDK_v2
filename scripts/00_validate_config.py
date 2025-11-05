from pathlib import Path
import sys
from src.cfg import load_config, validate_config
from src.io_utils import coloured, ensure_dir, write_json

def main():
    cfg_path = Path("configs/project.yaml")
    if not cfg_path.exists():
        print(coloured(f"[ERR] Missing config: {cfg_path}", "red"))
        sys.exit(2)
    try:
        cfg = load_config(cfg_path)
        validate_config(cfg)
    except Exception as e:
        print(coloured(f"[ERR] Config validation failed: {e}", "red"))
        sys.exit(3)
    out_dir = ensure_dir(Path("results") / cfg["_run_id"])
    write_json({
        "config_path": str(cfg_path.resolve()),
        "project": cfg["project"]["name"],
        "model_path": str(Path(cfg["paths"]["model_obj"]).resolve()),
        "gt_path": str(Path(cfg["paths"]["ground_truth_json"]).resolve()),
        "receivers": cfg["receivers"],
        "metrics": cfg["metrics"],
    }, out_dir / "run_manifest.json")
    print(coloured("[OK] Config looks good. Manifest written.", "green"))

if __name__ == "__main__":
    main()
