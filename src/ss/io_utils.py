from pathlib import Path
import json, hashlib
from datetime import datetime
import yaml

# src/io_utils.py  (extend COL_MAP)
COL_MAP = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}

def coloured(text: str, colour: str) -> str:
    return f"{COL_MAP.get(colour, '')}{text}{COL_MAP['reset']}"


def read_yaml(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_json(obj: dict, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
