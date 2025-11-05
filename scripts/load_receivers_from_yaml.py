#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/inspect_receivers.py
----------------------------
Read a Treble project YAML and print the receivers it defines (no YAML changes).
- Accepts xyz_m (preferred) or x/y/z fallback
- Uses 'label' or 'code' for naming
- Defaults type to 'mono' if unspecified
- Prints a tidy table and also dumps JSON for copy/paste

Usage:
  python scripts/inspect_receivers.py --config ./configs/project.yaml
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except Exception as e:
    print("[FATAL] PyYAML not installed. pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return data


def _coerce_receiver(rec: Dict[str, Any], idx: int) -> Dict[str, Any]:
    # Label/Code
    label = rec.get("label") or rec.get("code") or f"R{idx+1}"
    # Position: prefer xyz_m, else x/y/z
    pos = rec.get("xyz_m")
    if isinstance(pos, (list, tuple)) and len(pos) == 3:
        x, y, z = [float(pos[0]), float(pos[1]), float(pos[2])]
    else:
        try:
            x = float(rec.get("x"))
            y = float(rec.get("y"))
            z = float(rec.get("z"))
        except (TypeError, ValueError):
            raise ValueError(f"Receiver '{label}': position must be 'xyz_m: [x,y,z]' or separate x/y/z keys.")
    # Type
    rtype = (rec.get("type") or "mono").strip().lower()
    if rtype not in ("mono", "spatial"):
        rtype = "mono"  # default conservatively
    # Optional extras
    tags = rec.get("tags") if isinstance(rec.get("tags"), list) else []
    return {"label": label, "x": x, "y": y, "z": z, "type": rtype, "tags": tags}


def extract_receivers(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rcvs = cfg.get("receivers")
    if rcvs is None:
        return []
    if isinstance(rcvs, dict):
        # allow a dict form like {"items":[...]}
        rcvs = rcvs.get("items", [])
    if not isinstance(rcvs, list):
        raise ValueError("'receivers' must be a list (or a dict containing 'items').")
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rcvs):
        if not isinstance(r, dict):
            raise ValueError(f"Receiver at index {i} must be a mapping/object.")
        out.append(_coerce_receiver(r, i))
    return out


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No receivers found.")
        return
    # Compute widths
    cols = ["#", "label", "type", "x", "y", "z", "tags"]
    data = []
    for i, r in enumerate(rows, 1):
        data.append([
            str(i),
            r["label"],
            r["type"],
            f"{r['x']:.3f}",
            f"{r['y']:.3f}",
            f"{r['z']:.3f}",
            ", ".join(r["tags"]) if r["tags"] else ""
        ])
    widths = [max(len(c), max(len(row[j]) for row in data)) for j, c in enumerate(cols)]
    # Header
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    print(header)
    print(sep)
    # Rows
    for row in data:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(cols))))


def main():
    ap = argparse.ArgumentParser(description="Inspect receivers from a Treble project YAML (no YAML changes).")
    ap.add_argument("--config", required=True, help="Path to project.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    receivers = extract_receivers(cfg)

    print(f"Found {len(receivers)} receiver(s) in {cfg_path}:\n")
    print_table(receivers)

    # Also print JSON for programmatic reuse
    print("\nJSON:")
    print(json.dumps(receivers, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
