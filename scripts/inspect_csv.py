#!/usr/bin/env python
"""Quick script to inspect joined.csv structure"""

import pandas as pd
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_csv.py <run_dir>")
    sys.exit(1)

run_dir = Path(sys.argv[1])
csv_path = run_dir / "joined.csv"

if not csv_path.exists():
    print(f"Error: {csv_path} not found")
    sys.exit(1)

df = pd.read_csv(csv_path)

print(f"\n=== CSV Info ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nData types:")
print(df.dtypes)

if "metric" in df.columns:
    print(f"\nUnique metrics: {sorted(df['metric'].unique())}")
if "band_hz" in df.columns:
    print(f"Unique bands: {sorted(df['band_hz'].unique())}")