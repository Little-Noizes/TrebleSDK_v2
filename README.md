## TrebleSDK_v2

Clean restart of E-LAB’s Treble SDK + GA optimisation pipeline.

### Structure
- `scripts/` – sanity checks, forward run scripts, GA driver  
- `configs/` – YAML definitions  
- `input/` – CF2, OBJ, materials CSVs  
- `results/` – Treble outputs  
- `logs/` – run logs
- 'data' - wav files and json ultimate source of truth

### Goals
- Modular, reproducible runs
- Deterministic validation before GA
- CI-ready (GitHub Actions optional later)
- TrebleSDK_v2
