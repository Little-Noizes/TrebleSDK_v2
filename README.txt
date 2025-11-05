Process overview (four phases)
Phase 0 — Configuration + mesh facts

Goal: lock inputs, surface areas, receiver set, metrics to optimise, and frequency bands before any optimisation.

Script: 00_validate_config.py

Reads configs/project.yaml, validates schema & required paths.

Echoes chosen metrics (e.g., EDT, T20, C50) and bands (63–8000 Hz thirds or octaves).

Validates receivers list (IDs, coordinates, labels) and source(s).

Confirms input/results_main.json exists and matches requested metrics/bands.

Key functions

# src/cfg.py
def load_config(path: str) -> dict
def validate_config(cfg: dict) -> None  # raises ValueError with clear messages

# src/io_utils.py
def read_yaml(path) -> dict
def read_json(path) -> dict
def write_json(obj, path) -> None

Script: 01_mesh_scan.py

Parses OBJ (or a JSON export) and returns:

tags_by_face, material_by_tag, area_m2_by_tag, area_m2_total

Emits results/<run_id>/mesh_summary.csv with tag, count, m², mapped material.

Cross-checks that every tag has a material in config.

Key functions

# src/mesh_utils.py
def scan_obj_for_tags_and_areas(obj_path: str) -> dict[str, float]  # tag->area_m2
def verify_tag_material_map(tag_area: dict, map_cfg: dict) -> list[str]  # missing tags


Config snippet (schema)

project:
  name: Classroom_OBJ
model:
  path: input/model.obj
  tag_to_material:
    Plaster_Ceiling: My_Plasterboard_Ceiling
    Plaster_Walls:   My_Painted_Plasterboard
    Glass_10mm:      My_Glass_10mm
treble:
  directivity_cf2: input/directivity.cf2
  simulation_type: hybrid
  crossover_hz: 720
receivers:
  - id: r1
    xyz: [7.50, 3.40, 1.50]
  - id: r2
    xyz: [5.00, 2.00, 1.50]
metrics:
  # what to match and how to weight them
  bands: third     # or 'octave'
  fmin: 63
  fmax: 8000
  list:
    - name: EDT
      weight: 1.0
    - name: T20
      weight: 0.8
    - name: C50
      weight: 0.5
ground_truth:
  path: input/results_main.json   # measured/validated reference
stage1:
  # statistical calibration settings
  bounds:
    alpha_min: 0.02
    alpha_max: 0.95
  monotonic: true   # enforce ↑ with frequency
  smoothness: 0.10  # max delta between adjacent bands
stage2:
  # GA loop settings
  population: 40
  generations: 50
  mutation_rate: 0.2
  crossover_rate: 0.8
  elite: 2
  restart_stagnation_gens: 8

Phase 1 — Stage-1 statistical calibration (no Treble)

Goal: derive per-material absorption α(f) that best matches the average of metrics across receivers using the existing results_main.json. This produces a good starting point.

Script: 10_stage1_stat_fit.py

Loads config and results_main.json.

Aggregates target metrics as receiver average per metric & band.

Builds a linear/statistical room model (e.g., Sabine/Eyring per band) to map α(f) → predicted metric(s).

Solves a bounded, regularised optimisation for α(f) per material:

Constraints: alpha_min ≤ α[i] ≤ alpha_max, monotonic increasing if requested, smoothness between bands.

Loss: weighted Huber or MAE across selected metrics, averaged over bands.

Key functions

# src/metrics.py
def aggregate_targets(gt: dict, cfg) -> dict  # avg across receivers per metric & band
def norm_error(pred, target, norm="huber", delta=0.5) -> float
def metric_names(cfg) -> list[str]

# src/materials.py
def initial_alpha_guess(material_name: str, bands: list[float]) -> list[float]
def enforce_monotonic(arr: list[float]) -> list[float]
def clamp(arr, mn, mx): ...

# src/fit_stage1.py
def build_stat_model(inputs) -> "StatModel"  # callable mapping α -> metrics
def fit_absorption(stat_model, targets, bounds, constraints, weights) -> dict[str, list[float]]


Outputs

results/<run_id>/stage1_alpha.json – α(f) per material

results/<run_id>/stage1_summary.csv – fit errors by metric & band

results/<run_id>/stage1_preview.png – quick plots (optional)

Phase 2 — First forward test (Treble) + comparison

Goal: validate Stage-1 α(f) via a single Treble forward run, and compare Treble hybrid.json vs results_main.json at each receiver with clear, readable errors.

Script: 20_forward_once.py

Converts stage1_alpha.json into Treble material definitions and assignments.

Ensures CF2 is uploaded; constructs simulation with sources/receivers from config.

Submits the run, waits (or polls), fetches results (hybrid JSON + optional IR WAV).

Calls comparator to compute per-metric, per-receiver errors and a global score.

Script: 21_compare_hybrid_vs_main.py

Extracts selected metric (e.g., EDT or T20) and prints verbatim rows like:

r2 @ 1000 Hz — EDT: sim 0.71 s | ref 0.66 s | Δ +0.05 s


Computes norm/error on the same metrics (MAE/RMSE/Huber), shows overall loss.

Saves results/<run_id>/forward_eval.csv and a small HTML/PNG report.

Key functions

# src/treble_client.py
def init_tsdk() -> object
def ensure_project(name: str) -> object
def ensure_model(project, obj_path: str, model_name: str) -> object
def ensure_materials(tsdk, alpha_by_material: dict, bands, extra=dict) -> dict
def ensure_directivity(tsdk, cf2_path: str) -> object
def build_simulation(project, model, mats, cfg) -> object
def start_and_wait(sim) -> object  # returns results object/paths
def export_ir_and_metrics(sim, out_dir: str) -> dict  # paths; hybrid.json etc.

# src/eval_forward.py
def load_hybrid_results(hybrid_json_path: str) -> dict
def compare_to_ground_truth(hybrid: dict, gt: dict, cfg) -> dict
def write_comparison_csv(rows, out_csv_path: str) -> None
def global_loss(rows, weights) -> float


Outputs

results/<run_id>/hybrid.json (or Treble’s default output path copied in)

results/<run_id>/forward_eval.csv

results/<run_id>/score.txt (single scalar you can eyeball per iteration)

Phase 3 — Full GA loop (Stage-2)

Goal: iterate on α(f) (and optionally scattering s(f) or a subset of materials/bands) using Treble forward runs in the fitness loop; report convergence and keep best.

Script: 30_ga_loop.py

GA genome: concatenated vector of parameters (e.g., α for selected materials × bands; later enable scattering).

Initial population: centred on Stage-1 α with small random jitter respecting constraints.

Mutation: per-gene Gaussian or uniform within bounds; post-mutate clamp + monotonic fix.

Crossover: one-point or SBX; keep children valid via constraints.

Fitness: for each individual:

Write material set → Treble-ready definitions.

Launch forward run (possibly reusing geometry & CF2).

Evaluate global_loss() vs ground truth.

Cache (hash → score) to avoid duplicate sims.

Selection: tournament or NSGA-II if multi-objective (e.g., EDT & C50).

Stagnation control: if best score doesn’t improve N generations → random immigrant restart.

Checkpointing: save ga_state.pkl and hall_of_fame.json each gen.

Key functions

# src/ga_core.py
@dataclass
class GASettings: population:int; generations:int; mutation_rate:float; ...
def make_initial_population(stage1_alpha, settings, bounds, mask) -> list[individual]
def mutate(individual, settings, constraints) -> individual
def crossover(p1, p2, settings, constraints) -> (child1, child2)
def evaluate(individual, context) -> float  # runs a forward sim and returns loss
def run_ga(settings, context) -> dict  # returns best, history, hof

# src/materials.py
def pack_alpha(alpha_dict, order_spec) -> np.ndarray
def unpack_alpha(vector, order_spec) -> dict[str, list[float]]
def apply_constraints(alpha_dict, cfg) -> dict[str, list[float]]


Outputs

results/<run_id>/ga_history.csv (best/mean loss per gen)

results/<run_id>/best_alpha.json

results/<run_id>/best_forward_eval.csv

results/<run_id>/ga_convergence.png

Data contracts (so components interoperate)
results_main.json (ground truth)
{
  "bands": {"type": "third", "f": [63, 80, ..., 8000]},
  "receivers": ["r1","r2", "..."],
  "metrics": {
    "EDT": { "r1": [.. per-band ..], "r2": [...], "...": [...] },
    "T20": { "r1": [...], "r2": [...] }
  }
}

stage1_alpha.json
{
  "My_Plasterboard_Ceiling": {"bands_hz":[63,...,8000], "alpha":[0.12,0.13,...]},
  "My_Painted_Plasterboard": {"bands_hz":[...], "alpha":[...]}
}

forward_eval.csv

Columns:

receiver, metric, f_hz, sim_value, ref_value, delta, abs_delta, weight


(plus an aggregated last row per metric with MAE/Huber and a global weighted score)

Error / norm definitions (simple & readable)

Per band & receiver: Δ = sim − ref

MAE (default): average of |Δ| across (receivers × bands × metrics) with metric weights.

Huber(δ): quadratic if |Δ|≤δ, linear otherwise; δ default 0.5 s for decay metrics, 1.0 dB for clarity.

Global score: weighted sum across metrics; lower is better. Print it every run.

Logging & run IDs

Every script starts by creating a run_id = YYYYMMDD_HHMMSS.

All outputs go under results/<run_id>/....

Write results/<run_id>/run_manifest.json summarising config hash, genome mask, Treble job IDs, and file paths.

Minimal execution order

python scripts/00_validate_config.py

python scripts/01_mesh_scan.py

python scripts/10_stage1_stat_fit.py

python scripts/20_forward_once.py

python scripts/21_compare_hybrid_vs_main.py → confirm reasonable match

python scripts/30_ga_loop.py → full optimisation

Practical guardrails

Token discipline: estimate tokens before starting; abort if estimate > cap.

Caching: hash the α-vector (rounded) → reuse previous score if identical.

Time safety: limit concurrent sims to 1–2; queue others with back-off.

Resumability: GA can resume from ga_state.pkl + hall_of_fame.json.