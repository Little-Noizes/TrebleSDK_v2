#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_stage3_ga.py — Two-Stage GA Calibration (Absorption → Scattering)

Key features
------------
- Central GA workdir controls ALL I/O (seed copy, per-eval params, logs).
- Initial population centred on latest Stage-1 seed (stage1_alpha.json).
- Explicit params handoff to forward via --params_in_file (race-free).
- Runs forward as a MODULE:  python -m src.run_stage1_and_forward
  (so `src.*` imports resolve); auto-detects project root (folder with src/).
- Parallel evaluations (multiprocessing), evaluation cache, early-stop.
- Per-stage checkpoint/resume (pop/HOF/gen/RNG), per-gen receipts.
- Robust diagnostics: prints tail of stdout/stderr on failures.

CLI
---
(venv_treble) PS> python run_stage3_ga.py ^
  --config .\configs\project.yaml ^
  --workdir .\results\GA_calib_001 ^
  --label GA_calib_001

YAML (ga: block)
----------------
ga:
  pop_size: 18
  generations_A: 12
  generations_B: 10
  alpha_sigma: 0.03
  scatter_sigma: 0.05
  alpha_bounds: [0.0, 1.0]
  scatter_bounds: [0.0, 0.5]
  selection: "tournament"   # or "roulette"
  tourn_k: 3
  cx_prob: 0.5
  mut_prob: 0.7
  cx_eta: 0.3
  stageA_metrics: ["t20", "t30"]
  stageB_metrics: ["c50", "c80"]
  patience_A: 4
  patience_B: 4
  n_workers: 4
  eval_cache_round: 5
  forward_timeout_s: 5400

Outputs
-------
- <workdir>/ga_seed/stage1_alpha.json   (copied seed)
- <workdir>/eval_A_*/params.json        (candidate α for Stage A)
- <workdir>/eval_B_*/params.json        (candidate s for Stage B, α fixed)
- <workdir>/best_individual_stageA.json
- <workdir>/best_individual_stageB.json
- <workdir>/calibrated_parameters.json
- <workdir>/ga_log.csv
- <workdir>/chkpt_A.pkl, chkpt_B.pkl
- <workdir>/gen_receipts/stage{A|B}_genXX.json
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import yaml

try:
    from deap import base, creator, tools
except Exception:
    raise SystemExit("Missing dependency 'deap'. Install with: pip install deap")

# =============================
# Helpers (filesystem & parsing)
# =============================

def _read_yaml(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    data = yaml.safe_load(txt)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping.")
    return data

def _write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _find_latest_stage1_alpha(root: Path) -> Path:
    candidates = list(root.rglob("stage1_alpha.json"))
    if not candidates:
        raise FileNotFoundError(f"No stage1_alpha.json found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _load_stage1_alpha(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

# =====================
# Model parameterisation
# =====================

@dataclass
class GASettings:
    pop_size: int
    generations_A: int
    generations_B: int
    alpha_sigma: float
    scatter_sigma: float
    alpha_bounds: Tuple[float, float]
    scatter_bounds: Tuple[float, float]
    selection: str
    tourn_k: int
    cx_prob: float
    mut_prob: float
    cx_eta: float
    stageA_metrics: List[str]
    stageB_metrics: List[str]
    patience_A: int
    patience_B: int
    n_workers: int
    eval_cache_round: int
    forward_timeout_s: int
    max_evaluations: int = 25    # <-- ADD THIS with default
    max_tokens: float = 25     # <-- ADD THIS with default

@dataclass
class Problem:
    base_cfg_path: Path
    workdir: Path
    label: str
    output_root: Path
    results_root: Path
    stage1_alpha_path: Path
    seed_alpha: Dict[str, Any]
    bands: List[int]
    materials_order: List[str]
    scatter_keys: List[str]
    ga: GASettings
    python_exe: str
    seed_dir: Path
    project_root: Path  # directory containing src/

# ====================
# Vector ↔ struct maps
# ====================

def vector_from_alpha(seed_alpha: Dict[str, Any], materials_order: List[str]) -> List[float]:
    vec: List[float] = []
    for m in materials_order:
        vec.extend([float(x) for x in seed_alpha[m]["alpha"]])
    return vec

def alpha_from_vector(vec: List[float], seed_alpha: Dict[str, Any], materials_order: List[str]) -> Dict[str, Any]:
    out = {}
    i = 0
    for m in materials_order:
        bands = seed_alpha[m]["bands_hz"]
        n = len(bands)
        out[m] = {
            "bands_hz": bands,
            "alpha": [float(max(0.0, min(1.0, vec[i + k]))) for k in range(n)],
        }
        i += n
    return out

def vector_bounds_alpha(seed_alpha: Dict[str, Any], materials_order: List[str], bounds: Tuple[float, float]) -> List[Tuple[float, float]]:
    lo, hi = bounds
    bnds: List[Tuple[float, float]] = []
    for m in materials_order:
        n = len(seed_alpha[m]["alpha"])
        bnds.extend([(lo, hi)] * n)
    return bnds

def vector_from_scatter(yaml_cfg: Dict[str, Any], scatter_keys: List[str]) -> List[float]:
    out: List[float] = []
    mats = yaml_cfg.get("materials", {}) or {}
    for m in scatter_keys:
        node = mats.get(m, {}) or {}
        anchors = (node.get("anchors") or {})
        s = anchors.get("scattering")
        if isinstance(s, list) and s:
            out.append(float(s[0]))
        elif isinstance(s, (int, float)):
            out.append(float(s))
        else:
            out.append(0.10)  # default
    return out

def scatter_from_vector(vec: List[float], scatter_keys: List[str]) -> Dict[str, float]:
    return {k: float(max(0.0, min(1.0, v))) for k, v in zip(scatter_keys, vec)}

# =====================
# Temp YAML & params I/O
# =====================

def make_temp_yaml_with_candidate(
    base_cfg: Dict[str, Any],
    out_dir: Path,
    candidate_alpha: Optional[Dict[str, Any]] = None,
    candidate_scatter: Optional[Dict[str, float]] = None,
    *,
    fix_alpha: bool = False,
    stage_metrics: Optional[List[str]] = None,
) -> Path:
    """Fallback path if forward lacks --params_in_file support."""
    cfg = deepcopy(base_cfg)

    if candidate_alpha:
        mats = cfg.get("materials", {}) or {}
        for m, spec in candidate_alpha.items():
            if m not in mats: 
                continue
            node = mats[m] or {}
            anchors = node.get("anchors") or {}
            anchors["absorption"] = list(map(float, spec["alpha"]))
            node["anchors"] = anchors
            if fix_alpha:
                node["optimise_absorption"] = False
                node["deviation_groups"] = {
                    "low":  {"bands": [63, 125], "deviation_pct": 0.0},
                    "mid":  {"bands": [250, 500, 1000], "deviation_pct": 0.0},
                    "high": {"bands": [2000, 4000, 8000], "deviation_pct": 0.0},
                }
            mats[m] = node
        cfg["materials"] = mats

    if candidate_scatter:
        mats = cfg.get("materials", {}) or {}
        for m, sval in candidate_scatter.items():
            if m not in mats:
                continue
            node = mats[m] or {}
            anchors = node.get("anchors") or {}
            anchors["scattering"] = float(sval)
            node["anchors"] = anchors
            node["optimise_scattering"] = True
            mats[m] = node
        cfg["materials"] = mats

    if stage_metrics is not None:
        lossnode = cfg.get("loss", {}) or {}
        lossnode["metrics"] = stage_metrics
        cfg["loss"] = lossnode

    _ensure_dir(out_dir)
    tmp_yaml = out_dir / "project_tmp.yaml"
    _write_yaml(tmp_yaml, cfg)
    return tmp_yaml

def write_params_file(eval_dir: Path, alpha_block: Optional[Dict[str, Any]], scatter_block: Optional[Dict[str, float]]) -> Path:
    """Create `<eval_dir>/params.json` with keys the forward understands."""
    _ensure_dir(eval_dir)
    payload = {
        "alpha": alpha_block or {},
        "scatter": scatter_block or {}
    }
    path = eval_dir / "params.json"
    _write_json(path, payload)
    return path

# ==================
# Forward evaluation
# ==================

_loss_re = re.compile(r"^\s*Loss:\s*([+-]?(?:\d+\.\d+|\d+))\s*$", re.M)
_token_cost_re = re.compile(r"Project cost:\s*([0-9.]+)\s*tokens", re.I)
_balance_re = re.compile(r"Balance:\s*([0-9.]+)\s*tokens", re.I)

def _detect_project_root(start: Path) -> Path:
    """Walk up a few levels to find a folder that contains src/run_stage1_and_forward.py"""
    cur = start.resolve()
    for root in [cur, cur.parent, cur.parent.parent, cur.parent.parent.parent]:
        if (root / "src" / "run_stage1_and_forward.py").exists():
            return root
    raise RuntimeError("Could not locate project root containing src/run_stage1_and_forward.py")

def call_forward_and_get_loss(config_path: Path,
                              workdir: Path,
                              run_label: str,
                              python_exe: str,
                              params_in_file: Optional[Path],
                              project_root: Path,
                              timeout_s: int) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Run forward as a module from project_root; parse scalar Loss and token info.
    Returns: (loss, token_cost, balance_remaining)
    """
    cmd = [
        python_exe, "-m", "src.run_stage1_and_forward",
        "--config", str(config_path),
        "--run_label", run_label
    ]
    if params_in_file is not None:
        cmd += ["--params_in_file", str(params_in_file)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    # Stream output in real-time while also capturing it
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(project_root),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, shell=False
        )
        
        stdout_lines = []
        stderr_lines = []
        
        # Read stdout in real-time
        import select
        import sys
        
        while True:
            # Read available output
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                stdout_lines.append(line)
                # Display in real-time (with prefix for clarity)
                print(f"  [fwd] {line.rstrip()}")
                sys.stdout.flush()
        
        # Get any remaining stderr
        stderr = proc.stderr.read()
        if stderr:
            stderr_lines = stderr.splitlines(keepends=True)
        
        # Wait for completion
        proc.wait(timeout=timeout_s)
        
    except subprocess.TimeoutExpired as e:
        proc.kill()
        raise RuntimeError(f"Forward timed out after {timeout_s}s.\nCMD: {' '.join(cmd)}") from e

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    
    if proc.returncode != 0:
        tail_out = "\n".join(stdout.splitlines()[-80:])
        tail_err = "\n".join(stderr.splitlines()[-80:])
        raise RuntimeError(f"Forward rc={proc.returncode}\nSTDOUT (tail):\n{tail_out}\n\nSTDERR (tail):\n{tail_err}")

    # Parse loss
    m = _loss_re.search(stdout)
    if not m:
        tail_out = "\n".join(stdout.splitlines()[-80:])
        raise RuntimeError(f"Could not parse 'Loss:' line from forward.\nSTDOUT (tail):\n{tail_out}")
    loss = float(m.group(1))
    
    # Parse token cost
    token_cost = None
    m_cost = _token_cost_re.search(stdout)
    if m_cost:
        token_cost = float(m_cost.group(1))
    
    # Parse balance
    balance = None
    m_bal = _balance_re.search(stdout)
    if m_bal:
        balance = float(m_bal.group(1))
    
    return loss, token_cost, balance
# ======================
# GA: toolbox & operators
# ======================

def make_toolbox(problem: Problem, stage: str) -> Tuple[base.Toolbox, List[Tuple[float, float]]]:
    tb = base.Toolbox()

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    if stage == "A":
        seed_vec = vector_from_alpha(problem.seed_alpha, problem.materials_order)
        bounds = vector_bounds_alpha(problem.seed_alpha, problem.materials_order, problem.ga.alpha_bounds)
        sigma = problem.ga.alpha_sigma

        def _rand_gene(i):
            lo, hi = bounds[i]
            val = random.gauss(seed_vec[i], sigma)
            return max(lo, min(hi, val))

        tb.register("attr_float", _rand_gene, 0)
        tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_float, n=len(seed_vec))
        tb.register("population", tools.initRepeat, list, tb.individual)
        gene_bounds = bounds

    elif stage == "B":
        seed_s = vector_from_scatter(_read_yaml(problem.base_cfg_path), problem.scatter_keys)
        lo, hi = problem.ga.scatter_bounds

        def _rand_gene_s():
            mu = (lo + hi) / 2.0 if not seed_s else seed_s[0]
            val = random.gauss(mu, problem.ga.scatter_sigma)
            return max(lo, min(hi, val))

        tb.register("attr_float", _rand_gene_s)
        tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_float, n=len(problem.scatter_keys))
        tb.register("population", tools.initRepeat, list, tb.individual)
        gene_bounds = [(lo, hi)] * len(problem.scatter_keys)
    else:
        raise ValueError("stage must be 'A' or 'B'")

    if problem.ga.selection == "tournament":
        tb.register("select", tools.selTournament, tournsize=problem.ga.tourn_k)
    else:
        tb.register("select", tools.selRoulette)

    tb.register("mate", tools.cxBlend, alpha=problem.ga.cx_eta)

    def mut_gaussian(individual: List[float], mu=0.0, sigma=0.1, indpb=0.25):
        for i, x in enumerate(individual):
            if random.random() < indpb:
                individual[i] = x + random.gauss(mu, sigma)
        return (individual,)

    tb.register("mutate", mut_gaussian,
                mu=0.0,
                sigma=(problem.ga.alpha_sigma if stage == "A" else problem.ga.scatter_sigma),
                indpb=0.25)

    return tb, gene_bounds

# ===============
# Checkpoint utils
# ===============

def _chkpt_path(workdir: Path, stage: str) -> Path:
    return workdir / ("chkpt_A.pkl" if stage == "A" else "chkpt_B.pkl")

def save_checkpoint(workdir: Path, stage: str, gen: int, pop, hof, log_rows, rng_state):
    data = {"stage": stage, "gen": gen, "pop": pop, "hof": hof, "log": log_rows, "rng": rng_state}
    _ensure_dir(workdir)
    _chkpt_path(workdir, stage).write_bytes(pickle.dumps(data))

def try_resume(workdir: Path, stage: str):
    path = _chkpt_path(workdir, stage)
    if not path.exists():
        return None
    try:
        return pickle.loads(path.read_bytes())
    except Exception:
        return None

# ===================
# Eval cache (in-mem)
# ===================

class EvalCache:
    def __init__(self, digits: int):
        self.d = {}
        self.digits = digits
    def key(self, vec: List[float]):
        return tuple(round(x, self.digits) for x in vec)
    def get(self, vec: List[float]):
        return self.d.get(self.key(vec))
    def put(self, vec: List[float], val: float):
        self.d[self.key(vec)] = val

# =====================
# GA evaluate callbacks
# =====================

def evaluate_stageA_factory(problem: Problem, cache: EvalCache):
    base_cfg = _read_yaml(problem.base_cfg_path)
    def _eval(individual: List[float]) -> Tuple[float]:
        cached = cache.get(individual)
        if cached is not None:
            return (cached,)
        cand_alpha = alpha_from_vector(individual, problem.seed_alpha, problem.materials_order)
        eval_dir = problem.workdir / f"eval_A_{datetime.now():%Y%m%d_%H%M%S_%f}"
        _ensure_dir(eval_dir)
        params_path = write_params_file(eval_dir, alpha_block=cand_alpha, scatter_block=None)

        # Prefer modern handoff. If forward unpatched, fallback to anchors-in-YAML.
        try:
            loss = call_forward_and_get_loss(
                problem.base_cfg_path, eval_dir, f"{problem.label}_A",
                problem.python_exe, params_path, problem.project_root,
                timeout_s=problem.ga.forward_timeout_s
            )
        except Exception:
            tmp_yaml = make_temp_yaml_with_candidate(
                base_cfg=base_cfg, out_dir=eval_dir,
                candidate_alpha=cand_alpha, candidate_scatter=None,
                fix_alpha=False, stage_metrics=problem.ga.stageA_metrics
            )
            loss = call_forward_and_get_loss(
                tmp_yaml, eval_dir, f"{problem.label}_A",
                problem.python_exe, None, problem.project_root,
                timeout_s=problem.ga.forward_timeout_s
            )

        cache.put(individual, loss)
        return (loss,)
    return _eval

def evaluate_stageB_factory(problem: Problem, best_alpha_stageA: Dict[str, Any], cache: EvalCache):
    base_cfg = _read_yaml(problem.base_cfg_path)
    def _eval(individual: List[float]) -> Tuple[float]:
        cached = cache.get(individual)
        if cached is not None:
            return (cached,)
        cand_scatter = scatter_from_vector(individual, problem.scatter_keys)
        eval_dir = problem.workdir / f"eval_B_{datetime.now():%Y%m%d_%H%M%S_%f}"
        _ensure_dir(eval_dir)
        params_path = write_params_file(eval_dir, alpha_block=best_alpha_stageA, scatter_block=cand_scatter)

        try:
            loss = call_forward_and_get_loss(
                problem.base_cfg_path, eval_dir, f"{problem.label}_B",
                problem.python_exe, params_path, problem.project_root,
                timeout_s=problem.ga.forward_timeout_s
            )
        except Exception:
            tmp_yaml = make_temp_yaml_with_candidate(
                base_cfg=base_cfg, out_dir=eval_dir,
                candidate_alpha=best_alpha_stageA, candidate_scatter=cand_scatter,
                fix_alpha=True, stage_metrics=problem.ga.stageB_metrics
            )
            loss = call_forward_and_get_loss(
                tmp_yaml, eval_dir, f"{problem.label}_B",
                problem.python_exe, None, problem.project_root,
                timeout_s=problem.ga.forward_timeout_s
            )

        cache.put(individual, loss)
        return (loss,)
    return _eval

# =============
# Log utilities
# =============

def _save_log(workdir: Path, rows: List[Dict[str, Any]], append: bool=False) -> None:
    path = workdir / "ga_log.csv"
    _ensure_dir(workdir)
    fieldnames = ["stage", "gen", "best"]
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow(r)

def _save_gen_receipt(workdir: Path, stage: str, gen: int, best_loss: float, payload: Dict[str, Any]):
    d = workdir / "gen_receipts"
    _ensure_dir(d)
    (d / f"stage{stage}_gen{gen:02d}.json").write_text(
        json.dumps({
            "stage": stage,
            "gen": gen,
            "best_loss": best_loss,
            "payload": payload,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, indent=2),
        encoding="utf-8"
    )

# ==============
# GA main stages
# ==============

def run_stage_A(problem: Problem) -> Dict[str, Any]:
    tb, _ = make_toolbox(problem, stage="A")
    cache = EvalCache(problem.ga.eval_cache_round)

    pool = Pool(processes=problem.ga.n_workers) if problem.ga.n_workers > 1 else None
    tb.register("map", pool.map if pool else map)

    tb.register("evaluate", evaluate_stageA_factory(problem, cache))

    resumed = try_resume(problem.workdir, "A")
    # Initialize counters for stop conditions
    total_evaluations = 0
    total_tokens_spent = 0.0
    if resumed:
        pop = resumed["pop"]
        hof = resumed["hof"]
        log_rows = resumed["log"]
        start_gen = resumed["gen"] + 1
        random.setstate(resumed["rng"])
        print(f"[Stage A] Resuming at gen {start_gen}")
    else:
        pop = tb.population(n=problem.ga.pop_size)
        hof = tools.HallOfFame(1)
        log_rows = []
        start_gen = 0

    invalid = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(tb.map(tb.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    best_so_far = float("inf")
    stall = 0

    for gen in range(start_gen, problem.ga.generations_A):
        offspring = tb.select(pop, len(pop))
        offspring = list(map(tb.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < problem.ga.cx_prob:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mut in offspring:
            if random.random() < problem.ga.mut_prob:
                tb.mutate(mut)
                del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        # Check if we would exceed evaluation limit
        n_to_eval = len(invalid)
        if total_evaluations + n_to_eval > problem.ga.max_evaluations:
            print(f"\n[STOP] Would exceed max_evaluations={problem.ga.max_evaluations}")
            print(f"       Current: {total_evaluations}, Needed: {n_to_eval}")
            break

        fitnesses = list(tb.map(tb.evaluate, invalid))
        total_evaluations += n_to_eval
        # TODO: Add token tracking here if you can get cost per evaluation
        # For now, estimate: each evaluation ≈ some fixed token cost
        # You may need to read this from simulation results
        estimated_tokens_per_eval = 0.5  # Adjust based on your actual usage
        total_tokens_spent += n_to_eval * estimated_tokens_per_eval
        if total_tokens_spent >= problem.ga.max_tokens:
            print(f"\n[STOP] Exceeded max_tokens={problem.ga.max_tokens}")
            print(f"       Total tokens spent: {total_tokens_spent:.2f}")
            break

        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        best = hof[0]
        best_loss = float(best.fitness.values[0])
        log_rows.append({"stage": "A", "gen": gen, "best": best_loss})
        print(f"[Stage A] Gen {gen}: best loss = {best_loss:.6f}")
        _save_gen_receipt(problem.workdir, "A", gen, best_loss,
                          alpha_from_vector(best, problem.seed_alpha, problem.materials_order))

        if best_loss < best_so_far - 1e-4:
            best_so_far, stall = best_loss, 0
        else:
            stall += 1
        if stall >= problem.ga.patience_A:
            print(f"[Stage A] Early stopping at gen {gen} (no improvement for {stall} gens)")
            save_checkpoint(problem.workdir, "A", gen, pop, hof, log_rows, random.getstate())
            break

        save_checkpoint(problem.workdir, "A", gen, pop, hof, log_rows, random.getstate())

    best_alpha = alpha_from_vector(hof[0], problem.seed_alpha, problem.materials_order)
    (problem.workdir / "best_individual_stageA.json").write_text(json.dumps(best_alpha, indent=2), encoding="utf-8")
    _save_log(problem.workdir, log_rows)

    if pool:
        pool.close(); pool.join()
    return best_alpha

def run_stage_B(problem: Problem, best_alpha_stageA: Dict[str, Any]) -> Dict[str, Any]:
    tb, _ = make_toolbox(problem, stage="B")
    cache = EvalCache(problem.ga.eval_cache_round)

    pool = Pool(processes=problem.ga.n_workers) if problem.ga.n_workers > 1 else None
    tb.register("map", pool.map if pool else map)

    tb.register("evaluate", evaluate_stageB_factory(problem, best_alpha_stageA, cache))

    resumed = try_resume(problem.workdir, "B")
    if resumed:
        pop = resumed["pop"]
        hof = resumed["hof"]
        log_rows = resumed["log"]
        start_gen = resumed["gen"] + 1
        random.setstate(resumed["rng"])
        print(f"[Stage B] Resuming at gen {start_gen}")
    else:
        pop = tb.population(n=problem.ga.pop_size)
        hof = tools.HallOfFame(1)
        log_rows = []
        start_gen = 0

    invalid = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(tb.map(tb.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    best_so_far = float("inf")
    stall = 0

    for gen in range(start_gen, problem.ga.generations_B):
        offspring = tb.select(pop, len(pop))
        offspring = list(map(tb.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < problem.ga.cx_prob:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mut in offspring:
            if random.random() < problem.ga.mut_prob:
                tb.mutate(mut)
                del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(tb.map(tb.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        best = hof[0]
        best_loss = float(best.fitness.values[0])
        log_rows.append({"stage": "B", "gen": gen, "best": best_loss})
        print(f"[Stage B] Gen {gen}: best loss = {best_loss:.6f}")
        _save_gen_receipt(problem.workdir, "B", gen, best_loss,
                          scatter_from_vector(best, problem.scatter_keys))

        if best_loss < best_so_far - 1e-4:
            best_so_far, stall = best_loss, 0
        else:
            stall += 1
        if stall >= problem.ga.patience_B:
            print(f"[Stage B] Early stopping at gen {gen} (no improvement for {stall} gens)")
            save_checkpoint(problem.workdir, "B", gen, pop, hof, log_rows, random.getstate())
            break

        save_checkpoint(problem.workdir, "B", gen, pop, hof, log_rows, random.getstate())

    best_scatter = scatter_from_vector(hof[0], problem.scatter_keys)
    (problem.workdir / "best_individual_stageB.json").write_text(json.dumps(best_scatter, indent=2), encoding="utf-8")
    _save_log(problem.workdir, log_rows, append=True)

    if pool:
        pool.close(); pool.join()
    return best_scatter

# =====
# Main
# =====

def build_problem(args: argparse.Namespace) -> Problem:
    base_cfg_path = Path(args.config).resolve()
    base_cfg = _read_yaml(base_cfg_path)
    out_root = Path(args.workdir).resolve(); _ensure_dir(out_root)

    # resolve output/results roots from YAML if present
    paths = base_cfg.get("paths", {}) or {}
    output_root = Path(paths.get("output_root") or out_root).resolve()
    results_root = Path(paths.get("results_root") or output_root).resolve()

    # Detect project root (folder that contains src/)
    project_root = _detect_project_root(Path(__file__).parent)

    stage1_path = _find_latest_stage1_alpha(results_root)

    # copy seed into central GA folder
    seed_dir = out_root / "ga_seed"; _ensure_dir(seed_dir)
    shutil.copy2(stage1_path, seed_dir / "stage1_alpha.json")

    seed_alpha = _load_stage1_alpha(stage1_path)

    first_mat = next(iter(seed_alpha))
    bands = list(map(int, seed_alpha[first_mat]["bands_hz"]))
    materials_order = sorted(seed_alpha.keys())

    # which materials can have scattering optimised?
    scatter_keys: List[str] = []
    mats = base_cfg.get("materials", {}) or {}
    for mk, mv in mats.items():
        if isinstance(mv, dict) and mv.get("optimise_scattering", False):
            scatter_keys.append(mk)
    scatter_keys = sorted(set(scatter_keys))

    ga_node = base_cfg.get("ga", {}) or {}
    def _get(k, default):
        return ga_node.get(k, default)

    ga = GASettings(
        pop_size=int(_get("pop_size", 10)),
        generations_A=int(_get("generations_A", 5)),
        generations_B=int(_get("generations_B", 5)),
        alpha_sigma=float(_get("alpha_sigma", 0.03)),
        scatter_sigma=float(_get("scatter_sigma", 0.05)),
        alpha_bounds=tuple(map(float, _get("alpha_bounds", [0.0, 1.0]))),
        scatter_bounds=tuple(map(float, _get("scatter_bounds", [0.0, 0.5]))),
        selection=str(_get("selection", "tournament")),
        tourn_k=int(_get("tourn_k", 3)),
        cx_prob=float(_get("cx_prob", 0.5)),
        mut_prob=float(_get("mut_prob", 0.7)),
        cx_eta=float(_get("cx_eta", 0.3)),
        stageA_metrics=list(_get("stageA_metrics", ["t20", "t30"])),
        stageB_metrics=list(_get("stageB_metrics", ["c50", "c80"])),
        patience_A=int(_get("patience_A", 4)),
        patience_B=int(_get("patience_B", 4)),
        n_workers=int(_get("n_workers", 1)),
        eval_cache_round=int(_get("eval_cache_round", 5)),
        forward_timeout_s=int(_get("forward_timeout_s", 5400)),
    )

    python_exe = sys.executable

    return Problem(
        base_cfg_path=base_cfg_path,
        workdir=out_root,
        label=args.label,
        output_root=output_root,
        results_root=results_root,
        stage1_alpha_path=stage1_path,
        seed_alpha=seed_alpha,
        bands=bands,
        materials_order=materials_order,
        scatter_keys=scatter_keys,
        ga=ga,
        python_exe=python_exe,
        seed_dir=seed_dir,
        project_root=project_root,
    )

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Two-Stage GA with central workdir, params handoff, and parallel evals.")
    p.add_argument("--config", required=True, help="Path to base project.yaml")
    p.add_argument("--workdir", required=True, help="GA workdir (central)")
    p.add_argument("--label", default="ga_stage3", help="Run label prefix for forward calls")
    args = p.parse_args(argv)

    # Optional reproducibility seed
    if os.environ.get("GA_SEED"):
        seed = int(os.environ["GA_SEED"]); random.seed(seed)
        try:
            import numpy as _np; _np.random.seed(seed)
        except Exception:
            pass
        os.environ["PYTHONHASHSEED"] = str(seed)

    problem = build_problem(args)

    print("=== GA Stage 3 Driver ===")
    print(f"Base config:     {problem.base_cfg_path}")
    print(f"Workdir:         {problem.workdir}")
    print(f"Seed copied to:  {problem.seed_dir / 'stage1_alpha.json'}")
    print(f"Results root:    {problem.results_root}")
    print(f"Materials (|M|): {len(problem.materials_order)} → {problem.materials_order}")
    print(f"Bands:           {problem.bands}")
    print(f"Scatter keys:    {problem.scatter_keys}")
    print(f"GA: pop={problem.ga.pop_size} genA={problem.ga.generations_A} genB={problem.ga.generations_B} workers={problem.ga.n_workers}")

    # Stage A: α(f)
    best_alpha_A = run_stage_A(problem)

    # Stage B: scattering (α fixed)
    best_scatter_B = run_stage_B(problem, best_alpha_A)

    final = {
        "alpha": best_alpha_A,
        "scatter": best_scatter_B,
        "bands_hz": problem.bands,
        "materials": problem.materials_order,
        "scatter_keys": problem.scatter_keys,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "notes": "Stage-A absorption then Stage-B scattering; central GA folder + params handoff; parallel+cache+checkpoint+earlystop",
    }
    (problem.workdir / "calibrated_parameters.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n=== DONE ===")
    print(f"Saved: {problem.workdir / 'calibrated_parameters.json'}")

if __name__ == "__main__":
    main()
