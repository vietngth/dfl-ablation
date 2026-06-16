# SPO+ vs. SFGE comparison

Comparing some opt problems using SPO+ vs. SFGE. 

## The two methods

| | SPO+ | SFGE |
|---|---|---|
| Full name | Smart "Predict, then Optimize"+ | Score-Function Gradient Estimation |
| Reference | Elmachtoub & Grigas, *Management Science* 2022 | Silvestri et al., *JAIR* 2024 ([arXiv:2307.05213](https://arxiv.org/abs/2307.05213)) |
| Type | **Convex surrogate** of the SPO regret | **Gradient-free** (REINFORCE / score-function) on the *true* realized decision loss |
| Solver requirement | Needs the optimization solution + the linear objective structure | Treats the solver as a **black box** — forward solve only, no differentiation |
| Source here | reused directly from PyEPO (`pyepo.func.SPOPlus`) | score-function implementation in `dfl.py` (`train_sfge`) with a per-batch baseline for variance reduction |

Both train the **same** linear predictor `c_hat = W x` (no hidden layers), start from the same
initialization (cold, or a shared two-stage MSE warm start), and are scored by *PyEPO's
Gurobi-evaluated normalized regret*.

> Normalized regret = `(c·w(c_hat) − c·w*(c)) / |c·w*(c)|`, averaged over the test set, where
> `w(·)` is the decision the predicted cost induces and `w*` the true optimum. Lower is better;
> 0 means decisions are as good as if the true costs were known.

## Install (uv)

```bash
cd spo-vs-sfge
uv sync                       # creates .venv and installs numpy/scipy/torch/pyepo/gurobipy
uv run python spo_vs_sfge.py --problem knapsack --deg 4 --seeds 0,1,2
```

`torch` runs on GPU if available and falls back to CPU automatically (`dfl.dev`).
**Gurobi** is required by PyEPO's optimization models and the regret metric; the free
size-limited license covers every problem here. If you have a full license it is picked up
automatically.

## Usage

```bash
# running 4 problems
uv run python spo_vs_sfge.py --problem shortest_path   # 5x5 grid shortest path   (LP)
uv run python spo_vs_sfge.py --problem knapsack        # 0/1 knapsack, 16 items   (ILP)
uv run python spo_vs_sfge.py --problem tsp             # symmetric TSP, 8 nodes   (ILP)
uv run python spo_vs_sfge.py --problem portfolio       # Markowitz, 20 assets     (SOCP)

# misspecification & noise:
uv run python spo_vs_sfge.py --problem knapsack --deg 6
uv run python spo_vs_sfge.py --problem shortest_path --deg 6 --noise 0.5

# fairness / budget knobs:
uv run python spo_vs_sfge.py --problem knapsack --init warm --seeds 0,1,2,3,4
uv run python spo_vs_sfge.py --problem portfolio --spo-epochs 200 --sfge-epochs 150
```

Key flags (`--help` for all): `--deg` misspecification degree, `--noise` noise half-width,
`--init {cold,warm}` shared initialization, `--n-train`, `--seeds`, and per-method epoch/lr knobs.

## Implication of --deg
Elmachtoub–Grigas synthetic generator of *PyEPO* is dopted for this tag. See:

(`pyepo.data.shortestpath.genData`, `pyepo.data.knapsack.genData`, also `tsp`/`portfolio`).

Features `x ∈ R^p` are standard Gaussian; a fixed random Bernoulli(½) matrix `B` maps them to costs
through a **polynomial of degree `deg`**:

```
c_i  =  ( (B x)_i / sqrt(p)  +  3 )^deg  +  1            # then rescaled by / 3.5^deg
c_i *=  epsilon_i ,   epsilon_i ~ Uniform(1 − noise_width, 1 + noise_width)
```

(knapsack additionally multiplies by 5 and rounds up to integer values). So:

- **`deg = 1`** → the generator's signal is affine in the features. With the default
  bias-free predictor this is not strictly well-specified for every problem; use `--bias`
  to run the corresponding sensitivity check.
- **`deg > 1`** → the cost is a degree-`deg` *polynomial* the linear model **cannot represent** →
  **misspecified**. The higher `deg`, the larger the bias, and the more the *decision-aware* loss
  can beat MSE — and the more the choice between SPO+ and SFGE matters.
- **`--noise`** sets `noise_width`, the half-width of the multiplicative `Uniform` noise (added
  observation noise on top of the misspecification). Available for `shortest_path`/`knapsack`;
  the PyEPO `tsp`/`portfolio` generators expose no noise hook, so `--noise` is ignored there.

This is the standard misspecification protocol used throughout the SPO/DFL literature
(Elmachtoub & Grigas 2022; the PyEPO benchmark of Tang & Khalil 2022).

## Results

<!-- RESULTS:START -->
The static tables below are prior reference numbers: **normalized regret, lower is better**,
mean ± std over 3 seeds, **cold init**, `n_train = 400`, linear predictor, scored by
PyEPO/Gurobi. The current deterministic runner writes the authoritative per-run metadata and
regrets to JSON; use `uv run python run_all.py` to regenerate current results.

**TL;DR**: SFGE is not uniformly worse than SPO+, but robustly beats SPO+ on ILP (knapsack) and SOCP (portfolio) problems, including the misspecification settings. SPO+ still wins LP (shortest path) and TSP.

### Table 1: 4 problems (deg = 4)

| problem | category | SPO+ | SFGE | winner |
|---|---|---|---|---|
| shortest_path | LP | **0.0078** ± 0.0025 | 0.0155 ± 0.0030 | SPO+ |
| knapsack | ILP | 0.0753 ± 0.0038 | **0.0625** ± 0.0036 | **SFGE** |
| tsp | ILP | **0.0298** ± 0.0106 | 0.0562 ± 0.0106 | SPO+ |
| portfolio | SOCP | 0.0276 ± 0.0004 | **0.0259** ± 0.0003 | **SFGE** |

### Table 2: misspecification with **knapsack (ILP)**

| deg | SPO+ | SFGE | SFGE improvement |
|---|---|---|---|
| 1 (affine signal) | 0.1901 ± 0.0036 | **0.1605** ± 0.0030 | −16% |
| 2 | 0.1406 ± 0.0067 | **0.1168** ± 0.0025 | −17% |
| 4 | 0.0754 ± 0.0042 | **0.0625** ± 0.0014 | −17% |
| 6 | 0.0605 ± 0.0023 | **0.0414** ± 0.0047 | −32% |
| 8 (severe) | 0.0667 ± 0.0085 | **0.0317** ± 0.0042 | **−52%** |

### Table 3: misspecification with **shortest_path (LP)** 

| deg | SPO+ | SFGE | winner |
|---|---|---|---|
| 1 | **0.0001** ± 0.0000 | 0.0025 ± 0.0016 | SPO+ |
| 2 | **0.0010** ± 0.0003 | 0.0036 ± 0.0009 | SPO+ |
| 4 | **0.0088** ± 0.0024 | 0.0253 ± 0.0026 | SPO+ |
| 6 | **0.0243** ± 0.0091 | 0.0576 ± 0.0106 | SPO+ |
| 8 | **0.0546** ± 0.0194 | 0.1756 ± 0.0420 | SPO+ |

### Table 4: noise with **shortest_path (LP)**, deg = 6

| noise_width | SPO+ | SFGE | winner |
|---|---|---|---|
| 0.0 | **0.0263** ± 0.0106 | 0.0505 ± 0.0068 | SPO+ |
| 0.25 | **0.0391** ± 0.0064 | 0.0640 ± 0.0114 | SPO+ |
| 0.5 | **0.0855** ± 0.0081 | 0.1100 ± 0.0158 | SPO+ |
| 1.0 | **0.3193** ± 0.0299 | 0.3441 ± 0.0299 | SPO+ |

### Table 5: learning-rate sweep — **full per-`lr` results** (deg = 4, 3 seeds, cold init)

Tables 1–4 fix the Adam learning rate at `1e-2` for **both** methods. But the step size is a
sensitive knob — *especially* for SFGE, whose REINFORCE estimator barely moves at small `lr` and
needs a larger step than SPO+. A single-`lr` comparison can therefore flatter or punish either
method. Below is the **complete sweep**: every `lr ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`, SPO+ and SFGE
side by side, mean ± std over 3 seeds (normalized regret, lower is better; **bold** = that method's
best `lr`; `lr=1e-2` is the row used by Tables 1–4). Machine-readable: `experiment_results/sweep_lr.json`.

#### Table 5a — shortest_path (LP)
| lr | SPO+ | SFGE | winner @ lr |
|---|---|---|---|
| 1e-3 | 0.0306 ± 0.0198 | 0.3081 ± 0.0904 | SPO+ |
| 3e-3 | **0.0082 ± 0.0034** | 0.0888 ± 0.0331 | SPO+ |
| 1e-2 | 0.0091 ± 0.0035 | 0.0180 ± 0.0039 | SPO+ |
| 3e-2 | 0.0104 ± 0.0034 | 0.0106 ± 0.0035 | SPO+ |
| 1e-1 | 0.0128 ± 0.0060 | **0.0092 ± 0.0035** | SFGE |

#### Table 5b — knapsack (ILP)
| lr | SPO+ | SFGE | winner @ lr |
|---|---|---|---|
| 1e-3 | 0.1537 ± 0.0015 | 0.2609 ± 0.0031 | SPO+ |
| 3e-3 | 0.0906 ± 0.0036 | 0.1336 ± 0.0086 | SPO+ |
| 1e-2 | 0.0764 ± 0.0035 | 0.0627 ± 0.0022 | SFGE |
| 3e-2 | **0.0704 ± 0.0036** | **0.0595 ± 0.0016** | SFGE |
| 1e-1 | 0.0709 ± 0.0038 | 0.0602 ± 0.0024 | SFGE |

#### Table 5c — tsp (ILP)
| lr | SPO+ | SFGE | winner @ lr |
|---|---|---|---|
| 1e-3 | 0.1792 ± 0.0341 | 0.6142 ± 0.1009 | SPO+ |
| 3e-3 | 0.0651 ± 0.0192 | 0.2356 ± 0.0286 | SPO+ |
| 1e-2 | 0.0327 ± 0.0129 | 0.0559 ± 0.0158 | SPO+ |
| 3e-2 | 0.0267 ± 0.0101 | 0.0311 ± 0.0122 | SPO+ |
| 1e-1 | **0.0246 ± 0.0089** | **0.0273 ± 0.0109** | SPO+ |

#### Table 5d — portfolio (SOCP)
| lr | SPO+ | SFGE | winner @ lr |
|---|---|---|---|
| 1e-3 | 0.0487 ± 0.0046 | 0.1159 ± 0.0153 | SPO+ |
| 3e-3 | **0.0260 ± 0.0005** | 0.0517 ± 0.0012 | SPO+ |
| 1e-2 | 0.0289 ± 0.0006 | 0.0255 ± 0.0005 | SFGE |
| 3e-2 | 0.0324 ± 0.0018 | **0.0247 ± 0.0004** | SFGE |
| 1e-1 | 0.0402 ± 0.0023 | 0.0248 ± 0.0004 | SFGE |

#### Best-tuned head-to-head (each method at its own best `lr`)
| problem | category | SPO+ (best lr) | SFGE (best lr) | best-tuned winner |
|---|---|---|---|---|
| shortest_path | LP   | **0.0082** ± 0.0034 @ 3e-3 | 0.0092 ± 0.0035 @ 1e-1 | SPO+ |
| knapsack      | ILP  | 0.0704 ± 0.0036 @ 3e-2 | **0.0595** ± 0.0016 @ 3e-2 | **SFGE** |
| tsp           | ILP  | **0.0246** ± 0.0089 @ 1e-1 | 0.0273 ± 0.0109 @ 1e-1 | SPO+ |
| portfolio     | SOCP | 0.0260 ± 0.0005 @ 3e-3 | **0.0247** ± 0.0004 @ 3e-2 | **SFGE** |

**Reading the per-`lr` tables.** At **small `lr`** SPO+ wins almost everywhere — not because it is
better, but because SFGE's score-function estimator barely leaves the initialization (e.g. SFGE on
shortest_path is 0.31 at `lr=1e-3`, vs 0.009 at `lr=1e-1` — a 33× swing). As `lr` grows SFGE catches
up and overtakes on the integer/conic problems. The fixed `lr=1e-2` row (Tables 1–4) sits in a region
that happens to favor SPO+ on shortest_path; the per-`lr` view shows that is a tuning effect, not a
structural gap.

**What the sweep changes — and what it doesn't.**
- **The qualitative conclusion survives tuning.** With each method at its own best `lr`, SPO+ still
  wins the LP/TSP-style decisions and SFGE still wins the integer-knapsack/SOCP-portfolio decisions
  — the same split as the fixed-`lr` tables. The headline result is *not* a learning-rate artifact.
- **But the LP margin shrinks sharply.** At the fixed `lr=1e-2`, SFGE on shortest_path looks ~2×
  worse than SPO+ (0.0180 vs 0.0091). That gap is mostly a tuning artifact: SFGE at `lr=1e-2` is far
  from its optimum and recovers to **0.0092** at `lr=1e-1` — within ~12% of SPO+. So the fixed-`lr`
  Tables 3–4 *overstate* SPO+'s LP dominance.
- **SFGE is the more `lr`-sensitive method** (its regret spread across the grid is 3–13× SPO+'s, e.g.
  shortest_path 0.299 vs 0.022): at too-small `lr` it stalls near the init. SPO+ is sensitive too,
  but over a narrower band. Both should be tuned; reporting a single shared `lr` is not a fair test.

## Reproducing the tables

```bash
uv run python run_all.py        # prints Tables 1–4 and writes experiment_results/results.json
```

`run_all.py` uses the same deterministic execution helper as the CLI, sweeping problems / degrees /
noise and recording per-seed regrets, seed streams, run configuration, and solver sanity checks in
JSON. Full default TSP/SPO+ runs are solver-heavy; for a quick bounded artifact, reduce the explicit
budgets, for example:

```bash
uv run python run_all.py --n-test 200 --spo-epochs 5 --sfge-epochs 10 --sfge-samples 4 \
  --check-solvers --json-out experiment_results/results.json
```

For the learning-rate sweep (Table 5):

```bash
uv run python sweep_lr.py --problems shortest_path,knapsack,tsp,portfolio --deg 4 \
  --lrs 1e-3,3e-3,1e-2,3e-2,1e-1 --seeds 0,1,2 \
  --json-out experiment_results/sweep_lr.json --md-out experiment_results/sweep_lr.md
```

It writes per-`lr` regret curves (`sweep_lr.md`) and a machine-readable artifact (`sweep_lr.json`).
TSP/SPO+ is solver-heavy (a Gurobi MTZ solve per instance per step); the full sweep takes a while.

## Files

| file | description |
|---|---|
| `spo_vs_sfge.py` | CLI to run experiment by choice |
| `dfl.py` | All logic code for solvers and problems are crammed up here |
| `run_all.py` | Run all experiments (Tables 1–4) |
| `sweep_lr.py` | Learning-rate sweep, fair per-method tuning (Table 5) |
