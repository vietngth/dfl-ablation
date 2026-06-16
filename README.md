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

## Files

| file | description |
|---|---|
| `spo_vs_sfge.py` | CLI to run experiment by choice |
| `dfl.py` | All logic code for solvers and problems are crammed up here |
| `run_all.py` | Run all experiments |
