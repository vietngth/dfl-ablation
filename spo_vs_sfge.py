#!/usr/bin/env python
"""
Sidenote: both SPO and SFGE use the same predictor (linear) and same initialization processes.
Normalized regret is taken from PyEPO's Gurobi's.

Examples
--------
  python spo_vs_sfge.py --problem knapsack --deg 4 --seeds 0,1,2
  python spo_vs_sfge.py --problem shortest_path --deg 6 --n-train 200 --noise 0.5
  python spo_vs_sfge.py --problem tsp --deg 2 --seeds 0,1,2,3,4 --init warm
  python spo_vs_sfge.py --problem portfolio --deg 8 --spo-epochs 200 --sfge-epochs 150
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pyepo import metric
import dfl


def build_cfg(problem, seed, deg, n_train, n_test, noise, bias=False):
    """Build the harness cfg dict for one of the four problems."""
    if problem not in dfl.SETUPS:
        raise ValueError(f"unknown problem {problem!r}")
    if noise and problem in ("tsp", "portfolio"):
        print(f"  [note] --noise is ignored for {problem} (no noise hook in its generator)")
    return dfl.SETUPS[problem](seed, deg, n_train=n_train, noise=noise, n_test=n_test, bias=bias)


def parse_seeds(seeds):
    """Parse a comma-separated seed list."""
    return [int(s.strip()) for s in seeds.split(",") if s.strip()]


def summarize(values):
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": arr.tolist()}


def winner_summary(spo_values, sfge_values):
    spo = summarize(spo_values)
    sfge = summarize(sfge_values)
    winner = "SPO+" if spo["mean"] < sfge["mean"] else "SFGE"
    loser_mean = sfge["mean"] if winner == "SPO+" else spo["mean"]
    winner_mean = spo["mean"] if winner == "SPO+" else sfge["mean"]
    lower_pct = (loser_mean - winner_mean) / max(loser_mean, 1e-12) * 100
    return {"SPO+": spo, "SFGE": sfge, "winner": winner, "winner_lower_regret_pct": float(lower_pct)}


def run_seed(
    problem,
    seed,
    deg,
    n_train,
    n_test,
    noise,
    init,
    spo_epochs,
    spo_lr,
    sfge_epochs,
    sfge_samples,
    sfge_sigma,
    sfge_lr,
    warm_epochs=40,
    bias=False,
    check_solvers=False,
):
    """Run both methods once from the same deterministic data/model seed."""
    cfg = build_cfg(problem, seed, deg, n_train, n_test, noise, bias=bias)
    initial_state = dfl.clone_state_dict(cfg["initial_state"])
    solver_check = dfl.solver_sanity(cfg) if check_solvers else None

    if init == "warm":
        warm = dfl.train_two_stage(cfg, epochs=warm_epochs, initial_state=initial_state)
        train_state = dfl.clone_state_dict(warm)
    else:
        train_state = initial_state

    spo_model = dfl.train_spoplus(cfg, epochs=spo_epochs, lr=spo_lr, initial_state=train_state)
    sfge_model = dfl.train_sfge(
        cfg,
        epochs=sfge_epochs,
        n_samples=sfge_samples,
        sigma=sfge_sigma,
        lr=sfge_lr,
        initial_state=train_state,
    )
    r_spo = metric.regret(spo_model, cfg["om"], cfg["ld_te"])
    r_sfge = metric.regret(sfge_model, cfg["om"], cfg["ld_te"])
    return {
        "problem": problem,
        "seed": int(seed),
        "deg": int(deg),
        "noise": float(noise),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "init": init,
        "bias": bool(bias),
        "device": dfl.dev,
        "seed_streams": {
            name: int(dfl.stream_seed(seed, name))
            for name in ["run", "model", "warm", "warm_loader", "spo", "spo_loader", "sfge"]
        },
        "solver_check": solver_check,
        "regret": {"SPO+": float(r_spo), "SFGE": float(r_sfge)},
        "winner": "SPO+" if r_spo < r_sfge else "SFGE",
    }


def run_experiment(
    problem,
    seeds,
    deg,
    n_train,
    n_test,
    noise,
    init,
    spo_epochs,
    spo_lr,
    sfge_epochs,
    sfge_samples,
    sfge_sigma,
    sfge_lr,
    warm_epochs=40,
    bias=False,
    check_solvers=False,
):
    runs = []
    for seed in seeds:
        runs.append(run_seed(
            problem=problem,
            seed=seed,
            deg=deg,
            n_train=n_train,
            n_test=n_test,
            noise=noise,
            init=init,
            spo_epochs=spo_epochs,
            spo_lr=spo_lr,
            sfge_epochs=sfge_epochs,
            sfge_samples=sfge_samples,
            sfge_sigma=sfge_sigma,
            sfge_lr=sfge_lr,
            warm_epochs=warm_epochs,
            bias=bias,
            check_solvers=check_solvers,
        ))
    return {
        "problem": problem,
        "deg": int(deg),
        "noise": float(noise),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "init": init,
        "bias": bool(bias),
        "training": {
            "warm_epochs": int(warm_epochs),
            "spo_epochs": int(spo_epochs),
            "spo_lr": float(spo_lr),
            "sfge_epochs": int(sfge_epochs),
            "sfge_samples": int(sfge_samples),
            "sfge_sigma": float(sfge_sigma),
            "sfge_lr": float(sfge_lr),
        },
        "runs": runs,
        "summary": winner_summary(
            [r["regret"]["SPO+"] for r in runs],
            [r["regret"]["SFGE"] for r in runs],
        ),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description="Fair SPO+ vs. SFGE comparison (PyEPO-native).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--problem", default="knapsack",
                    choices=["shortest_path", "knapsack", "tsp", "portfolio"])
    ap.add_argument("--deg", type=int, default=4, help="polynomial misspecification degree (1=well-specified)")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.0, help="multiplicative noise width (knapsack/shortest_path)")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--init", default="cold", choices=["cold", "warm"],
                    help="cold = both from scratch; warm = both from a two-stage MSE model (SAME for both)")
    ap.add_argument("--bias", action="store_true", help="use a bias term in the shared linear predictor")
    ap.add_argument("--warm-epochs", type=int, default=40)
    ap.add_argument("--spo-epochs", type=int, default=100)
    ap.add_argument("--spo-lr", type=float, default=1e-2)
    ap.add_argument("--sfge-epochs", type=int, default=120)
    ap.add_argument("--sfge-samples", type=int, default=8)
    ap.add_argument("--sfge-sigma", type=float, default=0.5)
    ap.add_argument("--sfge-lr", type=float, default=1e-2)
    ap.add_argument("--check-solvers", action="store_true",
                    help="compare each local SFGE solver with cached PyEPO optima on training costs")
    ap.add_argument("--json-out", help="write per-seed regrets, metadata, and summary to this JSON file")
    args = ap.parse_args()

    try:
        from scipy import stats
    except Exception:
        stats = None

    seeds = parse_seeds(args.seeds)
    print(f"\nSPO+ vs SFGE | problem={args.problem} deg={args.deg} n_train={args.n_train} "
          f"n_test={args.n_test} noise={args.noise} init={args.init} bias={args.bias} "
          f"seeds={seeds} device={dfl.dev}")
    print(f"SPO+: PyEPO SPOPlus ({args.spo_epochs} ep) | SFGE: score-function "
          f"({args.sfge_epochs} ep, {args.sfge_samples} samples, sigma={args.sfge_sigma}) | "
          f"metric: PyEPO normalized regret (Gurobi)\n", flush=True)

    result = run_experiment(
        problem=args.problem,
        seeds=seeds,
        deg=args.deg,
        n_train=args.n_train,
        n_test=args.n_test,
        noise=args.noise,
        init=args.init,
        spo_epochs=args.spo_epochs,
        spo_lr=args.spo_lr,
        sfge_epochs=args.sfge_epochs,
        sfge_samples=args.sfge_samples,
        sfge_sigma=args.sfge_sigma,
        sfge_lr=args.sfge_lr,
        warm_epochs=args.warm_epochs,
        bias=args.bias,
        check_solvers=args.check_solvers,
    )

    for run in result["runs"]:
        print(f"  seed {run['seed']}:  SPO+ = {run['regret']['SPO+']:.4f}   "
              f"SFGE = {run['regret']['SFGE']:.4f}", flush=True)

    spo = np.array(result["summary"]["SPO+"]["values"])
    sfge = np.array(result["summary"]["SFGE"]["values"])
    print("\n--- summary (normalized regret, lower is better) ---")
    print(f"  SPO+ : {spo.mean():.4f} ± {spo.std():.4f}")
    print(f"  SFGE : {sfge.mean():.4f} ± {sfge.std():.4f}")
    winner = result["summary"]["winner"]
    diff = result["summary"]["winner_lower_regret_pct"]
    line = f"  winner: {winner}  ({diff:.0f}% lower regret"
    if stats is not None and len(seeds) >= 3:
        try: line += f", Wilcoxon p={stats.wilcoxon(spo, sfge).pvalue:.3f}"
        except Exception: pass
    print(line + ")")

    if args.json_out:
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": dfl.dev,
            "experiments": [result],
        }
        out = write_json(args.json_out, payload)
        print(f"\nJSON written to {out}")


if __name__ == "__main__":
    main()
