#!/usr/bin/env python
"""
Generate all result tables and write a machine-readable JSON artifact.
"""
import argparse
from datetime import datetime, timezone

import dfl
from spo_vs_sfge import parse_seeds, run_experiment, write_json


CATEGORIES = {"shortest_path": "LP", "knapsack": "ILP", "tsp": "ILP", "portfolio": "SOCP"}


def add_common_args(ap):
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--init", default="cold", choices=["cold", "warm"])
    ap.add_argument("--bias", action="store_true", help="use a bias term in the shared linear predictor")
    ap.add_argument("--warm-epochs", type=int, default=40)
    ap.add_argument("--spo-epochs", type=int, default=100)
    ap.add_argument("--spo-lr", type=float, default=1e-2)
    ap.add_argument("--sfge-epochs", type=int, default=120)
    ap.add_argument("--sfge-samples", type=int, default=8)
    ap.add_argument("--sfge-sigma", type=float, default=0.5)
    ap.add_argument("--sfge-lr", type=float, default=1e-2)
    ap.add_argument("--check-solvers", action="store_true")
    ap.add_argument("--json-out", default="experiment_results/results.json")


def fmt(summary, method):
    return f"{summary[method]['mean']:>7.4f} +/- {summary[method]['std']:<6.4f}"


def main():
    ap = argparse.ArgumentParser(description="Generate robust SPO+ vs SFGE result tables.")
    add_common_args(ap)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    cache = {}

    def evaluate(problem, deg, noise=0.0):
        key = (
            problem, deg, noise, args.n_train, args.n_test, args.init, args.bias,
            tuple(seeds), args.warm_epochs, args.spo_epochs, args.spo_lr,
            args.sfge_epochs, args.sfge_samples, args.sfge_sigma, args.sfge_lr,
            args.check_solvers,
        )
        if key not in cache:
            print(f"\n[run] problem={problem} deg={deg} noise={noise} seeds={seeds}", flush=True)
            cache[key] = run_experiment(
                problem=problem,
                seeds=seeds,
                deg=deg,
                n_train=args.n_train,
                n_test=args.n_test,
                noise=noise,
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
        return cache[key]

    tables = []

    print("=== TABLE 1: per-problem (deg=4) ===", flush=True)
    print(f"{'problem':>14} {'category':>10} | {'SPO+':>16} {'SFGE':>16} | winner", flush=True)
    rows = []
    for problem in ["shortest_path", "knapsack", "tsp", "portfolio"]:
        exp = evaluate(problem, 4)
        summary = exp["summary"]
        rows.append({"problem": problem, "category": CATEGORIES[problem], "experiment": exp})
        print(f"{problem:>14} {CATEGORIES[problem]:>10} | {fmt(summary, 'SPO+')} "
              f"{fmt(summary, 'SFGE')} | {summary['winner']}", flush=True)
    tables.append({"name": "per_problem_deg4", "rows": rows})

    print("\n=== TABLE 2: misspecification sweep, knapsack (ILP), deg in {1,2,4,6,8} ===", flush=True)
    print(f"{'deg':>4} | {'SPO+':>16} {'SFGE':>16} | winner", flush=True)
    rows = []
    for deg in [1, 2, 4, 6, 8]:
        exp = evaluate("knapsack", deg)
        summary = exp["summary"]
        rows.append({"deg": deg, "experiment": exp})
        print(f"{deg:>4} | {fmt(summary, 'SPO+')} {fmt(summary, 'SFGE')} | {summary['winner']}",
              flush=True)
    tables.append({"name": "knapsack_misspecification", "rows": rows})

    print("\n=== TABLE 3: misspecification sweep, shortest_path (LP), deg in {1,2,4,6,8} ===", flush=True)
    print(f"{'deg':>4} | {'SPO+':>16} {'SFGE':>16} | winner", flush=True)
    rows = []
    for deg in [1, 2, 4, 6, 8]:
        exp = evaluate("shortest_path", deg)
        summary = exp["summary"]
        rows.append({"deg": deg, "experiment": exp})
        print(f"{deg:>4} | {fmt(summary, 'SPO+')} {fmt(summary, 'SFGE')} | {summary['winner']}",
              flush=True)
    tables.append({"name": "shortest_path_misspecification", "rows": rows})

    print("\n=== TABLE 4: noise sweep, shortest_path (LP), deg=6, noise in {0,0.25,0.5,1.0} ===", flush=True)
    print(f"{'noise':>6} | {'SPO+':>16} {'SFGE':>16} | winner", flush=True)
    rows = []
    for noise in [0.0, 0.25, 0.5, 1.0]:
        exp = evaluate("shortest_path", 6, noise=noise)
        summary = exp["summary"]
        rows.append({"noise": noise, "experiment": exp})
        print(f"{noise:>6} | {fmt(summary, 'SPO+')} {fmt(summary, 'SFGE')} | {summary['winner']}",
              flush=True)
    tables.append({"name": "shortest_path_noise_deg6", "rows": rows})

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": dfl.dev,
        "run_config": {
            "seeds": seeds,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "init": args.init,
            "bias": args.bias,
            "warm_epochs": args.warm_epochs,
            "spo_epochs": args.spo_epochs,
            "spo_lr": args.spo_lr,
            "sfge_epochs": args.sfge_epochs,
            "sfge_samples": args.sfge_samples,
            "sfge_sigma": args.sfge_sigma,
            "sfge_lr": args.sfge_lr,
            "check_solvers": args.check_solvers,
        },
        "tables": tables,
        "unique_experiments": list(cache.values()),
    }
    out = write_json(args.json_out, payload)
    print(f"\nJSON written to {out}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
