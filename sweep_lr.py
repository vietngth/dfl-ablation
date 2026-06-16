#!/usr/bin/env python
"""
Learning-rate sweep: SPO+ vs SFGE.

The fixed-lr tables (run_all.py / README Tables 1-4) compare both methods at a single
learning rate (1e-2). SPO+'s convex surrogate is known to be *sensitive* to the step size,
so a single-lr comparison can flatter or punish either method. This script sweeps the Adam
learning rate over a grid for BOTH methods on each problem and reports:

  (1) regret vs lr for each method (the sensitivity curve), and
  (2) the BEST-TUNED head-to-head (each method at its own best lr) -- the fair comparison.

Both methods share the same deterministic data/model seed at every grid point (via
dfl's seed streams), so differences are due to the optimizer step size alone.

Run:
  .venv/bin/python sweep_lr.py --problems shortest_path,knapsack,tsp,portfolio --deg 4 \
      --lrs 1e-3,3e-3,1e-2,3e-2,1e-1 --seeds 0,1,2 --json-out experiment_results/sweep_lr.json
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

import dfl
from spo_vs_sfge import parse_seeds, run_experiment, write_json

CATEGORIES = {"shortest_path": "LP", "knapsack": "ILP", "tsp": "ILP", "portfolio": "SOCP"}


def sweep_problem(problem, lrs, seeds, deg, noise, n_train, n_test,
                  spo_epochs, sfge_epochs, sfge_samples, sfge_sigma, init):
    """Run both methods over the lr grid; return per-lr summaries + best-tuned picks."""
    points = []
    for lr in lrs:
        print(f"  [{problem} deg={deg}] lr={lr:g} ...", flush=True)
        exp = run_experiment(
            problem=problem, seeds=seeds, deg=deg, n_train=n_train, n_test=n_test,
            noise=noise, init=init, spo_epochs=spo_epochs, spo_lr=lr,
            sfge_epochs=sfge_epochs, sfge_samples=sfge_samples, sfge_sigma=sfge_sigma,
            sfge_lr=lr, warm_epochs=40, bias=False, check_solvers=False,
        )
        s = exp["summary"]
        points.append({"lr": lr,
                       "SPO+": {"mean": s["SPO+"]["mean"], "std": s["SPO+"]["std"]},
                       "SFGE": {"mean": s["SFGE"]["mean"], "std": s["SFGE"]["std"]}})
        print(f"      SPO+ {s['SPO+']['mean']:.4f}+/-{s['SPO+']['std']:.4f}   "
              f"SFGE {s['SFGE']['mean']:.4f}+/-{s['SFGE']['std']:.4f}", flush=True)
    best = {}
    for m in ("SPO+", "SFGE"):
        bp = min(points, key=lambda p: p[m]["mean"])
        best[m] = {"lr": bp["lr"], "mean": bp[m]["mean"], "std": bp[m]["std"]}
    spread = {m: (max(p[m]["mean"] for p in points) - min(p[m]["mean"] for p in points))
              for m in ("SPO+", "SFGE")}
    winner = "SPO+" if best["SPO+"]["mean"] < best["SFGE"]["mean"] else "SFGE"
    return {"problem": problem, "category": CATEGORIES[problem], "deg": deg, "noise": noise,
            "lrs": lrs, "points": points, "best_tuned": best,
            "regret_spread": spread, "winner_best_tuned": winner}


def md_table(res):
    """Markdown sensitivity table for one problem."""
    out = [f"### {res['problem']} ({res['category']}), deg={res['deg']}"
           + (f", noise={res['noise']}" if res['noise'] else ""), "",
           "| lr | SPO+ | SFGE |", "|---|---|---|"]
    bspo, bsfge = res["best_tuned"]["SPO+"]["lr"], res["best_tuned"]["SFGE"]["lr"]
    for p in res["points"]:
        spo = f"{p['SPO+']['mean']:.4f} ± {p['SPO+']['std']:.4f}"
        sfge = f"{p['SFGE']['mean']:.4f} ± {p['SFGE']['std']:.4f}"
        if p["lr"] == bspo: spo = f"**{spo}**"
        if p["lr"] == bsfge: sfge = f"**{sfge}**"
        out.append(f"| {p['lr']:g} | {spo} | {sfge} |")
    b = res["best_tuned"]
    out += ["", f"best-tuned: SPO+ {b['SPO+']['mean']:.4f} @ lr={b['SPO+']['lr']:g}  |  "
            f"SFGE {b['SFGE']['mean']:.4f} @ lr={b['SFGE']['lr']:g}  ->  "
            f"**{res['winner_best_tuned']}** "
            f"(SPO+ regret spread across lr = {res['regret_spread']['SPO+']:.4f}, "
            f"SFGE = {res['regret_spread']['SFGE']:.4f})", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="SPO+ vs SFGE learning-rate sweep.")
    ap.add_argument("--problems", default="shortest_path,knapsack,tsp,portfolio")
    ap.add_argument("--lrs", default="1e-3,3e-3,1e-2,3e-2,1e-1")
    ap.add_argument("--deg", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--init", default="cold", choices=["cold", "warm"])
    ap.add_argument("--spo-epochs", type=int, default=100)
    ap.add_argument("--sfge-epochs", type=int, default=120)
    ap.add_argument("--sfge-samples", type=int, default=8)
    ap.add_argument("--sfge-sigma", type=float, default=0.5)
    ap.add_argument("--json-out", default="experiment_results/sweep_lr.json")
    ap.add_argument("--md-out", default="experiment_results/sweep_lr.md")
    args = ap.parse_args()

    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    seeds = parse_seeds(args.seeds)
    print(f"LR SWEEP | problems={problems} deg={args.deg} lrs={lrs} seeds={seeds} "
          f"device={dfl.dev}\n", flush=True)

    results = []
    for problem in problems:
        results.append(sweep_problem(
            problem, lrs, seeds, args.deg, args.noise, args.n_train, args.n_test,
            args.spo_epochs, args.sfge_epochs, args.sfge_samples, args.sfge_sigma, args.init))

    payload = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "device": dfl.dev,
               "run_config": {"deg": args.deg, "noise": args.noise, "seeds": seeds, "lrs": lrs,
                              "n_train": args.n_train, "n_test": args.n_test, "init": args.init,
                              "spo_epochs": args.spo_epochs, "sfge_epochs": args.sfge_epochs,
                              "sfge_samples": args.sfge_samples, "sfge_sigma": args.sfge_sigma},
               "results": results}
    write_json(args.json_out, payload)

    md = ["# Learning-rate sweep: SPO+ vs SFGE",
          f"_deg={args.deg}, {len(seeds)} seeds, {args.init} init, n_train={args.n_train}, "
          f"normalized regret (lower better); bold = each method's best lr._", ""]
    md += [md_table(r) for r in results]
    from pathlib import Path
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text("\n".join(md) + "\n")

    print("\n" + "=" * 64)
    for r in results:
        b = r["best_tuned"]
        print(f"{r['problem']:>14} ({r['category']}): best-tuned SPO+ {b['SPO+']['mean']:.4f}"
              f"@{b['SPO+']['lr']:g}  SFGE {b['SFGE']['mean']:.4f}@{b['SFGE']['lr']:g}  "
              f"-> {r['winner_best_tuned']}", flush=True)
    print(f"\nJSON -> {args.json_out}\nMD   -> {args.md_out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
