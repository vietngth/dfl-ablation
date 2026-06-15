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
import numpy as np

from pyepo import metric
import dfl


def build_cfg(problem, seed, deg, n_train, n_test, noise):
    """Build the harness cfg dict for one of the four problems."""
    if problem not in dfl.SETUPS:
        raise ValueError(f"unknown problem {problem!r}")
    if noise and problem in ("tsp", "portfolio"):
        print(f"  [note] --noise is ignored for {problem} (no noise hook in its generator)")
    return dfl.SETUPS[problem](seed, deg, n_train=n_train, noise=noise, n_test=n_test)


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
    ap.add_argument("--spo-epochs", type=int, default=100)
    ap.add_argument("--spo-lr", type=float, default=1e-2)
    ap.add_argument("--sfge-epochs", type=int, default=120)
    ap.add_argument("--sfge-samples", type=int, default=8)
    ap.add_argument("--sfge-sigma", type=float, default=0.5)
    ap.add_argument("--sfge-lr", type=float, default=1e-2)
    args = ap.parse_args()

    try:
        from scipy import stats
    except Exception:
        stats = None

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"\nSPO+ vs SFGE | problem={args.problem} deg={args.deg} n_train={args.n_train} "
          f"noise={args.noise} init={args.init} seeds={seeds} device={dfl.dev}")
    print(f"SPO+: PyEPO SPOPlus ({args.spo_epochs} ep) | SFGE: score-function "
          f"({args.sfge_epochs} ep, {args.sfge_samples} samples, sigma={args.sfge_sigma}) | "
          f"metric: PyEPO normalized regret (Gurobi)\n", flush=True)

    spo, sfge = [], []
    for seed in seeds:
        cfg = build_cfg(args.problem, seed, args.deg, args.n_train, args.n_test, args.noise)
        warm = dfl.train_two_stage(cfg) if args.init == "warm" else None
        cfg["warm"] = warm                                  # SAME init for both methods
        r_spo = metric.regret(dfl.train_spoplus(cfg, warm, args.spo_epochs, args.spo_lr),
                              cfg["om"], cfg["ld_te"])
        r_sfge = metric.regret(dfl.train_sfge(cfg, epochs=args.sfge_epochs, n_samples=args.sfge_samples,
                                              sigma=args.sfge_sigma, lr=args.sfge_lr),
                               cfg["om"], cfg["ld_te"])
        spo.append(r_spo); sfge.append(r_sfge)
        print(f"  seed {seed}:  SPO+ = {r_spo:.4f}   SFGE = {r_sfge:.4f}", flush=True)

    spo, sfge = np.array(spo), np.array(sfge)
    print("\n--- summary (normalized regret, lower is better) ---")
    print(f"  SPO+ : {spo.mean():.4f} ± {spo.std():.4f}")
    print(f"  SFGE : {sfge.mean():.4f} ± {sfge.std():.4f}")
    winner = "SPO+" if spo.mean() < sfge.mean() else "SFGE"
    diff = abs(spo.mean() - sfge.mean()) / max(spo.mean(), 1e-9) * 100
    line = f"  winner: {winner}  ({diff:.0f}% lower regret"
    if stats is not None and len(seeds) >= 3:
        try: line += f", Wilcoxon p={stats.wilcoxon(spo, sfge).pvalue:.3f}"
        except Exception: pass
    print(line + ")")


if __name__ == "__main__":
    main()
