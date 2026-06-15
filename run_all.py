"""
Generate all README tables: per-problem SPO+ vs SFGE, and a misspecification scenario.
"""
import numpy as np
from pyepo import metric
import dfl

SEEDS = [0, 1, 2]

def evaluate(problem, deg, noise=0.0, n_train=400):
    spo, sfge = [], []
    for s in SEEDS:
        cfg = dfl.SETUPS[problem](s, deg, n_train=n_train, noise=noise)
        cfg["warm"] = None
        spo.append(metric.regret(dfl.train_spoplus(cfg), cfg["om"], cfg["ld_te"]))
        sfge.append(metric.regret(dfl.train_sfge(cfg), cfg["om"], cfg["ld_te"]))
    return np.mean(spo), np.std(spo), np.mean(sfge), np.std(sfge)

print("=== TABLE 1: per-problem (deg=4, cold init, 3 seeds) ===", flush=True)
print(f"{'problem':>14} {'category':>10} | {'SPO+':>16} {'SFGE':>16} | winner", flush=True)
cats = {"shortest_path": "LP", "knapsack": "ILP", "tsp": "ILP", "portfolio": "SOCP"}
for p in ["shortest_path", "knapsack", "tsp", "portfolio"]:
    sm, ss, fm, fs = evaluate(p, 4)
    win = "SPO+" if sm < fm else "SFGE"
    print(f"{p:>14} {cats[p]:>10} | {sm:>7.4f} +/- {ss:<6.4f} {fm:>7.4f} +/- {fs:<6.4f} | {win}", flush=True)

print("\n=== TABLE 2: misspecification sweep, knapsack (ILP), deg in {1,2,4,6,8} ===", flush=True)
print(f"{'deg':>4} | {'SPO+':>16} {'SFGE':>16} | winner", flush=True)
for d in [1, 2, 4, 6, 8]:
    sm, ss, fm, fs = evaluate("knapsack", d)
    win = "SPO+" if sm < fm else "SFGE"
    print(f"{d:>4} | {sm:>7.4f} +/- {ss:<6.4f} {fm:>7.4f} +/- {fs:<6.4f} | {win}", flush=True)

print("\n=== TABLE 3: misspecification sweep, shortest_path (LP), deg in {1,2,4,6,8} ===", flush=True)
print(f"{'deg':>4} | {'SPO+':>16} {'SFGE':>16} | winner", flush=True)
for d in [1, 2, 4, 6, 8]:
    sm, ss, fm, fs = evaluate("shortest_path", d)
    win = "SPO+" if sm < fm else "SFGE"
    print(f"{d:>4} | {sm:>7.4f} +/- {ss:<6.4f} {fm:>7.4f} +/- {fs:<6.4f} | {win}", flush=True)

print("\n=== TABLE 4: noise sweep, shortest_path (LP), deg=6, noise in {0,0.25,0.5,1.0} ===", flush=True)
print(f"{'noise':>6} | {'SPO+':>16} {'SFGE':>16} | winner", flush=True)
for nz in [0.0, 0.25, 0.5, 1.0]:
    sm, ss, fm, fs = evaluate("shortest_path", 6, noise=nz)
    win = "SPO+" if sm < fm else "SFGE"
    print(f"{nz:>6} | {sm:>7.4f} +/- {ss:<6.4f} {fm:>7.4f} +/- {fs:<6.4f} | {win}", flush=True)

print("\nDONE", flush=True)
