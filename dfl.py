"""
SPO+ vs SFGE comparison.

  * batched GPU/CPU forward solvers  (shortest-path DAG, 0/1 knapsack DP, Markowitz SOCP);
  * four PyEPO-backed problem set-ups (shortest_path, knapsack, tsp, portfolio);
  * a two-stage MSE trainer (used only for the optional warm start);
  * the SFGE trainer (score-function / REINFORCE on the realized decision loss).

Re-used from PyEPO: data gen, opt models, normalized-regret metrics
Self-implementation: batched solvers + SFGE estimator 
(batch solvers just mean solving the problem in parallel)
"""
import itertools
import random

import numpy as np
import torch
import torch.nn as nn

from pyepo.data import shortestpath, knapsack, tsp, portfolio
from pyepo.model.grb import shortestPathModel, knapsackModel, tspMTZModel, portfolioModel
from pyepo.data.dataset import optDataset
from torch.utils.data import DataLoader

# Fall back to CPU automatically so this runs on a peer's machine without a GPU.
dev = "cuda" if torch.cuda.is_available() else "cpu"
PF = 5   # number of predictive features (Elmachtoub-Grigas generator default here)
_SEED_MOD = 2**31 - 1


def stream_seed(seed, stream):
    """Derive a stable positive seed for one stochastic stream."""
    offsets = {
        "run": 11,
        "model": 101,
        "warm": 211,
        "warm_loader": 307,
        "spo": 401,
        "spo_loader": 503,
        "sfge": 601,
    }
    if stream not in offsets:
        raise ValueError(f"unknown seed stream {stream!r}")
    return (int(seed) * 1_000_003 + offsets[stream]) % _SEED_MOD


def seed_everything(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiment runs."""
    seed = int(seed) % _SEED_MOD
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed


def clone_state_dict(model_or_state):
    """Clone a model/state dict so trainers cannot mutate the shared baseline."""
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return {k: v.detach().clone() for k, v in state.items()}


def _fork_rng():
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    return torch.random.fork_rng(devices=devices)


def _cpu_generator(seed):
    gen = torch.Generator()
    gen.manual_seed(int(seed) % _SEED_MOD)
    return gen


def _device_generator(seed):
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed) % _SEED_MOD)
    return gen


# Batched forward solvers (shared, black-box argmin oracles)
def build_dag_solver(arcs, num_nodes, source, sink):
    """Exact shortest path on a DAG. arcs: list[(u, v)] in cost-vector order."""
    out_by_node = [[] for _ in range(num_nodes)]
    for e, (u, v) in enumerate(arcs):
        out_by_node[u].append((v, e))
    E = len(arcs)

    def solve_batch(c):                                  # c (M,E) -> w (M,E) one-hot path
        M = c.shape[0]; dv = c.device; INF = float("inf")
        dist = torch.full((M, num_nodes), INF, device=dv); dist[:, source] = 0.0
        pe = torch.full((M, num_nodes), -1, dtype=torch.long, device=dv)
        pn = torch.full((M, num_nodes), -1, dtype=torch.long, device=dv)
        for u in range(num_nodes):
            for (v, e) in out_by_node[u]:
                nd = dist[:, u] + c[:, e]; better = nd < dist[:, v]
                dist[:, v] = torch.where(better, nd, dist[:, v])
                pe[:, v] = torch.where(better, torch.full_like(pe[:, v], e), pe[:, v])
                pn[:, v] = torch.where(better, torch.full_like(pn[:, v], u), pn[:, v])
        w = torch.zeros((M, E), device=dv)
        cur = torch.full((M,), sink, dtype=torch.long, device=dv)
        midx = torch.arange(M, device=dv)
        for _ in range(num_nodes):
            active = cur != source
            if not active.any(): break
            e = pe[midx, cur]
            w[midx[active], e[active]] = 1.0
            cur = torch.where(active, pn[midx, cur], cur)
        return w
    return solve_batch


def grid_arcs(H, W):
    """East/south arcs of an HxW grid, node index r*W+c, source 0, sink HW-1."""
    arcs = []
    for r in range(H):
        for c in range(W):
            n = r * W + c
            if c + 1 < W: arcs.append((n, n + 1))
    for r in range(H):
        for c in range(W):
            n = r * W + c
            if r + 1 < H: arcs.append((n, n + W))
    return arcs, H * W, 0, H * W - 1


def knap1_dp(v, w, C, want_sel=True):
    """0/1 single-constraint knapsack (exact DP) with selection backtrack.
    v,w: (M,n) values & INTEGER weights (>=1). C int. -> best (M,), sel (M,n)."""
    M, n = v.shape; dv = v.device
    NEG = torch.finfo(v.dtype).min / 4
    dp = torch.zeros(M, C + 1, device=dv, dtype=v.dtype)
    keep = torch.zeros(M, n, C + 1, dtype=torch.bool, device=dv) if want_sel else None
    rng = torch.arange(C + 1, device=dv)
    for i in range(n):
        wi = w[:, i].long(); vi = v[:, i]
        src = rng[None, :] - wi[:, None]; valid = src >= 0
        cand = torch.where(valid, torch.gather(dp, 1, src.clamp(min=0)) + vi[:, None],
                           torch.full_like(dp, NEG))
        take = cand > dp
        if want_sel: keep[:, i, :] = take
        dp = torch.where(take, cand, dp)
    best = dp[:, C]
    if not want_sel:
        return best, None
    sel = torch.zeros(M, n, dtype=torch.bool, device=dv)
    c = torch.full((M,), C, dtype=torch.long, device=dv); midx = torch.arange(M, device=dv)
    for i in range(n - 1, -1, -1):
        t = keep[midx, i, c]; sel[:, i] = t
        c = torch.where(t, c - w[:, i].long(), c)
    return best, sel


def proj_simplex(v):
    """Euclidean projection of each row of v onto {w>=0, sum w = 1}."""
    n = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = u.cumsum(-1) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    rho = ((u - css / ind) > 0).sum(-1, keepdim=True).clamp(min=1)
    theta = css.gather(-1, rho - 1) / rho
    return (v - theta).clamp(min=0)


def solve_portfolio_socp(r, Sigma, rho, bis=24, pg=40, hi0=1e7):
    """Variance-constrained Markowitz: max r^T w s.t. sum w = 1, w >= 0,
    w^T Sigma w <= rho. Batched Lagrangian bisection (verified == Gurobi)."""
    M, n = r.shape
    lmax = float(torch.linalg.eigvalsh(Sigma)[-1])
    rscale = r.abs().mean() + 1e-9
    lo = torch.zeros(M, 1, device=r.device); hi = torch.full((M, 1), hi0, device=r.device)
    w = torch.full((M, n), 1.0 / n, device=r.device)
    for _ in range(bis):
        mid = (lo + hi) / 2
        lr = (1.0 / (2.0 * mid * lmax + rscale)).clamp(max=1.0)
        for _ in range(pg):
            w = proj_simplex(w + lr * (r - 2 * mid * (w @ Sigma)))
        risky = (w @ Sigma * w).sum(-1, keepdim=True) > rho
        lo = torch.where(risky, mid, lo); hi = torch.where(risky, hi, mid)
    return w


# Problem set-ups: each returns a cfg dict consumed by the trainers
def _common(om, x, c, dim, ps_solve, objective, standardize, seed, n_train, bias=False):
    """Pack a PyEPO optModel + data into the shared cfg dict (predictor factory,
    DataLoaders, standardized cost tensor for the SFGE estimator, objective sign)."""
    xtr, ctr, xte, cte = x[:n_train], c[:n_train], x[n_train:], c[n_train:]
    ds_tr = optDataset(om, xtr, ctr); ds_te = optDataset(om, xte, cte)
    Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)
    shift = Ctr.mean() if standardize == "affine" else 0.0

    def make(model_seed=None, state_dict=None):
        model_seed = stream_seed(seed, "model") if model_seed is None else model_seed
        with _fork_rng():
            torch.manual_seed(int(model_seed) % _SEED_MOD)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(model_seed) % _SEED_MOD)
            model = nn.Linear(PF, dim, bias=bias).to(dev)
        if state_dict is not None:
            model.load_state_dict({k: v.detach().to(dev).clone() for k, v in state_dict.items()})
        return model

    def make_train_loader(stream="spo_loader"):
        return DataLoader(
            ds_tr,
            batch_size=128,
            shuffle=True,
            generator=_cpu_generator(stream_seed(seed, stream)),
        )

    def make_test_loader():
        return DataLoader(ds_te, batch_size=256)

    initial_state = clone_state_dict(make(model_seed=stream_seed(seed, "model")))
    cfg = dict(om=om, dim=dim, seed=seed, ps_solve=ps_solve,
               bias=bias, make=make, initial_state=initial_state,
               make_train_loader=make_train_loader, make_test_loader=make_test_loader,
               ld_tr=make_train_loader("spo_loader"),
               ld_te=make_test_loader(), ds_tr=ds_tr, ds_te=ds_te,
               Xtr=torch.tensor(xtr, dtype=torch.float32, device=dev),
               Cs=(Ctr - shift) / Ctr.std(),
               sign=(1.0 if objective == "min" else -1.0))
    return cfg


def setup_sp(seed, deg, n_train=400, noise=0.0, n_test=1000, H=5, W=5, bias=False):
    """5x5 grid shortest path (LP). `noise` -> generator's noise_width (multiplicative)."""
    om = shortestPathModel((H, W)); arcs = list(om.arcs)
    sb = build_dag_solver(arcs, H * W, 0, H * W - 1)
    x, c = shortestpath.genData(n_train + n_test, PF, (H, W), deg=deg, noise_width=noise, seed=seed)
    return _common(om, x, c, len(arcs), sb, "min", "affine", seed, n_train, bias=bias)


def setup_knap(seed, deg, n_train=400, noise=0.0, n_test=1000, NIT=16, bias=False):
    """0/1 knapsack, 16 items, single capacity (ILP). `noise` -> noise_width."""
    W_np, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)
    weights = W_np[0].astype(int); CAP = int(weights.sum() * 0.5)
    om = knapsackModel(weights=W_np.astype(int), capacity=[CAP])
    Wt = torch.tensor(weights, dtype=torch.float32, device=dev)
    sb = lambda v: knap1_dp(v, Wt.expand(v.shape[0], -1), CAP)[1].float()
    _, x, c = knapsack.genData(n_train + n_test, PF, NIT, dim=1, deg=deg, noise_width=noise, seed=seed)
    return _common(om, x, c, NIT, sb, "max", "scale", seed, n_train, bias=bias)


def setup_tsp(seed, deg, n_train=280, noise=0.0, n_test=120, N=8, bias=False):
    """Symmetric TSP, 8 nodes (ILP). Exact brute-force over tours == Gurobi.
    The PyEPO TSP generator exposes no noise hook, so `noise` is ignored."""
    om = tspMTZModel(num_nodes=N); ei = {e: i for i, e in enumerate(om.edges)}; E = om.num_cost
    rows = []
    for perm in itertools.permutations(range(1, N)):
        if perm[0] > perm[-1]: continue
        cyc = [0] + list(perm); v = torch.zeros(E)
        for a, b in zip(cyc, cyc[1:] + [0]): v[ei[(min(a, b), max(a, b))]] = 1.0
        rows.append(v)
    T = torch.stack(rows).to(dev)
    sb = lambda c: T[(c @ T.T).argmin(1)]
    x, c = tsp.genData(n_train + n_test, PF, N, deg=deg, noise_width=0, seed=seed)
    return _common(om, x, c, E, sb, "min", "affine", seed, n_train, bias=bias)


def setup_port(seed, deg, n_train=200, noise=0.0, n_test=400, NA=20, bias=False):
    """Variance-constrained Markowitz portfolio, 20 assets (SOCP). `noise` ignored."""
    cov, x, r = portfolio.genData(n_train + n_test, PF, NA, deg=deg, noise_level=1, seed=seed)
    om = portfolioModel(num_assets=NA, covariance=cov, gamma=2.25); rho = float(om.risk_level)
    Sig = torch.tensor(cov, dtype=torch.float32, device=dev)
    sb = lambda rhat: solve_portfolio_socp(rhat, Sig, rho, bis=16, pg=25)
    return _common(om, x, r, NA, sb, "max", "scale", seed, n_train, bias=bias)


SETUPS = {"shortest_path": setup_sp, "knapsack": setup_knap, "tsp": setup_tsp, "portfolio": setup_port}


# Trainers
def _adam(model, lr=1e-2):
    return torch.optim.Adam(model.parameters(), lr)


def _start_state(cfg, state_dict=None, warm=None):
    if state_dict is not None:
        return clone_state_dict(state_dict)
    if warm is not None:
        return clone_state_dict(warm)
    if cfg.get("warm") is not None:
        return clone_state_dict(cfg["warm"])
    return clone_state_dict(cfg["initial_state"])


def train_two_stage(cfg, epochs=40, lr=1e-2, initial_state=None):
    """Plain MSE predictor. Used only as the (optional) shared warm start."""
    seed_everything(stream_seed(cfg["seed"], "warm"))
    m = cfg["make"](state_dict=_start_state(cfg, initial_state)); opt = _adam(m, lr)
    loader = cfg["make_train_loader"]("warm_loader")
    for _ in range(epochs):
        for xb, cb, wb, zb in loader:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            opt.zero_grad(); ((m(xb) - cb) ** 2).mean().backward(); opt.step()
    return m


def train_sfge(cfg, epochs=120, n_samples=8, sigma=0.5, lr=1e-2, initial_state=None):
    """SFGE — Score-Function Gradient Estimation (Silvestri et al., JAIR 2024,
    arXiv:2307.05213). Places a Gaussian over the predicted parameters, samples,
    solves the (black-box) forward problem, and uses a score-function / REINFORCE
    estimator on the realized decision loss with a per-batch baseline for variance
    reduction. Gradient-free w.r.t. the solver: it never differentiates through it."""
    seed_everything(stream_seed(cfg["seed"], "sfge"))
    m = cfg["make"](state_dict=_start_state(cfg, initial_state))
    opt = _adam(m, lr)
    gen = _device_generator(stream_seed(cfg["seed"], "sfge"))
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    for _ in range(epochs):
        pred = m(X)                                                  # (B, D), differentiable in theta
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev, generator=gen)
            chat = pred.unsqueeze(0) + sigma * eps                   # (S, B, D) sampled predictions
            S, B, D = chat.shape
            w = solve(chat.reshape(S * B, D)).reshape(S, B, D)
            r = sgn * (w * Cs.unsqueeze(0)).sum(-1)                  # (S,B) per-sample loss (minimize)
            adv = r - r.mean(0, keepdim=True)                        # baseline-subtracted advantage
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)   # grad flows via pred
        surrogate = (adv * logp).mean()                             # REINFORCE: d/dtheta = E[adv * dlogp]
        opt.zero_grad(); surrogate.backward(); opt.step()
    return m


def train_spoplus(cfg, warm=None, epochs=100, lr=1e-2, initial_state=None):
    """SPO+ reused straight from PyEPO (pyepo.func.SPOPlus), the convex surrogate
    of Elmachtoub & Grigas (Management Science 2022)."""
    import pyepo.func as F
    seed_everything(stream_seed(cfg["seed"], "spo"))
    m = cfg["make"](state_dict=_start_state(cfg, initial_state, warm))
    opt = _adam(m, lr)
    spop = F.SPOPlus(cfg["om"])
    loader = cfg["make_train_loader"]("spo_loader")
    for _ in range(epochs):
        for xb, cb, wb, zb in loader:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            opt.zero_grad(); spop(m(xb), cb, wb, zb).mean().backward(); opt.step()
    return m


def solver_sanity(cfg, n=16):
    """Compare the local SFGE forward solver against cached PyEPO optima."""
    k = min(int(n), len(cfg["ds_tr"]))
    costs = cfg["ds_tr"].costs[:k].to(dev)
    ref = cfg["ds_tr"].sols[:k].to(dev)
    with torch.no_grad():
        got = cfg["ps_solve"](costs).to(dev)
    obj_got = (got * costs).sum(-1)
    obj_ref = (ref * costs).sum(-1)
    gap = obj_got - obj_ref if cfg["sign"] > 0 else obj_ref - obj_got
    return {
        "n": k,
        "max_directional_gap": float(gap.max().detach().cpu()),
        "max_abs_obj_gap": float((obj_got - obj_ref).abs().max().detach().cpu()),
        "mean_abs_obj_gap": float((obj_got - obj_ref).abs().mean().detach().cpu()),
    }
