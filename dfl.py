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
import torch
import torch.nn as nn

from pyepo.data import shortestpath, knapsack, tsp, portfolio
from pyepo.model.grb import shortestPathModel, knapsackModel, tspMTZModel, portfolioModel
from pyepo.data.dataset import optDataset
from torch.utils.data import DataLoader

# Fall back to CPU automatically so this runs on a peer's machine without a GPU.
dev = "cuda" if torch.cuda.is_available() else "cpu"
PF = 5   # number of predictive features (Elmachtoub-Grigas generator default here)


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
def _common(om, x, c, dim, ps_solve, objective, standardize, seed, n_train):
    """Pack a PyEPO optModel + data into the shared cfg dict (predictor factory,
    DataLoaders, standardized cost tensor for the SFGE estimator, objective sign)."""
    xtr, ctr, xte, cte = x[:n_train], c[:n_train], x[n_train:], c[n_train:]
    ds_tr = optDataset(om, xtr, ctr); ds_te = optDataset(om, xte, cte)
    Ctr = torch.tensor(ctr, dtype=torch.float32, device=dev)
    shift = Ctr.mean() if standardize == "affine" else 0.0
    cfg = dict(om=om, dim=dim, seed=seed, ps_solve=ps_solve,
               make=lambda: nn.Linear(PF, dim, bias=False).to(dev),
               ld_tr=DataLoader(ds_tr, batch_size=128, shuffle=True),
               ld_te=DataLoader(ds_te, batch_size=256), ds_tr=ds_tr,
               Xtr=torch.tensor(xtr, dtype=torch.float32, device=dev),
               Cs=(Ctr - shift) / Ctr.std(),
               sign=(1.0 if objective == "min" else -1.0))
    return cfg


def setup_sp(seed, deg, n_train=400, noise=0.0, n_test=1000, H=5, W=5):
    """5x5 grid shortest path (LP). `noise` -> generator's noise_width (multiplicative)."""
    om = shortestPathModel((H, W)); arcs = list(om.arcs)
    sb = build_dag_solver(arcs, H * W, 0, H * W - 1)
    x, c = shortestpath.genData(n_train + n_test, PF, (H, W), deg=deg, noise_width=noise, seed=seed)
    return _common(om, x, c, len(arcs), sb, "min", "affine", seed, n_train)


def setup_knap(seed, deg, n_train=400, noise=0.0, n_test=1000, NIT=16):
    """0/1 knapsack, 16 items, single capacity (ILP). `noise` -> noise_width."""
    W_np, _, _ = knapsack.genData(2, PF, NIT, dim=1, deg=1, seed=1)
    weights = W_np[0].astype(int); CAP = int(weights.sum() * 0.5)
    om = knapsackModel(weights=W_np.astype(int), capacity=[CAP])
    Wt = torch.tensor(weights, dtype=torch.float32, device=dev)
    sb = lambda v: knap1_dp(v, Wt.expand(v.shape[0], -1), CAP)[1].float()
    _, x, c = knapsack.genData(n_train + n_test, PF, NIT, dim=1, deg=deg, noise_width=noise, seed=seed)
    return _common(om, x, c, NIT, sb, "max", "scale", seed, n_train)


def setup_tsp(seed, deg, n_train=280, noise=0.0, n_test=120, N=8):
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
    return _common(om, x, c, E, sb, "min", "affine", seed, n_train)


def setup_port(seed, deg, n_train=200, noise=0.0, n_test=400, NA=20):
    """Variance-constrained Markowitz portfolio, 20 assets (SOCP). `noise` ignored."""
    cov, x, r = portfolio.genData(n_train + n_test, PF, NA, deg=deg, noise_level=1, seed=seed)
    om = portfolioModel(num_assets=NA, covariance=cov, gamma=2.25); rho = float(om.risk_level)
    Sig = torch.tensor(cov, dtype=torch.float32, device=dev)
    sb = lambda rhat: solve_portfolio_socp(rhat, Sig, rho, bis=16, pg=25)
    return _common(om, x, r, NA, sb, "max", "scale", seed, n_train)


SETUPS = {"shortest_path": setup_sp, "knapsack": setup_knap, "tsp": setup_tsp, "portfolio": setup_port}


# Trainers
def _adam(model, lr=1e-2):
    return torch.optim.Adam(model.parameters(), lr)


def train_two_stage(cfg, epochs=40):
    """Plain MSE predictor. Used only as the (optional) shared warm start."""
    m = cfg["make"](); opt = _adam(m)
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb = xb.float().to(dev), cb.float().to(dev)
            opt.zero_grad(); ((m(xb) - cb) ** 2).mean().backward(); opt.step()
    return m


def train_sfge(cfg, epochs=120, n_samples=8, sigma=0.5, lr=1e-2):
    """SFGE — Score-Function Gradient Estimation (Silvestri et al., JAIR 2024,
    arXiv:2307.05213). Places a Gaussian over the predicted parameters, samples,
    solves the (black-box) forward problem, and uses a score-function / REINFORCE
    estimator on the realized decision loss with a per-batch baseline for variance
    reduction. Gradient-free w.r.t. the solver: it never differentiates through it."""
    m = cfg["make"]()
    if cfg.get("warm") is not None:
        with torch.no_grad(): m.weight.copy_(cfg["warm"].weight)
    opt = _adam(m, lr)
    X, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
    for _ in range(epochs):
        pred = m(X)                                                  # (B, D), differentiable in theta
        with torch.no_grad():
            eps = torch.randn(n_samples, *pred.shape, device=dev)
            chat = pred.unsqueeze(0) + sigma * eps                   # (S, B, D) sampled predictions
            S, B, D = chat.shape
            w = solve(chat.reshape(S * B, D)).reshape(S, B, D)
            r = sgn * (w * Cs.unsqueeze(0)).sum(-1)                  # (S,B) per-sample loss (minimize)
            adv = r - r.mean(0, keepdim=True)                        # baseline-subtracted advantage
        logp = -((chat - pred.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)   # grad flows via pred
        surrogate = (adv * logp).mean()                             # REINFORCE: d/dtheta = E[adv * dlogp]
        opt.zero_grad(); surrogate.backward(); opt.step()
    return m


def train_spoplus(cfg, warm=None, epochs=100, lr=1e-2):
    """SPO+ reused straight from PyEPO (pyepo.func.SPOPlus), the convex surrogate
    of Elmachtoub & Grigas (Management Science 2022)."""
    import pyepo.func as F
    m = cfg["make"]()
    if warm is not None:
        with torch.no_grad(): m.weight.copy_(warm.weight)
    opt = _adam(m, lr)
    spop = F.SPOPlus(cfg["om"])
    for _ in range(epochs):
        for xb, cb, wb, zb in cfg["ld_tr"]:
            xb, cb, wb, zb = [t.float().to(dev) for t in (xb, cb, wb, zb)]
            opt.zero_grad(); spop(m(xb), cb, wb, zb).mean().backward(); opt.step()
    return m
