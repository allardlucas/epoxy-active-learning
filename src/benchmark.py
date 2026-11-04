"""Benchmark acquisition strategies on simulated labs.

A simulated lab ("oracle") gives the true strength of every recipe; a measurement
adds Gaussian noise (sd = 2 MPa, the typical replicate std in the paper).
To avoid favouring one model family, four oracles are used:
  gp51  - GP fitted on all 51 published measurements
  rf51  - Random Forest fitted on all 51 measurements
  xgb51 - XGBoost fitted on all 51 measurements (paper's cycle-3 hyperparameters)
  gp32  - GP fitted on the 32 initial design points only (never saw AL/BO points)
Each campaign starts from an initial design (the full 32-point Graeco-Latin design,
or a 10-point space-filling subset of it) and spends 15 experiments in 3 batches of 5,
like the paper's 3 AL cycles.
Search space: 4 x 4 molecular weights x r in [0.75, 1.5] (step 0.05)
x T_cure in [90, 210] (step 10) = 3,328 recipes.
Metric: true strength of the best recipe tested so far (initial + new),
and whether a new proposal beats the best initial recipe."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools, numpy as np, pandas as pd
from scipy.stats import norm, wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["MW_epoxy", "MW_curing_agent", "r", "T_cure"]
MWE, MWC = [370, 1650, 2900, 3800], [230, 400, 2000, 4000]
GRID = np.array(list(itertools.product(MWE, MWC, np.round(np.arange(0.75, 1.501, 0.05), 2),
                                       np.arange(90, 211, 10))), dtype=float)
LO, HI = GRID.min(0), GRID.max(0)
STRATEGIES = ["random", "greedy_xgb", "greedy_rf", "gp_mean", "gp_ei", "gp_ucb"]
ORACLES = ["gp51", "rf51", "xgb51", "gp32"]
NOISE_SD = 2.0

def scale(X):  # min-max to [0, 1], log for molecular weights
    X = np.asarray(X, float).copy(); X[:, :2] = np.log(X[:, :2])
    lo, hi = LO.copy(), HI.copy(); lo[:2], hi[:2] = np.log(lo[:2]), np.log(hi[:2])
    return (X - lo) / (hi - lo)

def gp(noise=0.1, fixed=False):
    w = WhiteKernel(noise, noise_level_bounds="fixed") if fixed else WhiteKernel(noise)
    k = C(1.0) * Matern(length_scale=[0.3] * 4, nu=2.5) + w
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0)

def load():
    return pd.read_csv(ROOT / "data" / "epoxy_adhesive_dataset.csv")

class Lab:
    """Simulated lab: `truth` = true strength of each grid recipe, `measure` adds noise."""
    def __init__(self, df, oracle="gp51", noise=NOISE_SD):
        d = df if oracle != "gp32" else df[df.stage == "initial_DoE"]
        X, y = scale(d[FEATURES].values), d.sigma_ad_MPa.values
        if oracle in ("gp51", "gp32"):
            m = gp(noise=0.02, fixed=True).fit(X, y)  # normalised scale: ~1.4 MPa, near-interpolating
        elif oracle == "rf51":
            m = RandomForestRegressor(n_estimators=500, random_state=0).fit(X, y)
        elif oracle == "xgb51":
            m = XGBRegressor(max_depth=4, gamma=2.86, learning_rate=0.4, n_estimators=100,
                             random_state=0).fit(X, y)
        else:
            raise ValueError(oracle)
        self.truth = np.clip(m.predict(scale(GRID)), 0, None); self.noise = noise
    def measure(self, idx, rng):
        return np.clip(self.truth[idx] + rng.normal(0, self.noise, len(idx)), 0, None)

def top_k(scores, cand, k, rng):
    """Top-k with random tie-breaking (tree models predict many exact ties)."""
    return cand[np.argsort(-(scores + rng.uniform(0, 1e-6, len(scores))))[:k]]

def propose(strategy, X, y, cand, k, rng):
    if strategy == "random":
        return rng.choice(cand, k, replace=False)
    if strategy == "greedy_xgb":  # the paper's AL loop: top-k predicted
        m = XGBRegressor(max_depth=4, gamma=2.7, n_estimators=100, learning_rate=0.1, random_state=0)
        return top_k(m.fit(scale(X), y).predict(scale(GRID[cand])), cand, k, rng)
    if strategy == "greedy_rf":
        m = RandomForestRegressor(max_depth=3, n_estimators=200, random_state=0)
        return top_k(m.fit(scale(X), y).predict(scale(GRID[cand])), cand, k, rng)
    if strategy in ("gp_mean", "gp_ei", "gp_ucb"):
        # gp_mean = same GP, same batching, but ranks by predicted mean only (no uncertainty):
        # an ablation that isolates the value of exploration.
        # Batch by "kriging believer": pick the best point, pretend its measurement equals
        # the GP mean, refit (hyperparameters frozen after the first fit), repeat k times.
        X, y, cand, chosen = X.copy(), y.copy(), cand.copy(), []
        m = gp().fit(scale(X), y); kernel = m.kernel_
        incumbent = m.predict(scale(X)).max()  # best posterior mean, robust to noisy maxima
        for i in range(k):
            if i > 0:
                m = GaussianProcessRegressor(kernel, normalize_y=True, optimizer=None).fit(scale(X), y)
            mu, sd = m.predict(scale(GRID[cand]), return_std=True)
            if strategy == "gp_mean":
                a = mu
            elif strategy == "gp_ei":
                z = (mu - incumbent) / np.maximum(sd, 1e-9)
                a = (mu - incumbent) * norm.cdf(z) + sd * norm.pdf(z)
            else:
                a = mu + 2.0 * sd
            j = int(np.argmax(a)); chosen.append(cand[j])
            X = np.vstack([X, GRID[cand[j]]]); y = np.append(y, mu[j]); cand = np.delete(cand, j)
        return np.array(chosen)
    raise ValueError(strategy)

def maximin_subset(points_idx, n, rng):
    """Space-filling subset: random first point, then repeatedly the farthest one."""
    P = scale(GRID[points_idx]); chosen = [int(rng.integers(len(P)))]
    while len(chosen) < n:
        d = np.min(((P[:, None, :] - P[None, chosen, :]) ** 2).sum(-1), axis=1)
        d[chosen] = -1; chosen.append(int(np.argmax(d)))
    return [points_idx[i] for i in chosen]

def campaign(strategy, lab, init_idx, seed, batches=3, k=5):
    """Returns, after 0, 5, 10, 15 new experiments: best true value tested so far
    (initial + new), and whether a new proposal beats the best initial recipe."""
    rng = np.random.default_rng(seed)
    idx = list(init_idx); y = list(lab.measure(np.array(idx), rng))
    init_best = lab.truth[idx].max(); out = [(init_best, False)]
    for _ in range(batches):
        cand = np.setdiff1d(np.arange(len(GRID)), idx)
        new = propose(strategy, GRID[idx], np.array(y), cand, k, rng)
        idx += list(new); y += list(lab.measure(new, rng))
        new_best = lab.truth[idx[len(init_idx):]].max()
        out.append((max(init_best, new_best), bool(new_best > init_best)))
    return out

def campaign_trace(strategy, lab, init_idx, seed, batches=3, k=5):
    """Same campaign as `campaign` (same random draws), but keeps every proposal:
    batches = [[(grid index, true value, measured value), ...], ...]."""
    rng = np.random.default_rng(seed)
    idx = list(init_idx); y = list(lab.measure(np.array(idx), rng))
    best = [float(lab.truth[idx].max())]; out = []
    for _ in range(batches):
        cand = np.setdiff1d(np.arange(len(GRID)), idx)
        new = propose(strategy, GRID[idx], np.array(y), cand, k, rng)
        meas = lab.measure(new, rng)
        idx += list(new); y += list(meas)
        out.append([(int(i), float(lab.truth[i]), float(m)) for i, m in zip(new, meas)])
        best.append(max(best[-1], float(lab.truth[new].max())))
    return dict(batches=out, best_so_far=best)

def _run(args):
    oracle, n_init, strategy, n_seeds = args
    df = load(); lab = Lab(df, oracle)
    doe = df[df.stage == "initial_DoE"][FEATURES].values
    doe_idx = [int(np.argmin(np.abs(GRID - row).sum(1))) for row in doe]
    rows = []
    for seed in range(n_seeds):
        start = doe_idx if n_init == len(doe_idx) else maximin_subset(doe_idx, n_init, np.random.default_rng(1000 + seed))
        for b, (best, beats) in enumerate(campaign(strategy, lab, start, seed)):
            rows.append(dict(oracle=oracle, n_init=n_init, strategy=strategy, seed=seed, n_new=5 * b,
                             best_true=best, beats_init=beats, optimum=lab.truth.max(),
                             regret=lab.truth.max() - best))
    return rows

def stats(df):
    """Paired comparison GP-EI vs the paper's greedy XGBoost (same seeds), after 15 experiments."""
    last = df[df.n_new == 15]; out = []
    for (o, n), g in last.groupby(["oracle", "n_init"]):
        p = g.pivot(index="seed", columns="strategy", values="best_true")
        diff = p["gp_ei"] - p["greedy_xgb"]
        pval = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
        abl = p["gp_ei"] - p["gp_mean"]  # value of using uncertainty, same model
        pabl = wilcoxon(abl).pvalue if (abl != 0).any() else 1.0
        row = dict(oracle=o, n_init=n, ei_minus_greedy_xgb=diff.mean(), ei_wins=int((diff > 0).sum()),
                   n_seeds=len(diff), wilcoxon_p=pval, ei_minus_gp_mean=abl.mean(), ablation_p=pabl)
        for s in STRATEGIES:
            row[f"mean_{s}"] = p[s].mean()
            row[f"beats_init_{s}"] = g[g.strategy == s].beats_init.mean()
        out.append(row)
    return pd.DataFrame(out)

def main(n_seeds=20, n_init_list=(32, 10)):
    jobs = [(o, n, s, n_seeds) for o in ORACLES for n in n_init_list for s in STRATEGIES]
    with ProcessPoolExecutor() as ex:
        rows = [r for chunk in ex.map(_run, jobs) for r in chunk]
    df = pd.DataFrame(rows); df.to_csv(ROOT / "results" / "benchmark.csv", index=False)
    st = stats(df); st.to_csv(ROOT / "results" / "benchmark_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(st[["oracle", "n_init", "mean_random", "mean_greedy_xgb", "mean_gp_ei",
              "ei_minus_greedy_xgb", "wilcoxon_p", "mean_gp_mean",
              "ei_minus_gp_mean", "ablation_p"]].round(2).to_string(index=False))
    return df, st

if __name__ == "__main__":
    main()
