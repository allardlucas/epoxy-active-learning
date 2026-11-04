"""Analyses behind the explorer app. Each function answers one critical question
about Pruksawan et al. (2019) and returns plain JSON-able data for the front end."""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark import GRID, Lab, campaign_trace, gp, maximin_subset  # noqa: E402

FEATURES = ["MW_epoxy", "MW_curing_agent", "r", "T_cure"]
TARGET = "sigma_ad_MPa"
BEST_INITIAL = 31.9  # sample 17, best of the 32-point design
# XGBoost hyperparameters per cycle, Table S4 of the paper
PAPER_XGB = [dict(max_depth=5, gamma=3.3, learning_rate=0.1), dict(max_depth=4, gamma=2.7, learning_rate=0.1),
             dict(max_depth=4, gamma=1.8, learning_rate=0.1), dict(max_depth=4, gamma=2.86, learning_rate=0.4)]


@lru_cache
def load() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "epoxy_adhesive_dataset.csv")


# ---------------------------------------------------------------- 1. campaign
def campaign_data():
    """The 51 published measurements, and how the active-learning proposals compare
    with the best point of the initial design."""
    df = load()
    al = df[df.stage == "active_learning"]
    return dict(
        rows=df.replace({np.nan: None}).to_dict("records"),
        best_initial=BEST_INITIAL,
        al_beating_initial=int((al[TARGET] > BEST_INITIAL).sum()),
        al_with_mwc400=int((al.MW_curing_agent == 400).sum()),
        n_al=len(al),
    )


# ---------------------------------------------------------------- 2. R²
def make_model(name, max_depth=None, cycle=None):
    if name == "xgb":
        p = PAPER_XGB[cycle] if cycle is not None else dict(PAPER_XGB[0], max_depth=max_depth or 5)
        return XGBRegressor(n_estimators=100, random_state=0, **p)
    if name == "rf":
        return RandomForestRegressor(max_depth=max_depth or 3, n_estimators=100, random_state=2)
    if name == "enet":
        return ElasticNet(alpha=0.1)
    raise ValueError(name)


def loocv_split(model="xgb", max_depth=None, paper_hp=False):
    """LOOCV R² as the paper computes it (on every point), and the same LOOCV
    predictions scored only on the 32 initial design points, which cover the space."""
    df = load(); curve = []; preds = None
    for cycle, n in enumerate([32, 37, 42, 47]):
        sub = df[df.id <= n]; y = sub[TARGET].values
        m = make_model(model, max_depth, cycle if paper_hp else None)
        p = cross_val_predict(make_pipeline(StandardScaler(), m), sub[FEATURES].values, y, cv=LeaveOneOut())
        doe = (sub.stage == "initial_DoE").values
        curve.append(dict(n=n, r2_all=r2_score(y, p), r2_doe=r2_score(y[doe], p[doe]),
                          r2_new=r2_score(y[~doe], p[~doe]) if (~doe).sum() > 2 else None))
        preds = [dict(id=int(i), stage=s, measured=float(a), predicted=float(b))
                 for i, s, a, b in zip(sub.id, sub.stage, y, p)]
    return dict(curve=curve, predictions_47=preds, paper=dict(start=0.68, end=0.85))


# ---------------------------------------------------------------- 3. replay of the BO step
@lru_cache
def _models_47():
    df = load(); d = df[df.id <= 47]
    X, y = d[FEATURES].values.astype(float), d[TARGET].values
    mu, sd = X.mean(0), X.std(0)
    xgb = XGBRegressor(n_estimators=100, random_state=0, **PAPER_XGB[3]).fit((X - mu) / sd, y)
    Xg = X.copy(); Xg[:, :2] = np.log(Xg[:, :2]); lo, hi = Xg.min(0), Xg.max(0)
    g = gp().fit((Xg - lo) / (hi - lo), y)
    return xgb, (mu, sd), g, (lo, hi)


def replay(mwe=370, mwc=400):
    """What the paper's fixed XGBoost (47 points) and a GP on the same 47 points
    predict over the (r, T) plane, and the chance each gave to beat 31.9 MPa."""
    xgb, (mu, sd), g, (lo, hi) = _models_47()
    r = np.round(np.arange(0.75, 1.5001, 0.01), 2); T = np.arange(90, 211, 1)
    R, TT = np.meshgrid(r, T)
    X = np.c_[np.full(R.size, mwe), np.full(R.size, mwc), R.ravel(), TT.ravel()].astype(float)
    p_xgb = xgb.predict((X - mu) / sd)
    Xg = X.copy(); Xg[:, :2] = np.log(Xg[:, :2])
    m, s = g.predict((Xg - lo) / (hi - lo), return_std=True)
    p_beat = 1 - norm.cdf((BEST_INITIAL - m) / s)
    values, counts = np.unique(p_xgb.round(2), return_counts=True)
    bo = load(); bo = bo[bo.stage == "bayesian_opt"]
    pts = []
    for _, b in bo.iterrows():
        x = np.array([[b.MW_epoxy, b.MW_curing_agent, b.r, b.T_cure]], float)
        xg = x.copy(); xg[:, :2] = np.log(xg[:, :2])
        gm, gs = g.predict((xg - lo) / (hi - lo), return_std=True)
        pts.append(dict(id=int(b.id), r=b.r, T=b.T_cure, measured=b[TARGET], xgb=float(xgb.predict((x - mu) / sd)[0]),
                        gp_mean=float(gm[0]), gp_sd=float(gs[0]), p_beat=float(1 - norm.cdf((BEST_INITIAL - gm[0]) / gs[0]))))
    shape = R.shape
    return dict(r=r.tolist(), T=T.tolist(), xgb=p_xgb.reshape(shape).tolist(), gp_mean=m.reshape(shape).tolist(),
                gp_sd=s.reshape(shape).tolist(), p_beat=p_beat.reshape(shape).tolist(),
                plateaus=[dict(value=float(v), n=int(c)) for v, c in sorted(zip(values, counts), reverse=True)],
                bo_points=pts if (mwe, mwc) == (370, 400) else [], best_initial=BEST_INITIAL)


# ---------------------------------------------------------------- 4. simulated labs
@lru_cache
def _lab(oracle):
    return Lab(load(), oracle)


def simulate(oracle="gp51", strategy="greedy_xgb", n_init=32, seed=0):
    """One campaign on a simulated lab: 3 batches of 5 experiments, like the paper."""
    df = load(); lab = _lab(oracle)
    doe = df[df.stage == "initial_DoE"][FEATURES].values
    doe_idx = [int(np.argmin(np.abs(GRID - row).sum(1))) for row in doe]
    start = doe_idx if n_init >= 32 else maximin_subset(doe_idx, n_init, np.random.default_rng(1000 + seed))
    trace = campaign_trace(strategy, lab, start, seed)
    rec = lambda i: dict(zip(FEATURES, GRID[i].tolist()))
    best = int(np.argmax(lab.truth))
    return dict(
        initial=[dict(rec(i), true=float(lab.truth[i])) for i in start],
        batches=[[dict(rec(i), true=float(t), measured=float(m)) for i, t, m in b] for b in trace["batches"]],
        best_so_far=trace["best_so_far"], optimum=dict(rec(best), true=float(lab.truth[best])),
    )


def benchmark_summary():
    """Precomputed: 20 paired campaigns per (oracle, start, strategy), src/benchmark.py."""
    b = pd.read_csv(ROOT / "results" / "benchmark.csv"); last = b[b.n_new == 15]
    g = last.groupby(["oracle", "n_init", "strategy"]).best_true.agg(["mean", "sem"]).reset_index()
    opt = last.groupby("oracle").optimum.first().to_dict()
    return dict(rows=g.to_dict("records"), optimum=opt)
