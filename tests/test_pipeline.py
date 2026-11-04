import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import benchmark as b
from reproduce import load, FEATURES

def test_dataset_matches_paper():
    df = load()
    assert len(df) == 51
    assert df.stage.value_counts().to_dict() == {"initial_DoE": 32, "active_learning": 15, "bayesian_opt": 4}
    assert df.sigma_ad_MPa.max() == 35.8

def test_initial_design_is_graeco_latin():
    # each level of each factor appears exactly 8 times in the two 4x4 squares
    init = load().query("stage == 'initial_DoE'")
    for f in FEATURES:
        assert set(init[f].value_counts()) == {8}

def test_initial_points_are_on_grid():
    init = load().query("stage == 'initial_DoE'")[FEATURES].values
    assert all(np.abs(b.GRID - row).sum(1).min() < 1e-9 for row in init)

def test_propose_returns_unique_candidates():
    df = load()
    X, y = df[FEATURES].values[:20], df.sigma_ad_MPa.values[:20]
    cand = np.arange(200)
    for s in ["random", "greedy_xgb", "gp_ei"]:
        out = b.propose(s, X, y, cand, 5, np.random.default_rng(0))
        assert len(set(out)) == 5 and set(out) <= set(cand)
