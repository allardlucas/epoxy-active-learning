"""Tests of the explorer app's API: each endpoint answers, and returns the
numbers the slides rely on."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "api"))
from main import app  # noqa: E402

client = TestClient(app)


def get(url, status=200):
    r = client.get(url)
    assert r.status_code == status, r.text
    return r.json()


def test_campaign_no_active_learning_proposal_beats_initial_best():
    d = get("/api/campaign")
    assert len(d["rows"]) == 51
    assert d["best_initial"] == 31.9
    assert d["n_al"] == 15
    assert d["al_beating_initial"] == 0
    assert d["al_with_mwc400"] == 13


def test_loocv_gain_is_smaller_on_the_initial_design():
    d = get("/api/loocv")
    curve = {c["n"]: c for c in d["curve"]}
    assert sorted(curve) == [32, 37, 42, 47]
    start, end = curve[32], curve[47]
    assert start["r2_all"] == pytest.approx(start["r2_doe"])
    # R² rises on the growing set, much less on the 32 original points
    assert end["r2_all"] - start["r2_all"] > end["r2_doe"] - start["r2_doe"]


def test_replay_record_was_unlikely_for_the_gp():
    d = get("/api/replay?mwe=370&mwc=400")
    assert len(d["r"]) == 76 and len(d["T"]) == 121
    pts = {p["id"]: p for p in d["bo_points"]}
    assert sorted(pts) == [48, 49, 50, 51]
    assert pts[51]["measured"] == 35.8
    assert pts[51]["p_beat"] < 0.05
    # a tree model predicts a few flat plateaus over the whole plane
    assert len(d["plateaus"]) <= 15


def test_replay_other_recipes_have_no_bo_points():
    assert get("/api/replay?mwe=1650&mwc=2000")["bo_points"] == []


def test_simulate_runs_three_batches_of_five():
    d = get("/api/simulate?oracle=gp32&strategy=greedy_xgb&n_init=10&seed=0")
    assert len(d["initial"]) == 10
    assert [len(b) for b in d["batches"]] == [5, 5, 5]
    best = d["best_so_far"]
    assert len(best) == 4 and best == sorted(best)
    assert best[-1] <= d["optimum"]["true"] + 1e-9


def test_benchmark_summary_covers_all_strategies():
    d = get("/api/benchmark")
    assert {r["strategy"] for r in d["rows"]} == {"random", "greedy_xgb", "greedy_rf", "gp_mean", "gp_ei", "gp_ucb"}
    assert set(d["optimum"]) == {"gp51", "rf51", "xgb51", "gp32"}


def test_exploration_never_beats_mean_only_gp():
    sc = get("/api/benchmark")["scenarios"]
    assert len(sc) == 8
    # GP + EI vs GP mean only: never a significant difference, on any fake lab or start
    assert all(s["p_ei_vs_mean"] > 0.05 for s in sc)


@pytest.mark.parametrize("url", [
    "/api/loocv?model=svm",
    "/api/replay?mwe=999",
    "/api/simulate?n_init=20",
    "/api/simulate?strategy=nope",
])
def test_bad_parameters_are_rejected(url):
    get(url, status=400)


def test_few_recipes_beat_the_plan():
    h = {r["oracle"]: r for r in get("/api/benchmark")["headroom"]}
    assert set(h) == {"gp51", "gp32", "rf51", "xgb51"}
    # the 32-trial plan already sits in the top few percent of every fake lab
    assert all(r["share_better"] < 0.05 for r in h.values())
    assert h["gp51"]["n_better"] < 30
