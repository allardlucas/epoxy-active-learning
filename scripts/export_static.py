"""Precompute every answer the explorer app can ask the API for, so the front end
can be published as a static site (Cloudflare Pages, free plan).

    python scripts/export_static.py out/data

Each request /api/<endpoint>?a=1&b=2 is written to <out>/<endpoint>/a-1_b-2.json,
the path the front end builds when it is compiled with VITE_STATIC=1 (see apiUrl
in app/web/src/App.jsx). Answers go through the real API (FastAPI TestClient),
so the static site shows exactly what the Docker app computes.
"""
import json
import os
import sys

# One thread per worker process: the pool already uses every core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "api"))

SEEDS = 30  # simulator seeds 0..29; must match STATIC_SEEDS in App.jsx
MWE, MWC = [370, 1650, 2900, 3800], [230, 400, 2000, 4000]
ORACLES = ["gp51", "rf51", "xgb51", "gp32"]
STRATEGIES = ["random", "greedy_xgb", "greedy_rf", "gp_mean", "gp_ei", "gp_ucb"]


def urls():
    """Every URL the front end can request (same parameter sets as App.jsx)."""
    yield "/api/campaign"
    yield "/api/benchmark"
    for d in range(2, 9):
        yield f"/api/loocv?model=xgb&max_depth={d}"
        yield f"/api/loocv?model=xgb&max_depth={d}&paper_hp=true"
        yield f"/api/loocv?model=rf&max_depth={d}"
    yield "/api/loocv?model=enet"
    for e, c in product(MWE, MWC):
        yield f"/api/replay?mwe={e}&mwc={c}"
    for o, s, n, seed in product(ORACLES, STRATEGIES, (10, 32), range(SEEDS)):
        yield f"/api/simulate?oracle={o}&strategy={s}&n_init={n}&seed={seed}"


def target(out: Path, url: str) -> Path:
    parts = urlsplit(url)
    name = parts.path.removeprefix("/api/")
    key = "_".join(sorted(f"{k}-{v}" for k, v in parse_qsl(parts.query))) or "default"
    return out / name / f"{key}.json"


_client = None


def fetch(args):
    global _client
    out, url = args
    if _client is None:
        from fastapi.testclient import TestClient
        from main import app
        _client = TestClient(app)
    r = _client.get(url)
    r.raise_for_status()
    path = target(Path(out), url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r.json(), separators=(",", ":")))
    return url


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/data")
    todo = [(str(out), u) for u in urls()]
    with ProcessPoolExecutor() as pool:
        for i, _ in enumerate(pool.map(fetch, todo, chunksize=4), 1):
            if i % 200 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", flush=True)
