"""FastAPI server: JSON endpoints for the explorer, and the built React app as static files."""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import analysis
import benchmark

app = FastAPI(title="Epoxy active learning — critical explorer")


@app.get("/api/campaign")
def campaign():
    return analysis.campaign_data()


@app.get("/api/loocv")
def loocv(model: str = "xgb", max_depth: int | None = None, paper_hp: bool = False):
    if model not in ("xgb", "rf", "enet"):
        raise HTTPException(400, "model must be xgb, rf or enet")
    return analysis.loocv_split(model, max_depth, paper_hp)


@app.get("/api/replay")
def replay(mwe: int = 370, mwc: int = 400):
    if mwe not in benchmark.MWE or mwc not in benchmark.MWC:
        raise HTTPException(400, "unknown molecular weight")
    return analysis.replay(mwe, mwc)


@app.get("/api/simulate")
def simulate(oracle: str = "gp51", strategy: str = "greedy_xgb", n_init: int = 32, seed: int = 0):
    if oracle not in benchmark.ORACLES or strategy not in benchmark.STRATEGIES or n_init not in (10, 32):
        raise HTTPException(400, "bad parameters")
    return analysis.simulate(oracle, strategy, n_init, seed)


@app.get("/api/benchmark")
def bench():
    return analysis.benchmark_summary()


WEB = Path(os.environ.get("WEB_DIST", Path(__file__).resolve().parents[1] / "web" / "dist"))
if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        return FileResponse(WEB / "index.html")
