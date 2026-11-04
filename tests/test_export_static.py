"""The static export writes each answer where the front end will look for it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_static import SEEDS, target, urls  # noqa: E402


def test_file_names_match_the_front_end_keys():
    # apiUrl() in App.jsx: "k-v" pairs, sorted, joined by "_"
    assert target(Path("d"), "/api/loocv?model=xgb&max_depth=5") == Path("d/loocv/max_depth-5_model-xgb.json")
    assert target(Path("d"), "/api/campaign") == Path("d/campaign/default.json")


def test_every_request_is_exported_once():
    all_urls = list(urls())
    assert len(all_urls) == len(set(all_urls))
    assert sum("/api/simulate" in u for u in all_urls) == 4 * 6 * 2 * SEEDS
    assert sum("/api/replay" in u for u in all_urls) == 16
