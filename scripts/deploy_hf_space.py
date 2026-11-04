"""Publish the full explorer app (Python API + React front end, one Docker image)
as a Hugging Face Space, free CPU hardware.

    HF_TOKEN=... python scripts/deploy_hf_space.py

The Space is <your HF username>/epoxy-explorer. Hugging Face builds it from the
repository's Dockerfile; the README written here only adds the Space settings.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = "epoxy-explorer"
GITHUB = "https://github.com/allardlucas/epoxy-active-learning"
# What the Dockerfile needs, nothing else
KEEP = ["Dockerfile", ".dockerignore", "app", "src", "data", "results"]
IGNORE = shutil.ignore_patterns("node_modules", "dist", "__pycache__", ".pytest_cache")

README = f"""---
title: Epoxy active learning, critical explorer
emoji: 🧪
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
short_description: Critical reproduction of Pruksawan et al. (2019)
---

Interactive explorer for a critical reproduction of Pruksawan et al. (2019),
*Prediction and optimization of epoxy adhesive strength from a small dataset through
active learning*. Code, data and method: [{GITHUB}]({GITHUB}).
"""


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = f"{api.whoami()['name']}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip() or "local"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for name in KEEP:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, out / name, ignore=IGNORE)
            else:
                shutil.copy2(src, out / name)
        (out / "README.md").write_text(README, encoding="utf-8")
        api.upload_folder(folder_path=out, repo_id=repo_id, repo_type="space",
                          commit_message=f"Deploy {sha} from GitHub", delete_patterns="*")
    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
