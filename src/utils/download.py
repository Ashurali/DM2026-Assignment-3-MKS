"""Download competition data via kagglehub and cache to data/raw/."""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
COMPETITION = "nycu-data-mining-assignment-3"


def download_and_link() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    dst = DATA_DIR / "raw"
    if dst.exists() and any(dst.iterdir()):
        print(f"{dst} already populated — skipping download.")
        return dst

    print(f"Downloading {COMPETITION} via kagglehub...")
    src = Path(kagglehub.competition_download(COMPETITION))
    print(f"Cached at: {src}")

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied into: {dst}")
    return dst


if __name__ == "__main__":
    p = download_and_link()
    print("\nTop-level contents:")
    for child in sorted(p.iterdir()):
        marker = "/" if child.is_dir() else ""
        print(f"  {child.name}{marker}")
