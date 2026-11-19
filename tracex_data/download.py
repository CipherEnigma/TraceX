"""Download the three raw datasets into data/raw/ (idempotent)."""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from .config import FOURSQUARE_URL, MASSIVE_STEPS_URL, RAW_DIR


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[download] exists, skipping: {dest}")
        return dest
    print(f"[download] {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "tracex-data/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def download_foursquare() -> Path:
    out_dir = RAW_DIR / "foursquare"
    zip_path = _fetch(FOURSQUARE_URL, out_dir / "dataset_tsmc2014.zip")
    target = out_dir / "dataset_tsmc2014"
    if not (target / "dataset_TSMC2014_NYC.txt").exists():
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():  # guard against path traversal
                if Path(member).is_absolute() or ".." in Path(member).parts:
                    raise ValueError(f"unsafe path in zip: {member}")
            z.extractall(out_dir)
    return target


def download_massive_steps(city: str = "Tokyo") -> Path:
    slug = city.replace(" ", "-")
    fname = f"{slug.lower().replace('-', '_')}_checkins.csv"
    out_dir = RAW_DIR / "massive_steps" / slug.lower().replace("-", "_")
    _fetch(MASSIVE_STEPS_URL.format(city=slug, file=fname), out_dir / fname)
    _fetch(MASSIVE_STEPS_URL.format(city=slug, file="README.md"), out_dir / "README.md")
    return out_dir / fname


def download_all() -> None:
    download_foursquare()
    download_massive_steps("Tokyo")


if __name__ == "__main__":
    download_all()
