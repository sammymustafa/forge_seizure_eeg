"""Download CHB-MIT files from PhysioNet with resume + retry."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .config import DATA_ROOT, PHYSIONET_BASE


def fetch(rel_path: str, dest_root: Path = DATA_ROOT, *, retries: int = 4,
          timeout: tuple[int, int] = (20, 180)) -> Path:
    """Download ``rel_path`` (e.g. ``chb01/chb01_03.edf``) under ``dest_root``.

    Skips the transfer when a non-empty file of the expected size is present.
    """
    dest = dest_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = PHYSIONET_BASE + rel_path

    expected = None
    for attempt in range(1, retries + 1):
        try:
            if expected is None:
                head = requests.head(url, timeout=timeout, allow_redirects=True)
                head.raise_for_status()
                expected = int(head.headers.get("Content-Length", 0)) or None
            if dest.exists() and expected and dest.stat().st_size == expected:
                return dest
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                if expected and tmp.stat().st_size != expected:
                    raise OSError(
                        f"short read for {rel_path}: {tmp.stat().st_size} != {expected}"
                    )
                tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - network retry loop
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt}/{retries} for {rel_path} after {exc!r}; "
                  f"sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_summary(case: str, dest_root: Path = DATA_ROOT) -> Path:
    return fetch(f"{case}/{case}-summary.txt", dest_root)
