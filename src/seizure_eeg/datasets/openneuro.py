"""Minimal OpenNeuro S3 client (public buckets, no credentials)."""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
_KEY_RE = re.compile(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", re.S)
_TOKEN_RE = re.compile(r"<NextContinuationToken>(.*?)</NextContinuationToken>")


def list_keys(prefix: str, timeout: int = 60) -> list[tuple[str, int]]:
    """Every ``(key, size)`` under ``prefix`` in the public OpenNeuro bucket."""
    token, out = None, []
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = f"{BUCKET}?{urllib.parse.urlencode(q)}"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode()
        out.extend((m.group(1), int(m.group(2))) for m in _KEY_RE.finditer(body))
        m = _TOKEN_RE.search(body)
        if not m:
            return out
        token = urllib.parse.unquote(m.group(1))


def fetch_key(key: str, dest_root: Path, *, strip_prefix: str = "",
              retries: int = 4, timeout: int = 180,
              expected_size: int | None = None) -> Path:
    """Download one object, skipping it when a correctly sized copy exists."""
    rel = key[len(strip_prefix):] if strip_prefix and key.startswith(strip_prefix) else key
    dest = dest_root / rel
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    for attempt in range(1, retries + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            if expected_size is not None and tmp.stat().st_size != expected_size:
                raise OSError(f"size mismatch for {key}")
            tmp.replace(dest)
            return dest
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")
