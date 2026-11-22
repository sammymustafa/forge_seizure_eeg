"""Parse CHB-MIT ``chbXX-summary.txt`` annotation files.

Seizure intervals are taken from the subject summary, never inferred from a
filename. A file that appears in the summary with no seizure block is a genuine
seizure-free recording and is kept (it supplies interictal windows).
"""
from __future__ import annotations

import re
from pathlib import Path

FILE_BLOCK_RE = re.compile(
    r"File Name:\s*(?P<file>\S+\.edf)(?P<block>.*?)(?=File Name:|\Z)", re.S
)
START_RE = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", re.I)
END_RE = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", re.I)
NSEIZ_RE = re.compile(r"Number of Seizures in File:\s*(\d+)", re.I)
CHANNEL_RE = re.compile(r"Channel\s+\d+:\s*(\S+)", re.I)


def parse_summary(summary_path: str | Path) -> dict[str, list[tuple[int, int]]]:
    """Return ``{edf_name: [(start_s, end_s), ...]}`` for every file in the summary."""
    text = Path(summary_path).read_text(errors="replace")
    out: dict[str, list[tuple[int, int]]] = {}
    for m in FILE_BLOCK_RE.finditer(text):
        block = m.group("block")
        starts = [int(s) for s in START_RE.findall(block)]
        ends = [int(s) for s in END_RE.findall(block)]
        if len(starts) != len(ends):
            raise ValueError(
                f"{Path(summary_path).name}/{m.group('file')}: "
                f"{len(starts)} starts vs {len(ends)} ends"
            )
        declared = NSEIZ_RE.search(block)
        if declared is not None and int(declared.group(1)) != len(starts):
            raise ValueError(
                f"{Path(summary_path).name}/{m.group('file')}: summary declares "
                f"{declared.group(1)} seizures but {len(starts)} were parsed"
            )
        for a, b in zip(starts, ends):
            if b <= a:
                raise ValueError(
                    f"{Path(summary_path).name}/{m.group('file')}: non-positive "
                    f"seizure duration {a}-{b}"
                )
        out[m.group("file")] = sorted(zip(starts, ends))
    return out


def summary_channels(summary_path: str | Path) -> list[list[str]]:
    """Return every declared channel block (a subject may switch montage mid-study)."""
    text = Path(summary_path).read_text(errors="replace")
    blocks: list[list[str]] = []
    for chunk in re.split(r"Channels in EDF Files:", text)[1:]:
        head = chunk.split("File Name:")[0]
        names = [n.upper() for n in CHANNEL_RE.findall(head)]
        if names:
            blocks.append(names)
    return blocks


def seizure_files(annotations: dict[str, list[tuple[int, int]]]) -> list[str]:
    return [f for f, s in annotations.items() if s]


def seizure_free_files(annotations: dict[str, list[tuple[int, int]]]) -> list[str]:
    return [f for f, s in annotations.items() if not s]
