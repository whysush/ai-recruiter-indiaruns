"""Robust candidate loading.

Handles a JSON array, JSONL, or gzip-compressed variants of either. We stream
JSONL so the 100K-line / ~465 MB file never has to be parsed twice or held in an
intermediate string. Records are returned as plain dicts — we do not mutate the
raw structure, we only validate the id format and surface a normalized text blob
that downstream modules reuse.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Iterator

CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield candidate dicts from .json / .jsonl / .json.gz / .jsonl.gz.

    A JSON array is detected by sniffing the first non-whitespace byte; otherwise
    the file is treated as line-delimited JSON.
    """
    path = Path(path)
    with _open_maybe_gzip(path) as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if first == "":
            return
        rest_is_array = first == "["
        if rest_is_array:
            # Small relative to the JSONL pool; load the whole array.
            payload = first + f.read()
            for rec in json.loads(payload):
                yield rec
        else:
            # Line-delimited. Re-attach the first char we already consumed.
            f_line = first + f.readline()
            while f_line:
                line = f_line.strip()
                if line:
                    yield json.loads(line)
                f_line = f.readline()


def load_candidates(path: str | Path, validate: bool = True) -> list[dict]:
    """Load all candidates into memory, validating ids when asked."""
    out: list[dict] = []
    for rec in iter_candidates(path):
        if validate:
            cid = rec.get("candidate_id", "")
            if not CANDIDATE_ID_RE.match(cid):
                raise ValueError(f"Bad candidate_id format: {cid!r}")
        out.append(rec)
    return out


def candidate_text(rec: dict) -> str:
    """Concatenate the free-text fields that carry genuine semantic signal.

    Order matters only for readability; the embedding model is bag-of-context. We
    use headline + summary + every career title and description + skill names. We
    deliberately exclude name, location, education institution/tier (bias proxies).
    """
    p = rec.get("profile", {})
    parts: list[str] = []
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])
    if p.get("current_title"):
        parts.append(f"Current role: {p['current_title']}")
    for job in rec.get("career_history", []):
        title = job.get("title", "")
        desc = job.get("description", "")
        parts.append(f"{title}. {desc}".strip())
    skills = [s.get("name", "") for s in rec.get("skills", []) if s.get("name")]
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    return "\n".join(parts)
