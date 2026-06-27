#!/usr/bin/env python3
"""Produce the ranked top-100 submission CSV — the TIMED ranking step.

This step must run within the challenge budget: <= 5 min wall-clock, <= 16 GB RAM,
CPU only, NO network. It uses NUMPY ONLY (no torch): semantic similarity comes from
the pre-computed embedding cache in artifacts/. If the cache is absent, the system
degrades gracefully to a deterministic lexical semantic proxy so it still runs.

Single-command reproduction:
    python scripts/run_submission.py --candidates data/candidates.jsonl --out submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airecruiter import dataio, embeddings, reasoning, score
from airecruiter.jd import build_job_spec

REPO = Path(__file__).resolve().parents[1]


def compute_anchor(records: list[dict]) -> date:
    """Anchor 'today' to the most recent last_active_date in the pool (data is
    synthetic/stale), so recency penalties stay sensible and reproducible."""
    best = date(2000, 1, 1)
    for r in records:
        try:
            d = date.fromisoformat(r["redrob_signals"]["last_active_date"])
            if d > best:
                best = d
        except Exception:
            continue
    return best


def align_semantic(records: list[dict], cache) -> np.ndarray | None:
    """Return cosine-to-ideal aligned to `records`, or None if no usable cache."""
    if cache is None:
        return None
    ids, matrix, ideal = cache
    cos = embeddings.cosine_to_ideal(matrix, ideal)
    id_to_cos = dict(zip(ids, cos))
    out = np.empty(len(records), dtype=np.float32)
    missing = 0
    for i, r in enumerate(records):
        v = id_to_cos.get(r["candidate_id"])
        if v is None:
            missing += 1
            out[i] = float(np.median(cos))  # neutral for any id not in cache
        else:
            out[i] = v
    if missing:
        print(f"[rank] WARNING: {missing} candidates missing from embedding cache", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd", default=str(REPO / "data" / "job_description.txt"))
    ap.add_argument("--cache", default=str(REPO / "artifacts"))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap candidate count")
    args = ap.parse_args()

    t0 = time.time()
    job = build_job_spec(Path(args.jd).read_text(encoding="utf-8"))
    reasoning.attach_job(job)

    print("[rank] loading candidates ...", flush=True)
    records = dataio.load_candidates(args.candidates)
    if args.limit:
        records = records[: args.limit]
    print(f"[rank] {len(records)} candidates loaded ({time.time()-t0:.1f}s)", flush=True)

    cache = embeddings.load_cache(args.cache)
    semantic_raw = align_semantic(records, cache)
    if semantic_raw is None:
        print("[rank] no embedding cache -> using deterministic lexical fallback", flush=True)
    else:
        print(f"[rank] semantic from embeddings ({len(cache[0])} cached vecs)", flush=True)

    anchor = compute_anchor(records)
    print(f"[rank] recency anchor = {anchor.isoformat()}", flush=True)

    rows = score.score_pool(records, job, semantic_raw, anchor)
    top = score.rank_pool(rows, top_k=args.top_k)
    top = score.rescale_scores(top)

    for r in top:
        r["reasoning"] = reasoning.build_reasoning(r)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top:
            w.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])

    print(f"[rank] wrote {out_path} with {len(top)} rows in {time.time()-t0:.1f}s total", flush=True)
    if len(top) < args.top_k:
        print(f"[rank] WARNING: only {len(top)} candidates available (< {args.top_k}). "
              f"No ids were fabricated.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
