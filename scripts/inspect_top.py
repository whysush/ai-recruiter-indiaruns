#!/usr/bin/env python3
"""Diagnostic: show the top-N with their component sub-scores, and audit how many
honeypots / aspirational stuffers leaked into the top-100. Not part of submission."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airecruiter import dataio, embeddings, score
from airecruiter.jd import build_job_spec
from airecruiter.honeypot import audit
from run_submission import align_semantic, compute_anchor  # type: ignore

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(REPO / "data" / "candidates.jsonl"))
    ap.add_argument("--cache", default=str(REPO / "artifacts"))
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    job = build_job_spec((REPO / "data" / "job_description.txt").read_text())
    records = dataio.load_candidates(args.candidates)
    cache = embeddings.load_cache(args.cache)
    sem = align_semantic(records, cache)
    anchor = compute_anchor(records)
    rows = score.score_pool_parallel(records, job, sem, anchor)
    # parallel drops _rec; re-attach for the inspected rows
    by_id = {r["candidate_id"]: r for r in records}
    top = score.rank_pool(rows, top_k=100)
    for r in top:
        r.setdefault("_rec", by_id[r["candidate_id"]])
    top = score.rescale_scores(top)

    print(f"\n=== TOP {args.n} ===")
    print(f"{'rank':>4} {'id':<14} {'score':>6} {'sem':>4} {'evid':>4} {'titl':>4} "
          f"{'care':>4} {'skil':>4} {'env':>4} {'pen':>4}  title")
    for r in top[: args.n]:
        p = r["_rec"]["profile"]
        print(f"{r['rank']:>4} {r['candidate_id']:<14} {r['score']:>6.3f} "
              f"{r['semantic_fit']:>4.2f} {r['evidence_fit']:>4.2f} {r['title_relevance']:>4.2f} "
              f"{r['career_relevance']:>4.2f} {r['skill_evidence']:>4.2f} {r['behavioral_mult']:>4.2f} "
              f"{r['honeypot_penalty']:>4.2f}  {p['current_title'][:28]}")

    # Honeypot leakage in top 100
    leaked = [r for r in top if r["honeypot_penalty"] >= 0.25]
    print(f"\n=== top-100 with strong honeypot penalty (>=0.25): {len(leaked)} ===")
    for r in leaked[:10]:
        print(f"  rank {r['rank']:>3} {r['candidate_id']} pen={r['honeypot_penalty']:.2f} "
              f"flags={r['honeypot_flags']}")

    # Title mix of top 100
    from collections import Counter
    tc = Counter(r["_rec"]["profile"]["current_title"] for r in top)
    print(f"\n=== top-100 current_title mix ===")
    for t, c in tc.most_common(15):
        print(f"  {c:>3}  {t}")


if __name__ == "__main__":
    main()
