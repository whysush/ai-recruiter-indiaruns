#!/usr/bin/env python3
"""ONE-TIME pre-computation: embed the candidate pool and the ideal-profile text.

This is the only step that may use the network (to download the local model once)
and may exceed the 5-minute ranking budget. Its output — a float16 embedding
matrix aligned to candidate order, the id list, and the ideal vector — is the
declared pre-computed artifact the timed ranking step consumes with numpy only.

Usage:
    python scripts/precompute_embeddings.py \
        --candidates data/candidates.jsonl \
        --cache artifacts/ \
        [--limit N]   # for quick local tests
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airecruiter import dataio, embeddings
from airecruiter.jd import build_job_spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd", default=str(Path(__file__).resolve().parents[1] / "data" / "job_description.txt"))
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--model", default=embeddings.MODEL_NAME)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[precompute] loading model {args.model} (CPU) ...", flush=True)
    model = embeddings.load_model(args.model)

    print("[precompute] loading candidates ...", flush=True)
    records = dataio.load_candidates(args.candidates)
    if args.limit:
        records = records[: args.limit]
    print(f"[precompute] {len(records)} candidates", flush=True)

    ids = [r["candidate_id"] for r in records]
    texts = [dataio.candidate_text(r) for r in records]

    print("[precompute] embedding ideal profile ...", flush=True)
    job = build_job_spec(Path(args.jd).read_text(encoding="utf-8"))
    ideal_vec = embeddings.embed_query(model, job.ideal_text)

    print(f"[precompute] embedding {len(texts)} candidate texts ...", flush=True)
    matrix = embeddings.embed_texts(model, texts, batch_size=args.batch_size)

    embeddings.save_cache(args.cache, ids, matrix, ideal_vec)
    dt = time.time() - t0
    print(
        f"[precompute] done in {dt:.0f}s -> {args.cache}  "
        f"(matrix {matrix.shape}, float16)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
