"""Combine every component into a final, ranked, interpretable score.

    final_score = (w_sem * semantic_fit + w_ev * evidence_fit)   # fused fit
                  * behavioral_envelope                          # bounded [0.5,1.2]
                  * location_shaper                              # tiny [~0.96,1.0]
                  - honeypot_penalty                             # [0,0.6]

Why a weighted sum and not pure RRF: our components are calibrated to [0,1] and we
want the *magnitude* of evidence (e.g. a measured NLP assessment of 82) to move the
score, not just its rank. Evidence is weighted above semantic because the data is
structured gold and raw embedding similarity rewards keyword density — exactly the
trap we must avoid. Semantic fit is the "see beyond keywords" booster that rescues
plain-language candidates whose descriptions show real work without buzzwords.

Determinism: every step is pure arithmetic over the loaded records and the cached
embeddings. Same inputs -> same scores -> same CSV.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from . import features as F
from . import honeypot as H
from . import jd

# Fusion weights (documented in README; evidence dominates).
W_SEMANTIC = 0.40
W_EVIDENCE = 0.60

# Evidence sub-weights.
W_TITLE = 0.32
W_CAREER = 0.30
W_SKILL = 0.24
W_SENIOR = 0.14
# Small additive bonuses for eval-maturity and nice-to-haves (capped).
EVAL_BONUS = 0.10
NICE_BONUS = 0.06


def _percentile_scale(values: np.ndarray, lo_pct=5, hi_pct=95) -> np.ndarray:
    """Robust min-max to [0,1] using percentile clipping (outlier-safe)."""
    lo = np.percentile(values, lo_pct)
    hi = np.percentile(values, hi_pct)
    if hi <= lo:
        return np.zeros_like(values)
    scaled = (values - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def evidence_fit(rec: dict, job: jd.JobSpec, blob: F.Blob) -> dict:
    """Return the evidence sub-scores and their weighted combination."""
    title = F.title_relevance(rec)
    career = F.career_relevance(blob)
    skill = F.skill_evidence(rec, job, blob)
    senior = F.seniority_alignment(rec, job)

    eval_cov = F.concept_coverage_low(blob.all_l, job.eval_concepts)
    nice_cov = F.concept_coverage_low(blob.all_l, job.nice_to_have_concepts)

    base = W_TITLE * title + W_CAREER * career + W_SKILL * skill + W_SENIOR * senior
    bonus = EVAL_BONUS * eval_cov + NICE_BONUS * nice_cov
    fit = min(1.0, base + bonus)
    return {
        "evidence_fit": fit,
        "title_relevance": title,
        "career_relevance": career,
        "skill_evidence": skill,
        "seniority_alignment": senior,
        "eval_coverage": eval_cov,
        "nice_coverage": nice_cov,
    }


def lexical_semantic_proxy(job: jd.JobSpec, blob: F.Blob) -> float:
    """Offline fallback for semantic_fit when embeddings are unavailable.

    Concept-group coverage over the candidate's free text vs. the role's must-have
    and eval concepts. Deterministic, no model needed. Weaker than embeddings at
    catching buzzword-free fits, hence only a fallback.
    """
    must = F.concept_coverage_low(blob.all_l, job.must_have_concepts)
    ev = F.concept_coverage_low(blob.all_l, job.eval_concepts)
    return min(1.0, 0.8 * must + 0.2 * ev)


def _score_one(rec: dict, blob: F.Blob, job: jd.JobSpec, sem: float,
               semantic_source: str, anchor: date, keep_rec: bool) -> dict:
    """Score a single candidate given its precomputed semantic value."""
    ev = evidence_fit(rec, job, blob)
    env_mult, env_parts = F.behavioral_envelope(rec, anchor)
    loc = F.location_factor(rec)
    loc_shaper = 0.92 + 0.08 * loc  # tiny: range ~[0.96, 1.0]
    pen, flags = H.audit(rec, blob)

    fused = W_SEMANTIC * sem + W_EVIDENCE * ev["evidence_fit"]
    final = fused * env_mult * loc_shaper - pen

    row = {
        "candidate_id": rec["candidate_id"],
        "final_score": final,
        "fused_fit": fused,
        "semantic_fit": sem,
        "semantic_source": semantic_source,
        "behavioral_mult": env_mult,
        "location_factor": loc,
        "honeypot_penalty": pen,
        "honeypot_flags": flags,
        "behavioral_parts": env_parts,
        **ev,
    }
    if keep_rec:
        row["_rec"] = rec
    return row


def _semantic_array(records, blobs, job, semantic_raw) -> tuple[np.ndarray, str]:
    if semantic_raw is not None:
        return _percentile_scale(np.asarray(semantic_raw, dtype=np.float32)), "embeddings"
    sem = np.array([lexical_semantic_proxy(job, b) for b in blobs], dtype=np.float32)
    return sem, "lexical_fallback"


def score_pool(
    records: list[dict],
    job: jd.JobSpec,
    semantic_raw: np.ndarray | None,
    anchor: date,
    keep_rec: bool = True,
) -> list[dict]:
    """Score every candidate (single process). `semantic_raw` is cosine-to-ideal
    aligned to records, or None to use the lexical fallback."""
    blobs = [F.Blob(r) for r in records]
    sem_scaled, src = _semantic_array(records, blobs, job, semantic_raw)
    return [
        _score_one(rec, blobs[i], job, float(sem_scaled[i]), src, anchor, keep_rec)
        for i, rec in enumerate(records)
    ]


# --- parallel scoring (CPU cores; deterministic merge) ---------------------

_WORKER: dict = {}


def _setup_workers(records, sem_scaled, src, job, anchor):
    _WORKER["records"] = records
    _WORKER["sem"] = sem_scaled
    _WORKER["src"] = src
    _WORKER["job"] = job
    _WORKER["anchor"] = anchor


def _score_range(rng: tuple[int, int]) -> list[dict]:
    """Score records[start:end] using fork-inherited globals. Returns rows WITHOUT
    the raw record (the parent re-attaches it for the top-100 only)."""
    start, end = rng
    records = _WORKER["records"]
    sem = _WORKER["sem"]
    src = _WORKER["src"]
    job = _WORKER["job"]
    anchor = _WORKER["anchor"]
    out = []
    for i in range(start, end):
        rec = records[i]
        blob = F.Blob(rec)
        out.append(_score_one(rec, blob, job, float(sem[i]), src, anchor, keep_rec=False))
    return out


def score_pool_parallel(
    records: list[dict],
    job: jd.JobSpec,
    semantic_raw: np.ndarray | None,
    anchor: date,
    n_workers: int = 0,
) -> list[dict]:
    """Parallel scorer. Splits the pool across CPU cores via the 'fork' start method
    so workers share the records/semantic arrays without pickling them. Falls back
    to single-process scoring if fork isn't available or anything goes wrong, so the
    result is identical either way (determinism is preserved by index order)."""
    import multiprocessing as mp
    import os

    n = len(records)
    if n_workers <= 0:
        n_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    if n_workers == 1 or n < 2000:
        return score_pool(records, job, semantic_raw, anchor, keep_rec=True)

    # Semantic must be scaled over the FULL pool before splitting.
    blobs_for_sem = None
    if semantic_raw is None:
        blobs_for_sem = [F.Blob(r) for r in records]
    sem_scaled, src = _semantic_array(records, blobs_for_sem, job, semantic_raw)

    try:
        ctx = mp.get_context("fork")
    except (ValueError, RuntimeError):
        return score_pool(records, job, semantic_raw, anchor, keep_rec=True)

    _setup_workers(records, sem_scaled, src, job, anchor)
    step = (n + n_workers - 1) // n_workers
    ranges = [(s, min(s + step, n)) for s in range(0, n, step)]
    try:
        with ctx.Pool(processes=n_workers) as pool:
            chunks = pool.map(_score_range, ranges)
    except Exception:
        return score_pool(records, job, semantic_raw, anchor, keep_rec=True)
    finally:
        _WORKER.clear()
    return [row for chunk in chunks for row in chunk]


def rank_pool(rows: list[dict], top_k: int = 100) -> list[dict]:
    """Select the top_k by final_score desc, tie-break by candidate_id ascending.

    Ranks are NOT assigned here — final ranks are set in rescale_scores AFTER the
    CSV score is rounded, so that any ties introduced by rounding still satisfy the
    validator's "equal score -> candidate_id ascending" rule.
    """
    ordered = sorted(rows, key=lambda r: (-r["final_score"], r["candidate_id"]))
    return ordered[:top_k]


def rescale_scores(top: list[dict]) -> list[dict]:
    """Round to the CSV score, then re-sort/rank to guarantee validator invariants.

    1. Map final_score to a readable (0,1] range (best ~0.99, worst-of-top ~0.30).
    2. Round to 4 decimals.
    3. Re-sort by (-rounded_score, candidate_id ascending) and assign ranks 1..k.

    Step 3 is the key correctness step: rounding can collapse two distinct raw
    scores into an equal CSV score, and the validator then requires those tied rows
    to be in candidate_id order. Sorting on the *rounded* score makes that
    automatic. Returns the finalized, rank-ordered list.
    """
    if not top:
        return top
    raw = np.array([r["final_score"] for r in top], dtype=np.float64)
    lo, hi = raw.min(), raw.max()
    if hi <= lo:
        scaled = np.full_like(raw, 0.5)
    else:
        scaled = 0.30 + 0.69 * (raw - lo) / (hi - lo)
    for r, s in zip(top, scaled):
        r["score"] = round(float(s), 4)

    top_sorted = sorted(top, key=lambda r: (-r["score"], r["candidate_id"]))
    for i, r in enumerate(top_sorted):
        r["rank"] = i + 1
    return top_sorted
