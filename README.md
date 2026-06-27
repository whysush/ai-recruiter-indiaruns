# AI Recruiter — Ranked Candidate Shortlisting

A reproducible, **offline, CPU-only** system that ranks the top-100 candidates from
a 100,000-profile pool for Redrob's *"Senior AI Engineer — Founding Team"* job
description — the way a careful recruiter would, by reading the **evidence** in a
profile rather than counting keywords.

> Built for the *Intelligent Candidate Discovery & Ranking Challenge*. The thing we
> demo **is** the thing that produces the submission CSV — there is no second,
> unused subsystem.

---

## TL;DR — how it works

```
final_score = ( 0.40 · semantic_fit  +  0.60 · evidence_fit )   # fused fit, both in [0,1]
              · behavioral_envelope                              # bounded multiplier [0.5, 1.2]
              · location_shaper                                  # tiny [~0.96, 1.0], bona-fide hybrid-role logistics
              − honeypot_penalty                                 # [0, 0.6]
```

Each term is a named module, computed by pure arithmetic, so every line of the
reasoning we hand a recruiter is faithful to a number the code actually produced.

| Term | Module | What it captures |
|---|---|---|
| **semantic_fit** | `embeddings.py` + `score.py` | Cosine similarity between an *ideal-candidate* description (written from the JD) and each candidate's combined text, via a **local** sentence-embedding model. Sees *meaning*, so it rescues candidates who did the work without the buzzwords and resists pure keyword overlap. |
| **evidence_fit** *(dominant)* | `features.py` | Title relevance, **duration-weighted** career relevance, skill evidence **validated by `skill_assessment_scores`**, and seniority alignment to the JD's 6–8 yr ideal. |
| **behavioral_envelope** | `features.py` | Availability (recruiter response rate, notice period, recency), reliability (interview/offer rates), demand (recruiter saves/searches). A *nudge*, never a driver. |
| **honeypot_penalty** | `honeypot.py` | A consistency audit that sinks impossible profiles, keyword stuffers, and "transitioning-into-ML" aspirational false-positives. |

---

## Why this beats keyword matching (a concrete before/after)

We inspected the real data first. Two findings shaped the whole design:

1. **Skills are near-uniform noise** — every skill appears on ~12% of the pool, so
   *skill presence carries almost no signal*. A ranker that rewards "most AI skills
   listed" is ranking noise. The JD says this explicitly: *"The right answer is not
   find candidates whose skills section contains the most AI keywords. That's a trap
   we've explicitly built into the dataset."*
2. **The discriminative signal lives in titles, career-history descriptions,
   summaries, and `skill_assessment_scores`** (measured, not self-claimed).

**Before (keyword/BM25):** `CAND_0000001` — a Backend/Data Engineer whose summary
reads *"I'm building competence on the ML side… interested in transitioning toward
AI/ML"* and whose skill list is stuffed with `NLP (advanced)`, `Fine-tuning LLMs
(advanced)`, `Speech Recognition`, `TTS`. Token-dense → keyword search ranks it
near the top.

**After (this system):** it is **down-ranked** because
- `evidence_fit` is low: title is *Backend Engineer*, career descriptions are data
  pipelines, and there are **no relevant assessment scores** to back the AI claims;
- the **honeypot audit** fires three flags: *aspirational summary*, *CV/speech
  expertise without NLP/IR* (an explicit JD negative), and *AI skills with no
  supporting role or assessment*.

Meanwhile a *plain-language* candidate whose description says *"built a
recommendation and search ranking system with embeddings and FAISS at a product
company"* scores high on **both** semantic and evidence fit — even if they never
wrote the word "RAG".

---

## Repository layout

```
airecruiter/
  airecruiter/
    dataio.py       # load json / jsonl / .gz, validate ids, build candidate text
    jd.py           # decompose the ONE target JD -> requirements + ideal-profile text
    embeddings.py   # local model wrapper + precomputed embedding cache (numpy at rank time)
    features.py     # evidence features + bounded behavioral envelope
    honeypot.py     # consistency / honeypot audit -> penalty + human-readable flags
    score.py        # combine terms, rank, tie-break, rescale
    reasoning.py    # faithful 1-2 sentence justifications from real numbers
    fairness.py     # blocked-feature constants + guard used by the test suite
  scripts/
    precompute_embeddings.py   # ONE-TIME: embed the pool, cache to artifacts/ (may use network/GPU-free, >5min OK)
    run_submission.py          # TIMED ranking step: numpy-only, no network, writes submission.csv
    validate_submission.py     # the official challenge validator (unmodified)
  tests/            # fairness, honeypot behavior, output format
  data/             # JD text + dataset (dataset gitignored)
  artifacts/        # cached embeddings (gitignored; regenerate with precompute)
  requirements.txt
  submission_metadata.yaml
```

---

## How to reproduce

### 0. Install
The **ranking step needs only numpy**. The one-time precompute needs the embedding
model (isolated in a virtualenv so it never pollutes your system Python):

```bash
python3 -m virtualenv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy + sentence-transformers (CPU torch)
```

### 1. One-time pre-computation (declared; may exceed 5 min, downloads model once)
```bash
python scripts/precompute_embeddings.py \
    --candidates data/candidates.jsonl \
    --cache artifacts/
```
Produces `artifacts/cand_embeddings.npy` (float16, ~77 MB), `artifacts/cand_ids.txt`,
and `artifacts/ideal_vec.npy`. This is the **declared pre-computed artifact**.

### 2. The timed ranking step (≤5 min, CPU, 16 GB, **no network**)
```bash
python scripts/run_submission.py \
    --candidates data/candidates.jsonl \
    --out submission.csv
```
If `artifacts/` is missing, the system **degrades gracefully** to a deterministic
lexical semantic proxy and still produces a valid CSV — it never requires the model
at rank time.

### 3. Validate
```bash
python scripts/validate_submission.py submission.csv   # -> "Submission is valid."
```

### Tests
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
*(The env var only sidesteps unrelated third-party pytest plugins that may be
installed system-wide; it is not needed in a clean environment.)*

---

## Design decisions, honestly

- **Weighted sum, not pure RRF.** Our components are calibrated to `[0,1]`, and we
  want the *magnitude* of measured evidence (e.g. an NLP assessment of 82/100) to
  move the score, not only its rank. Evidence is weighted above semantic because raw
  embedding similarity still rewards keyword density — the exact failure mode we
  must avoid — so evidence acts as the grounding and semantics as the booster.
- **Embeddings are precomputed and cached.** The only slow step is embedding 100K
  profiles; we do it once and ship a numpy artifact, so the reproduced ranking step
  is trivially within the 5-minute CPU budget and needs no torch and no network.
- **Recency is anchored to `max(last_active_date)`** in the pool (the data is
  synthetic/stale), so "inactive" penalties are measured against the dataset's own
  clock, not the wall clock.
- **Fairness by design.** `anonymized_name`, education `tier`/`grade`/`institution`,
  gender, age, graduation year, and nationality are **never** ranking signals; a
  test (`tests/test_fairness.py`) asserts the scored modules don't reference them.
  City / relocation-willingness *is* used as a small shaper because the role is an
  explicitly hybrid Pune/Noida position — a bona-fide occupational requirement, not
  a demographic proxy.

### What is and isn't implemented (no vaporware)
**Implemented:** local-embedding semantic fit with disk cache; assessment-validated
skill evidence; duration-weighted career relevance; bounded behavioral envelope;
a 9-rule honeypot/consistency audit with human-readable flags; faithful,
rank-consistent reasoning; the official format validator; fairness/honeypot/format
tests. **Not implemented (and not claimed):** no LLM calls anywhere in the scored
path; no vector database (a single batch dot-product over 100K rows is instant); no
learning-to-rank training (no labels are provided); no online/A-B components.

---

## Runtime
- Structured feature pass over the full 100K pool: a few seconds on CPU.
- Ranking step end-to-end (with cached embeddings): well under the 5-minute budget.
- One-time embedding precompute: minutes on CPU (declared, cached).
