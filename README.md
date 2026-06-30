# AI Recruiter — ranked candidate shortlisting

This is our entry for the Redrob *Intelligent Candidate Discovery & Ranking Challenge*.
The job is to take a 100,000-profile pool and pull out the 100 people who actually fit
the "Senior AI Engineer — Founding Team" role, ranked best-first. It runs offline on a
CPU and the whole thing is deterministic, so the same input always gives the same CSV.

One thing we cared about from the start: the demo *is* the submission pipeline. There's
no separate impressive-looking app sitting next to a crude scorer. What produces
`submission.csv` is exactly what we'd defend in an interview.

## The short version of how it scores

```
final_score = ( 0.40 * semantic_fit  +  0.60 * evidence_fit )   # the "fit", both 0..1
              * behavioral_envelope                              # availability multiplier, 0.5..1.2
              * location_shaper                                  # tiny nudge for the hybrid role
              - honeypot_penalty                                 # 0..0.6, for dodgy profiles
```

Every term is its own module and it's all plain arithmetic, which matters because the
one-line reasoning we attach to each candidate is built from those same numbers — we
never write a sentence the code can't back up.

- **semantic_fit** (`embeddings.py`, `score.py`) — cosine similarity between an
  "ideal candidate" paragraph we wrote from the JD and each candidate's own text, using a
  local sentence-transformer. This is what lets us catch someone who clearly built the
  right systems but never used the trendy words.
- **evidence_fit** (`features.py`) — the heavier term. Title relevance, how much of the
  career (weighted by duration) is genuinely relevant, and skill claims *checked against
  the measured `skill_assessment_scores`*, plus a seniority check against the 6–8 year
  sweet spot.
- **behavioral_envelope** (`features.py`) — response rate, notice period, how recently
  they logged in, interview/offer follow-through. It only nudges; it can't carry a weak
  candidate.
- **honeypot_penalty** (`honeypot.py`) — a consistency audit that drops impossible
  profiles, keyword stuffers, and the "I'm transitioning into ML" crowd.

## Why we didn't just match keywords

We looked at the data before writing any scoring code, and two things basically decided
the whole design.

First, the skills lists are noise. Every skill shows up on roughly 12% of the pool, so
"has the most AI skills" is, almost literally, a random ranking. The JD even warns you
about this — it says finding the candidates with the most AI keywords is a trap they
built into the dataset on purpose. So we lean on the stuff that's hard to fake: titles,
what people actually wrote about their roles, their summaries, and the assessment scores
(those are measured on the platform, not self-reported).

Here's the example we keep coming back to. `CAND_0000001` is a backend/data engineer
whose summary says things like *"building competence on the ML side… interested in
transitioning toward AI/ML,"* and the skills list is packed with NLP, Fine-tuning LLMs,
Speech Recognition, TTS. A keyword/BM25 ranker loves this profile and floats it near the
top. Our system drops it to **rank 57,835 / 100,000** — the career relevance comes out at
0.11 because the actual role descriptions are data pipelines, and the honeypot audit adds
a 0.30 penalty for the aspirational summary and the CV/speech skills (which the JD calls
out as a negative for this role).

The flip side works too: a profile that plainly says *"built a recommendation and search
ranking system with embeddings and FAISS at a product company"* scores well on both
semantic and evidence fit even if it never says "RAG". In our full run the top 100 came
out as entirely real AI/ML titles, and **none** of them tripped the honeypot audit (the
challenge disqualifies you above a 10% honeypot rate in the top 100).

## What's in here

```
airecruiter/
  airecruiter/
    dataio.py       # load json / jsonl / .gz, check ids, build the text blob per candidate
    jd.py           # turn the JD into requirements + the ideal-profile text
    embeddings.py   # local model wrapper + the cached embeddings (rank time is numpy only)
    features.py     # the evidence features + the behavioral envelope
    honeypot.py     # the consistency / honeypot audit -> penalty + readable flags
    score.py        # put it together, rank, break ties, rescale
    reasoning.py    # the 1-2 sentence justification, built from real numbers
    fairness.py     # the blocked-feature list + the guard the tests use
  scripts/
    precompute_embeddings.py   # one-time: embed the pool, cache to artifacts/
    run_submission.py          # the timed ranking step (numpy only) -> submission.csv
    validate_submission.py     # the official validator, copied in as-is
  tests/            # fairness, honeypot behaviour, output format
  data/             # JD text + schema (the big candidates file is NOT committed)
  artifacts/        # the cached embeddings (committed, ~76 MB)
```

## Running it

You need the candidate pool first — we don't commit it because it's ~487 MB and it's the
released dataset anyway. Drop it in `data/`:

```bash
data/candidates.jsonl          # candidates.jsonl.gz works too
```

The embeddings in `artifacts/` are looked up by `candidate_id`, so they match the released
file no matter what order it's in. The JD text is already in the repo.

The ranking step only needs numpy. The one-time embedding step needs the model, and that
one we keep in a virtualenv so it doesn't touch your system Python:

```bash
python3 -m virtualenv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy + sentence-transformers (CPU torch)
```

**One-time precompute** (downloads the model once, can take longer than 5 min — that's
fine, the spec exempts precompute):

```bash
python scripts/precompute_embeddings.py --candidates data/candidates.jsonl --cache artifacts/
```

That writes `cand_embeddings.npy`, `cand_ids.txt` and `ideal_vec.npy` into `artifacts/`.
Since those are already committed, you can skip this unless you want to regenerate them.

**The actual ranking step** — this is the part that has to stay under 5 minutes, CPU only,
no network:

```bash
python scripts/run_submission.py --candidates data/candidates.jsonl --out submission.csv
```

It re-runs the official validator at the end and prints "Submission is valid." If the
`artifacts/` cache happens to be missing, it falls back to a deterministic lexical proxy
so it still produces a valid CSV — it never needs the model at rank time.

**Validate on its own** if you want:

```bash
python scripts/validate_submission.py submission.csv
```

**Tests:**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

(The env var is only there because this machine has some unrelated ROS pytest plugins that
crash collection. You don't need it in a clean environment.)

**Docker sandbox** (this is our answer to the §10.5 sandbox requirement). Because the
embeddings are committed, the image is numpy-only — no torch, no network:

```bash
docker build -t airecruiter .
docker run --rm airecruiter            # ranks the bundled 50-candidate sample in seconds
# or point it at your own <=100-candidate file:
docker run --rm -v "$PWD/mysample.json:/app/mysample.json" airecruiter \
    python scripts/run_submission.py --candidates mysample.json --cache artifacts/ --out out.csv
```

## A few design calls worth explaining

**Why a weighted sum and not RRF.** We wanted the actual magnitude of the evidence to
matter, not just the rank order. An NLP assessment of 82/100 should push a candidate up,
not just nudge their position. Evidence is weighted above semantic on purpose: raw
embedding similarity still rewards keyword density, which is the exact thing we're trying
not to fall for, so evidence does the grounding and the embedding is the booster on top.

**Why precompute the embeddings.** Embedding 100K profiles is the only genuinely slow
part. Do it once, cache it as a numpy array, and the ranking step that produces the CSV is
trivially inside the 5-minute budget and doesn't need torch or a network connection at all.

**The "today" date is the pool's own clock.** The data is synthetic and a bit stale, so we
anchor recency to the latest `last_active_date` in the pool (2026-05-27) instead of the
real wall clock — otherwise everyone looks inactive.

**Fairness.** We never score on name, education tier/grade/institution, gender, age,
graduation year, or nationality — and there's a test (`test_fairness.py`) that actually
greps the scoring modules to make sure none of those sneak in. We do use city /
willingness-to-relocate as a small factor, but only because this is an explicitly hybrid
Pune/Noida role, so it's a real job requirement rather than a proxy for anything.

## What we built vs. what we didn't (being honest)

Built: the local-embedding semantic term with a disk cache, assessment-validated skill
evidence, duration-weighted career relevance, the bounded behavioral multiplier, a 9-rule
honeypot/consistency audit with readable flags, reasoning that's generated from each
candidate's real numbers, and the fairness/honeypot/format tests.

Didn't build, and aren't claiming: there's no LLM anywhere in the scored path, no vector
database (one dot product over 100K rows is instant, so an index would be overkill), no
learning-to-rank model (there are no labels to train on), and nothing online/A-B.

## Timings we actually measured

Run on a 12-core CPU box, system Python, numpy only:

- The ranking step is about **61 seconds** for the full 100K — roughly 4s to load, the
  rest is the parallel scoring plus the embedding cosine. Well under the 5-minute limit,
  no network, no torch.
- The one-time embedding precompute is around 20 minutes for 100K on CPU. That's cached to
  the ~76 MB float16 artifact and, again, the spec doesn't count it against the ranking
  budget.
- The scorer is spread across cores with a fork-based pool, and we have a test that checks
  the parallel result is bit-for-bit identical to running it single-process.
