Team Name : xyzcodes
Problem Statement : Intelligent Candidate Discovery & Ranking Challenge — from a noisy
100,000-profile pool, produce a trustworthy top-100 ranked shortlist for Redrob's
"Senior AI Engineer — Founding Team" job description: not keyword filtering, but
genuinely understanding who fits the role.
Team Leader Name : Garvit Sahni


## Solution Overview

**What is your proposed solution?**
A structured-evidence ranker with one genuine semantic component, unified so that the
thing we demo IS the thing that produces the submission CSV. Every candidate gets an
interpretable score:

    final_score = (0.40 · semantic_fit + 0.60 · evidence_fit)   # fused fit, both in [0,1]
                  · behavioral_envelope                          # bounded multiplier [0.5, 1.2]
                  · location_shaper                              # tiny [~0.96, 1.0]
                  − honeypot_penalty                             # consistency audit [0, 0.6]

- semantic_fit: cosine similarity between a JD-derived "ideal candidate" description and
  each candidate's combined text, using a local sentence-embedding model (all-MiniLM-L6-v2),
  precomputed once and cached. This sees meaning, not keywords.
- evidence_fit (the dominant term): title relevance, duration-weighted career relevance,
  and skill evidence validated by MEASURED skill_assessment_scores, plus seniority alignment.
- behavioral_envelope: a bounded availability/reliability/demand multiplier — it nudges,
  it never dominates.
- honeypot_penalty: a 9-rule consistency audit that sinks impossible profiles, keyword
  stuffers and "transitioning-into-ML" aspirationals.

**What differentiates your approach from traditional candidate matching systems?**
We inspected the data first and found the trap the organizers built in: skills are
near-uniform noise — every skill appears on ~12% of the pool — so "most AI keywords"
literally ranks noise. Traditional keyword/BM25 matching therefore promotes the worst
candidates. Three differentiators:
1. We trust MEASURED competence (per-skill assessment scores) over self-claimed skill
   lists. A claimed "advanced NLP" only counts to the extent the platform assessment backs it.
2. We read the gap between what the JD says and what it means: a real semantic model
   rewards a profile that "built a recommendation system at a product company" even with
   zero buzzwords, and resists a buzzword-dense profile whose career shows none of it.
3. An explicit honeypot/consistency audit — a measurable anti-cheat layer, not a claim.
Concretely: the aspirational data-engineer "transitioning into ML" (CAND_0000001) that
BM25 ranks near the top lands at rank 57,835 / 100,000 in our system.


## JD Understanding & Candidate Evaluation

**Key requirements extracted from the JD (Senior AI Engineer — Founding Team):**
- Must-haves: production embeddings-based retrieval (sentence-transformers/BGE/E5),
  vector / hybrid search infra (FAISS, Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch,
  OpenSearch), strong Python, and rigorous ranking-evaluation experience (NDCG, MRR, MAP,
  A/B testing).
- Seniority: 5–9 years, ideal 6–8, with 4–5 in applied ML at PRODUCT (not services) companies;
  must have shipped an end-to-end ranking/search/recommendation system at scale.
- Nice-to-have: LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank, HR-tech, scale/inference,
  open-source.
- Explicit disqualifiers (we encode each as a negative signal): consulting-only careers
  (TCS/Infosys/Wipro/…), CV/speech/robotics-primary without NLP/IR, pure research without
  production, recent-LangChain-only "AI experience", title-chasing job-hoppers, and
  architects/tech-leads who stopped writing code.
- Logistics: Pune/Noida hybrid (also Hyderabad/Mumbai/Delhi NCR/Bangalore), sub-30-day notice
  preferred; an inactive, unresponsive candidate is "not actually available."

**Which signals matter most / how we evaluate fit beyond keywords:**
Strongest → weakest: (1) skill_assessment_scores on role-relevant skills — a measured number,
the best anti-honeypot signal; (2) current + historical TITLES (the decisive anti-stuffer
signal); (3) career-history DESCRIPTIONS judged semantically and by concept coverage; (4)
the summary (aspirational language is a NEGATIVE); (5) behavioral signals for availability.
Listed skills alone are deliberately down-weighted because the data proves they are noise.


## Ranking Methodology

**How the system retrieves, scores, and ranks:**
Single-pass batch scoring over all 100K candidates (no ANN index needed — one dot-product
over a 100K×384 matrix is instant). For each candidate we compute the four components above,
fuse them, sort descending, and take the top 100. Ties break deterministically by
candidate_id ascending; final scores are rescaled to a clean, strictly non-increasing range.

**Models / algorithms / heuristics:**
- Semantic: all-MiniLM-L6-v2 (local, CPU, 384-dim), cosine similarity to a hand-authored
  ideal-profile text derived from the JD's "how to read between the lines" section.
- Evidence: interpretable weighted features (title, duration-weighted career relevance,
  assessment-validated skill evidence, seniority Gaussian-style band) — weights sum to 1.0.
- Honeypot audit (9 rules), e.g.: a skill used longer than the whole career; multiple
  "expert" skills with 0 months/endorsements; AI skills claimed with no supporting role or
  assessment; CV/speech-primary without NLP/IR; consulting-only career; aspirational summary.
- Behavioral envelope: response rate, notice period, last-active recency (anchored to the
  pool's max last_active_date = 2026-05-27), interview/offer completion, recruiter saves.

**How signals are combined:**
Weighted fusion (not pure RRF) so the magnitude of measured evidence moves the score, not
just its rank. Evidence is weighted above semantic because raw embedding similarity still
rewards keyword density — evidence grounds it, semantics is the "see-beyond-keywords"
booster, behavioral signals fine-tune among equally-qualified candidates, and the honeypot
penalty is subtracted last.


## Explainability & Data Validation

**How ranking decisions are explained:**
Every candidate's 1–2 sentence reasoning is generated deterministically from that candidate's
OWN computed sub-scores and raw fields — e.g. "Strong fit: 7.6 yrs as Search Engineer; 4
ML/data-oriented roles; assessments: Milvus 76/100, PyTorch 72/100; responsive (94%);
45-day notice; profile text aligns closely with the role." Tone is graded by the candidate's
fit (Strong/Good/Plausible/Borderline), and genuine concerns are stated honestly
(out-of-band experience, honeypot flags, low response rate).

**How we prevent hallucinations / unsupported justifications:**
The reasoning is template-assembled from real numbers only — it never calls an LLM and can
only quote values that exist in the profile (years, current title, assessment scores,
response rate, notice days). There is no free-text generation step that could invent a skill
or employer. Result: 100/100 reasonings are distinct, ≤2 sentences, and zero contradict the
data. A fairness unit test also asserts the scorer never reads name/education-tier/grade/etc.

**How we handle inconsistent, low-quality, or suspicious profiles:**
The honeypot/consistency audit flags them and applies a bounded penalty, surfaced as a
human-readable caveat in the reasoning. In our full run, 0 of the top-100 trip the audit
(the challenge disqualifies submissions with >10% honeypot rate in the top-100).


## End-to-End Workflow

1. (One-time, declared) Pre-compute embeddings for all 100K candidates with the local model
   and cache them to a 76 MB float16 .npy artifact (committed to the repo).
2. Load candidates.jsonl (streamed) + the JD; decompose the JD into requirements + ideal text.
3. Load the cached embeddings; compute cosine-to-ideal (numpy only — no model at rank time).
4. Score every candidate in parallel across CPU cores (deterministic, fork-based): evidence
   features + behavioral envelope + honeypot audit, fused with semantic similarity.
5. Sort, tie-break by candidate_id, take top 100, rescale scores, generate faithful reasoning.
6. Write submission.csv and self-run the official validator → "Submission is valid."


## System Architecture

    candidates.jsonl (100K)        job_description.txt
            |                            |
            v                            v
      [dataio]                       [jd] decompose -> requirements + ideal-profile text
            |                            |
            |                  (one-time) [embeddings] all-MiniLM-L6-v2 -> artifacts/*.npy (cached)
            |                            |
            +-------------+--------------+
                          v
        RANK STEP (numpy-only, CPU, no network, parallel)
          [features] title / career / skill(assessment-validated) / seniority / behavioral
          [honeypot] 9-rule consistency audit -> penalty + flags
          [score]    fuse -> envelope -> -penalty -> sort -> tie-break -> rescale
          [reasoning] faithful 1-2 sentence justification from real numbers
                          |
                          v
              submission.csv  ->  validate_submission.py (official)


## Results & Performance

**Ranking quality:**
- Top-100 is entirely genuine AI/ML titles (Applied ML / Recommendation / AI / NLP / Search
  Engineers, Data Scientists), led by candidates with the strongest measured vector-DB and
  retrieval assessments — no HR Managers, Mechanical Engineers, or keyword stuffers.
- 0% honeypot-audit rate in the top-100 (limit: >10% disqualifies).
- Score spread 0.97 → 0.30 across the 100 (it differentiates; not all-equal).
- Mean experience 7.6 yrs (vs the JD's 6–8 ideal); only 2 juniors and 9 outside the 5–9 band.
- Before/after: a keyword-stuffer that BM25 ranks highly (CAND_0000001) is correctly buried
  at rank 57,835 / 100,000.
- Reasoning: 100/100 distinct, ≤2 sentences, none contradicting the data.

**Meeting the runtime & compute constraints (measured):**
| Constraint | Limit | Ours |
| Total runtime | ≤ 5 min wall-clock | ~63 s (full 100K) |
| Memory | ≤ 16 GB RAM | ~2.2 GB peak |
| Compute | CPU only, no GPU | numpy-only; runs on a Python with no torch installed |
| Network | Off | no API/network calls in the rank path |
| Disk | ≤ 5 GB intermediate | 76 MB embedding cache |
Embeddings are a one-time precompute (~20 min, spec-exempt); the timed ranking step needs
only numpy, so the no-network Stage-3 reproduction is trivially within budget.


## Technologies Used

- Python 3.10+ (3.11 in the Docker image).
- numpy — the entire ranking step (feature math + cached-embedding cosine). Chosen so the
  timed path is dependency-light, fast, and trivially reproducible offline.
- sentence-transformers + PyTorch (CPU) — used ONLY in the one-time precompute to embed the
  pool with all-MiniLM-L6-v2 (small, fast, well-understood, strong for short-text retrieval).
  Not imported at rank time.
- Python multiprocessing (fork) — deterministic parallel scoring across CPU cores.
- pytest — fairness / honeypot / format / parallel-determinism tests.
- Docker — a self-contained sandbox image (numpy-only) that runs the real rank step on a
  small sample, offline (spec §10.5).
No vector database, no learning-to-rank training (no labels provided), and no hosted LLM —
deliberately, and honestly stated.


## Submission Assets

- GitHub repository: the full source (clean, incremental git history), README, tests,
  Dockerfile, requirements.txt / pyproject.toml, submission_metadata.yaml, and the committed
  embedding artifacts so the no-network reproduction works.   [repo URL: TODO add after push]
- submission.csv — the top-100 ranking (rename to <participant_id>.csv before upload).
- Sandbox: `docker build -t airecruiter . && docker run --rm airecruiter` runs the ranker on
  a bundled ≤100-candidate sample, CPU-only and offline, in seconds.
- (Optional) demo video / deck: see DECK_OUTLINE.md for the slide-by-slide outline.   [TODO]
