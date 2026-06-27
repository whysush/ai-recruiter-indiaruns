# Deck outline — AI Recruiter (convert to PDF for submission)

Each bullet ≈ one slide. Keep it plain and specific.

## 1. The problem
- Recruiters skim hundreds of profiles and still miss the right person — keyword
  filters can't see what actually matters.
- Goal: rank candidates for Redrob's *Senior AI Engineer — Founding Team* role the
  way a great recruiter would — by understanding **fit**, from a noisy 100K pool.

## 2. What we learned from the data FIRST (before writing the scorer)
- 100K candidates; genuine AI/ML titles are a few hundred — a needle-in-haystack.
- **Skills are near-uniform noise:** every skill appears on ~12% of candidates.
  → "most AI keywords" is literally ranking noise — and the JD says so.
- Real signal lives in: **titles, career-history descriptions, summaries, and
  measured `skill_assessment_scores`.**
- The dataset embeds traps: keyword stuffers, "transitioning-into-ML" aspirationals,
  and ~80 impossible-profile honeypots (forced to relevance tier 0).

## 3. Why keyword / BM25 matching fails (before/after)
- **Before:** `CAND_0000001`, a Data Engineer whose summary says *"building
  competence on the ML side… transitioning toward AI/ML,"* stuffed with NLP /
  Fine-tuning / Speech / TTS skills → keyword search ranks it near the top.
- **After (ours):** down-ranked — weak title/career evidence, **no assessment
  scores** backing the AI claims, and three honeypot flags (aspirational summary;
  CV/speech without NLP/IR; AI skills with no supporting role).

## 4. Our approach — one unified pipeline (demo == submission)
```
final_score = (0.40·semantic_fit + 0.60·evidence_fit)   # fused fit
              · behavioral_envelope                       # [0.5, 1.2]
              · location_shaper                           # tiny, bona-fide hybrid role
              − honeypot_penalty                          # [0, 0.6]
```
- **Deep job understanding:** the JD is decomposed into must-haves, explicit
  disqualifiers (consulting-only, CV/speech, pure-research, framework-only,
  title-chasing) and an *ideal-profile* paragraph written from "how to read between
  the lines."
- **Contextual relevance (semantic):** local sentence-embedding cosine between that
  ideal profile and each candidate — sees meaning, rescues buzzword-free fits.
- **Signal integration (evidence):** title relevance, duration-weighted career
  relevance, and skill evidence **validated by measured assessments**; plus a
  bounded behavioral envelope using availability / reliability / demand.
- **Honeypot audit:** 9 consistency rules that compare claims against evidence.

## 5. Fairness by design
- Never ranked on: name, education tier/grade/institution, gender, age, graduation
  year, nationality. A unit test enforces it.
- City / relocation-willingness used only as a small shaper — the role is an
  explicitly hybrid Pune/Noida position (bona-fide requirement, not a proxy).

## 6. Reproducibility & compute
- Ranking step: **numpy-only, CPU, no network, ≤5 min** — embeddings are a declared
  one-time precompute cached to a numpy artifact.
- Deterministic: same input → same CSV. Graceful fallback if embeddings absent.
- One command: `python scripts/run_submission.py --candidates … --out submission.csv`.

## 7. Honesty / what we did NOT do
- No LLM in the scored path; no vector DB (a single dot-product over 100K is
  instant); no learning-to-rank (no labels). Every deck claim exists in the code.

## 8. Results to show in the talk
- Top-10 list with one-line faithful reasoning each.
- A honeypot / keyword-stuffer shown ranked low, with its flags.
- Runtime numbers; validator output "Submission is valid."
