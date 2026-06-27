"""Honeypot & consistency audit — the biggest differentiator on this dataset.

The challenge embeds ~80 honeypots ("subtly impossible profiles") that are forced
to relevance tier 0 in the ground truth, plus a larger population of keyword
stuffers and aspirational false-positives. Ranking any of these highly is the
clearest sign a system is reading keywords, not profiles.

This module returns, per candidate:
  - penalty in [0, ~0.6]  (subtracted from the fused fit score)
  - flags: human-readable strings, surfaced in the reasoning as honest caveats

Every check compares a CLAIM against EVIDENCE. We never special-case ids; the
checks are general profile-consistency rules.
"""

from __future__ import annotations

from datetime import date

from . import jd
from .features import all_signal_text, career_text, titles_text


def _years(rec: dict) -> float:
    return float(rec.get("profile", {}).get("years_of_experience", 0.0))


def audit(rec: dict) -> tuple[float, list[str]]:
    flags: list[str] = []
    penalty = 0.0
    p = rec.get("profile", {})
    skills = rec.get("skills", [])
    sig = rec.get("redrob_signals", {})
    yoe = _years(rec)
    summary = p.get("summary", "") or ""
    career = career_text(rec)
    titles = titles_text(rec)
    all_text = all_signal_text(rec)

    # 1) IMPOSSIBLE skill duration: claims using a skill longer than the whole
    #    career (with a small grace). A hallmark honeypot construction.
    max_career_months = yoe * 12 + 18
    impossible_dur = [
        s.get("name", "?")
        for s in skills
        if s.get("duration_months", 0) > max_career_months
    ]
    if impossible_dur:
        penalty += 0.30
        flags.append(
            f"impossible skill tenure: {impossible_dur[0]} claimed "
            f"{max(s.get('duration_months',0) for s in skills)}mo vs {yoe:.0f}y career"
        )

    # 2) "expert" proficiency with no time/endorsement behind it (0 duration or
    #    0 endorsements on multiple expert claims) — classic fabricated profile.
    bogus_expert = [
        s for s in skills
        if s.get("proficiency") == "expert"
        and (s.get("duration_months", 0) == 0 or s.get("endorsements", 0) == 0)
    ]
    if len(bogus_expert) >= 2:
        penalty += 0.25
        flags.append(f"{len(bogus_expert)} 'expert' skills with 0 months/endorsements")
    elif len(bogus_expert) == 1:
        penalty += 0.08

    # 3) Career-duration vs stated experience mismatch: total months in history
    #    wildly exceeds plausible given years_of_experience (overlapping/fake).
    total_months = sum(max(0, j.get("duration_months", 0)) for j in rec.get("career_history", []))
    if yoe > 0 and total_months > (yoe * 12) * 1.8 + 24:
        penalty += 0.18
        flags.append(f"career tenure {total_months}mo implausible vs {yoe:.0f}y stated")

    # 4) Tenure that predates the company's own existence is not directly
    #    checkable (no founding date), but a role with is_current=True yet a
    #    non-null past end_date is internally inconsistent.
    for j in rec.get("career_history", []):
        if j.get("is_current") and j.get("end_date"):
            try:
                if date.fromisoformat(j["end_date"]) < date(2026, 5, 1):
                    penalty += 0.10
                    flags.append("role marked current but has a past end date")
                    break
            except Exception:
                pass

    # 5) ASPIRATIONAL summary: describes a future identity, not demonstrated work.
    #    This is the aspirational false-positive ("transitioning into ML").
    asp_hits = jd.count_hits(summary, jd.ASPIRATIONAL_PHRASES)
    if asp_hits >= 2:
        penalty += 0.18
        flags.append("summary is aspirational ('transitioning/learning'), not demonstrated")
    elif asp_hits == 1:
        penalty += 0.07

    # 6) AI CLAIM WITHOUT CAREER EVIDENCE: candidate lists core AI/ML skills but
    #    NO role in career history demonstrates them. The keyword-stuffer.
    relevant_terms: list[str] = []
    for terms in jd.CORE_ROLE_CONCEPTS.values():
        relevant_terms += terms
    for terms in jd.ML_PRODUCTION_CONCEPTS.values():
        relevant_terms += terms
    claims_ai_skill = any(jd.any_hit(s.get("name", ""), relevant_terms) for s in skills)
    career_shows_ai = jd.any_hit(career, relevant_terms) or jd.any_hit(titles, relevant_terms)
    assessed = sig.get("skill_assessment_scores", {}) or {}
    has_relevant_assessment = any(jd.any_hit(k, relevant_terms) for k in assessed)
    if claims_ai_skill and not career_shows_ai and not has_relevant_assessment:
        penalty += 0.22
        flags.append("AI skills listed but no career role or assessment supports them")

    # 7) CV/SPEECH/ROBOTICS primary without NLP/IR — JD explicitly down-ranks.
    cv_hits = jd.count_hits(all_text, jd.CV_SPEECH_ROBOTICS)
    nlp_ir_terms = jd.ML_PRODUCTION_CONCEPTS["nlp"] + jd.CORE_ROLE_CONCEPTS["retrieval"] + jd.CORE_ROLE_CONCEPTS["search"] + jd.CORE_ROLE_CONCEPTS["ranking"]
    nlp_hits = jd.count_hits(all_text, nlp_ir_terms)
    if cv_hits >= 2 and cv_hits > nlp_hits:
        penalty += 0.12
        flags.append("primary expertise looks CV/speech/robotics, not NLP/IR")

    # 8) CONSULTING-ONLY career (JD: not a fit unless prior product experience).
    companies = [j.get("company", "").lower() for j in rec.get("career_history", [])]
    companies.append(p.get("current_company", "").lower())
    companies = [c for c in companies if c]
    if companies:
        consult = sum(1 for c in companies if jd.any_hit(c, jd.CONSULTING_FIRMS))
        if consult == len(companies):
            penalty += 0.12
            flags.append("entire career at services/consulting firms")

    # 9) FRAMEWORK-ONLY recent AI (LangChain/OpenAI wrappers) with no pre-LLM ML.
    fw_hits = jd.count_hits(all_text, jd.FRAMEWORK_ENTHUSIAST)
    deep_ml = jd.count_hits(all_text, jd.ML_PRODUCTION_CONCEPTS["ml_engineering"] + jd.ML_PRODUCTION_CONCEPTS["deep_learning"])
    if fw_hits >= 1 and deep_ml == 0:
        penalty += 0.08
        flags.append("AI experience looks framework-wrapper only (no production ML)")

    # Cap the penalty so it stays explainable and bounded.
    penalty = min(penalty, 0.60)
    return penalty, flags
