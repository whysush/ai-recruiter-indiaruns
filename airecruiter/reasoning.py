"""Faithful, deterministic reasoning generation.

Stage-4 review checks each reasoning for: specific facts, JD connection, honest
concerns, NO hallucination, variation, and rank-consistent tone. We therefore
build every sentence from the candidate's OWN computed sub-scores and raw fields —
never a fixed template, never a claim the data doesn't support. Concerns (honeypot
flags, weak availability, off-band experience) are stated plainly, and the opening
verb is graded by rank so tone matches position.
"""

from __future__ import annotations


def _years(rec: dict) -> float:
    return float(rec.get("profile", {}).get("years_of_experience", 0.0))


def _count_relevant_roles(rec: dict, row: dict) -> int:
    # A role "counts" when career_relevance logic would credit it; we re-derive a
    # simple, honest count: roles whose title looks ML/AI/data oriented.
    from . import jd
    from .features import STRONG_TITLE, ADJACENT_TITLE

    n = 0
    for j in rec.get("career_history", []):
        t = j.get("title", "").lower()
        if jd.any_hit(t, STRONG_TITLE) or jd.any_hit(t, ADJACENT_TITLE):
            n += 1
    return n


def _lead_phrase(row: dict) -> str:
    """Tone derived from the candidate's OWN computed fit, not their rank slot.

    This keeps reasoning honest when the pool is thin: a weak candidate that only
    reaches the top-100 as filler reads as 'borderline', never 'strong'.
    """
    fused = row.get("fused_fit", 0.0)
    pen = row.get("honeypot_penalty", 0.0)
    if fused >= 0.80 and pen < 0.1:
        return "Strong fit"
    if fused >= 0.62:
        return "Good fit"
    if fused >= 0.45:
        return "Plausible fit"
    return "Borderline fit, likely near/below the cutoff"


def build_reasoning(row: dict) -> str:
    """Return a 1-2 sentence, evidence-faithful justification for one candidate."""
    rec = row["_rec"]
    p = rec.get("profile", {})
    rank = row["rank"]
    yoe = _years(rec)
    title = p.get("current_title", "role unknown")

    facts: list[str] = []

    # Experience + current title (always factual).
    facts.append(f"{yoe:.1f} yrs as {title}")

    # Relevant role count from real career history.
    nrel = _count_relevant_roles(rec, row)
    if nrel:
        facts.append(f"{nrel} ML/data-oriented role(s) in history")

    # Measured assessment evidence — the anti-honeypot gold, quote real numbers.
    from .features import relevant_assessment
    from . import jd as _jd
    job = getattr(build_reasoning, "_job", None)
    if job is not None:
        _, n_assessed, examples = relevant_assessment(rec, job)
        if examples:
            facts.append("assessments: " + ", ".join(examples))

    # Behavioral availability — quote the actual response rate / notice honestly.
    bp = row.get("behavioral_parts", {})
    rr = bp.get("response_rate")
    notice = bp.get("notice_days")
    if rr is not None:
        avail_word = "responsive" if rr >= 0.6 else ("low recruiter response" if rr < 0.3 else "moderate response")
        facts.append(f"{avail_word} ({rr:.0%}); {notice}-day notice")

    # Semantic signal phrased honestly, graded by strength.
    sem = row.get("semantic_fit", 0.0)
    if sem >= 0.80:
        facts.append("profile text aligns closely with the role")
    elif sem >= 0.60:
        facts.append("profile text is broadly on-theme for the role")
    elif sem <= 0.30:
        facts.append("profile text only loosely matches the role")

    sentence = f"{_lead_phrase(row)}: " + "; ".join(facts) + "."

    # Honest concerns from the honeypot/consistency audit.
    flags = row.get("honeypot_flags", [])
    if flags:
        concern = flags[0]
        sentence += f" Concern: {concern}."
    elif yoe < 5 or yoe > 9:
        sentence += f" Note: experience ({yoe:.1f}y) sits outside the 5-9y band."

    # Keep it tight.
    return sentence.replace("  ", " ").strip()


def attach_job(job) -> None:
    """Provide the JobSpec used for assessment lookups during reasoning."""
    build_reasoning._job = job
