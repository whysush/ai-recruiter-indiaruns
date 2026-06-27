"""Structured evidence features — the gold the dataset hands us.

Skills in this dataset are near-uniform noise (every skill appears on ~12% of
candidates), so skill *presence* is almost worthless. The discriminative signal
lives in: job TITLES, career-history DESCRIPTIONS, the SUMMARY, and — crucially —
`skill_assessment_scores`, which are *measured* competence rather than self-claim.
Every feature below returns a value in [0,1] (except the behavioral multiplier)
and is kept separate and inspectable so reasoning can quote it.
"""

from __future__ import annotations

import math
from datetime import date

from . import jd

PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.85, "expert": 1.0}


# --- text assembly --------------------------------------------------------

def career_text(rec: dict) -> str:
    parts = []
    for job in rec.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " \n ".join(p for p in parts if p)


def titles_text(rec: dict) -> str:
    parts = [rec.get("profile", {}).get("current_title", "")]
    parts += [j.get("title", "") for j in rec.get("career_history", [])]
    return " \n ".join(p for p in parts if p)


def all_signal_text(rec: dict) -> str:
    """Everything except blocked/identity fields — used for concept counting."""
    p = rec.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    parts.append(career_text(rec))
    parts += [s.get("name", "") for s in rec.get("skills", [])]
    return " \n ".join(p for p in parts if p)


class Blob:
    """Per-candidate lowercased text, built ONCE and reused across every feature and
    the honeypot audit. Lowercasing once here (instead of inside every match) is the
    single biggest scoring speedup at 100K scale."""

    __slots__ = ("summary_l", "career_l", "titles_l", "all_l", "roles_l",
                 "skill_names_l", "companies_l")

    def __init__(self, rec: dict):
        p = rec.get("profile", {})
        self.summary_l = (p.get("summary", "") or "").lower()
        self.titles_l = titles_text(rec).lower()
        self.career_l = career_text(rec).lower()
        skills = rec.get("skills", [])
        self.skill_names_l = [(s.get("name", "") or "").lower() for s in skills]
        self.all_l = " \n ".join(
            [p.get("headline", "").lower(), self.summary_l,
             p.get("current_title", "").lower(), self.career_l]
            + self.skill_names_l
        )
        self.roles_l = [
            (max(0, j.get("duration_months", 0)),
             ((j.get("title", "") + ". " + j.get("description", "")).lower()))
            for j in rec.get("career_history", [])
        ]
        comps = [j.get("company", "").lower() for j in rec.get("career_history", [])]
        comps.append(p.get("current_company", "").lower())
        self.companies_l = [c for c in comps if c]


# --- concept coverage -----------------------------------------------------

def concept_coverage_low(low: str, concept_groups: dict) -> float:
    """Fraction of concept GROUPS with at least one hit in already-lowercased text.

    Grouping prevents a candidate from scoring high just because one group (say
    'embeddings') has many synonyms — each conceptual requirement counts once.
    """
    if not concept_groups:
        return 0.0
    covered = 0
    for _name, terms in concept_groups.items():
        if jd.match_any(low, terms):
            covered += 1
    return covered / len(concept_groups)


def concept_coverage(text: str, concept_groups: dict) -> float:
    """Convenience wrapper that lowercases first (non-hot callers / tests)."""
    return concept_coverage_low(text.lower() if text else "", concept_groups)


# --- skill evidence, validated by measured assessment scores --------------

def relevant_assessment(rec: dict, job: jd.JobSpec) -> tuple[float, int, list[str]]:
    """Mean assessment score (0..1) over skills relevant to the role.

    Returns (mean_score_0_1, n_relevant_assessed, named_examples). This is the
    strongest anti-honeypot signal: a measured number on a role-relevant skill.
    """
    scores = rec.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    vals = []
    named = []
    for skill_name, sc in scores.items():
        if jd.match_any(skill_name.lower(), jd.CORE_PLUS_EVAL_FLAT):
            vals.append(max(0.0, min(100.0, float(sc))) / 100.0)
            named.append((skill_name, float(sc)))
    if not vals:
        return 0.0, 0, []
    named.sort(key=lambda x: -x[1])
    examples = [f"{n} {int(round(s))}/100" for n, s in named[:3]]
    return sum(vals) / len(vals), len(vals), examples


def skill_evidence(rec: dict, job: jd.JobSpec, blob: Blob) -> float:
    """Combine relevant-skill proficiency/endorsement/duration with assessment.

    A self-claimed 'advanced' skill is only believed to the extent the platform
    assessment backs it. With no relevant assessment, the claim is heavily
    discounted (claims are cheap; the data showed skills are random noise).
    """
    claim = 0.0
    for s, name_l in zip(rec.get("skills", []), blob.skill_names_l):
        if not jd.match_any(name_l, jd.MUSTHAVE_FLAT):
            continue
        prof = PROFICIENCY_WEIGHT.get(s.get("proficiency", ""), 0.4)
        endo = math.log1p(s.get("endorsements", 0)) / math.log1p(50)  # 50 endo ~ 1.0
        dur = min(s.get("duration_months", 0) / 48.0, 1.0)            # 4 yrs ~ 1.0
        claim += prof * (0.5 + 0.25 * min(endo, 1.0) + 0.25 * dur)
    claim_norm = min(claim / 4.0, 1.0)  # ~4 strong relevant skills saturates

    assess, n_assessed, _ = relevant_assessment(rec, job)
    if n_assessed == 0:
        # Unvalidated claims are worth little on their own.
        return 0.35 * claim_norm
    # Measured competence dominates; claims add a little.
    return min(1.0, 0.75 * assess + 0.25 * claim_norm)


# --- title relevance: the decisive anti-stuffer signal --------------------

# Genuine product-ML role tokens vs. unrelated function tokens. The point of this
# feature is that a "Marketing Manager" with a perfect AI skill list is NOT a fit.
STRONG_TITLE = [
    "ml engineer", "machine learning engineer", "applied ml", "applied scientist",
    "ai engineer", "ai research engineer", "research engineer", "data scientist",
    "search engineer", "recommendation", "recommender", "relevance engineer",
    "nlp engineer", "ml scientist", "research scientist (ml", "ranking",
]
ADJACENT_TITLE = [
    "data engineer", "analytics engineer", "backend engineer", "software engineer",
    "ml", "data", "platform engineer", "ai specialist", "research",
]


def title_relevance(rec: dict) -> float:
    """0..1 from current title (weighted most) and historical titles."""
    cur = rec.get("profile", {}).get("current_title", "").lower()
    hist = [j.get("title", "").lower() for j in rec.get("career_history", [])]

    def score_title(t: str) -> float:
        if jd.match_any(t, STRONG_TITLE):
            return 1.0
        if jd.match_any(t, ADJACENT_TITLE):
            return 0.55
        return 0.0

    cur_s = score_title(cur)
    hist_s = max([score_title(t) for t in hist], default=0.0)
    # Current title weighted 0.65; best historical 0.35.
    return min(1.0, 0.65 * cur_s + 0.35 * hist_s)


# --- career relevance: duration-weighted, judged on title+description ------

def career_relevance(blob: Blob) -> float:
    """Fraction of career *months* spent in genuinely relevant roles.

    Each role is judged by how many distinct role concepts its title+description
    demonstrate (one combined-regex pass), not by title string alone — so 'built a
    recommendation system' counts even from an unglamorous title (the JD's
    plain-language Tier-5 case).
    """
    if not blob.roles_l:
        return 0.0
    total = 0.0
    relevant = 0.0
    for months, text_l in blob.roles_l:
        if months == 0:
            continue
        hits = jd.match_count(text_l, jd.CORE_PLUS_EVAL_FLAT)
        # Credit grows with distinct concepts demonstrated, saturating at ~3 so a
        # single buzzword doesn't fully count and one rich role doesn't dominate.
        relevant += months * min(1.0, hits / 3.0)
        total += months
    if total == 0:
        return 0.0
    return min(1.0, relevant / total)


# --- seniority alignment ---------------------------------------------------

def seniority_alignment(rec: dict, job: jd.JobSpec) -> float:
    """Smooth band around the JD's ideal 6-8 yrs; soft outside, never a hard cut.

    The JD says the band is 'a range, not a requirement' — so we use a plateau in
    [ideal_lo, ideal_hi] decaying gently outside, with a floor so a 4- or 11-year
    candidate with strong other signals is not erased.
    """
    y = float(rec.get("profile", {}).get("years_of_experience", 0.0))
    lo, hi = job.ideal_years_lo, job.ideal_years_hi
    if lo <= y <= hi:
        return 1.0
    if y < lo:
        # decay below: 0 yrs -> ~0.3 floor
        return max(0.3, 1.0 - (lo - y) * 0.14)
    # above hi: very senior is fine but mild decay (role writes code)
    return max(0.45, 1.0 - (y - hi) * 0.08)


# --- behavioral envelope: a bounded multiplier, never the main driver ------

def _recency_factor(last_active: str, anchor: date) -> float:
    """1.0 if active near anchor; decays to ~0.6 by ~8 months stale."""
    try:
        la = date.fromisoformat(last_active)
    except Exception:
        return 0.85
    days = (anchor - la).days
    if days <= 30:
        return 1.0
    if days >= 240:
        return 0.6
    return 1.0 - (days - 30) * (0.4 / 210)


def behavioral_envelope(rec: dict, anchor: date) -> tuple[float, dict]:
    """Bounded multiplier in [0.5, 1.2] from availability, reliability, demand.

    Returns (multiplier, parts) where parts holds the human-readable pieces used
    in reasoning. Designed so it nudges, never dominates: a great-on-paper but
    unreachable candidate is down-weighted, not deleted.
    """
    s = rec.get("redrob_signals", {})

    # Availability
    resp = float(s.get("recruiter_response_rate", 0.0))           # 0..1
    notice = float(s.get("notice_period_days", 90))               # 0..180
    otw = bool(s.get("open_to_work_flag", False))
    recency = _recency_factor(s.get("last_active_date", ""), anchor)
    notice_factor = max(0.0, 1.0 - notice / 180.0)                # sub-30 best
    availability = 0.45 * resp + 0.25 * notice_factor + 0.15 * (1.0 if otw else 0.0) + 0.15 * recency

    # Reliability ( -1 == unknown -> neutral 0.6 )
    icr = s.get("interview_completion_rate", -1)
    oar = s.get("offer_acceptance_rate", -1)
    icr = 0.6 if icr is None or icr < 0 else float(icr)
    oar = 0.6 if oar is None or oar < 0 else float(oar)
    reliability = 0.6 * icr + 0.4 * oar

    # Demand (mild positive): saved/searched by recruiters
    saved = float(s.get("saved_by_recruiters_30d", 0))
    searched = float(s.get("search_appearance_30d", 0))
    demand = 0.5 * min(saved / 15.0, 1.0) + 0.5 * min(searched / 60.0, 1.0)

    # Compose around 1.0: availability is the biggest mover.
    raw = 0.78 + 0.30 * availability + 0.10 * (reliability - 0.6) + 0.06 * demand
    mult = max(0.5, min(1.2, raw))

    raw_icr = s.get("interview_completion_rate", -1)
    parts = {
        "response_rate": resp,
        "notice_days": int(notice),
        "open_to_work": otw,
        "recency_factor": round(recency, 2),
        "interview_completion": None if raw_icr in (None, -1) or raw_icr < 0 else float(raw_icr),
        "saved_by_recruiters_30d": int(saved),
    }
    return mult, parts


# --- location: bona-fide hybrid-role logistics, used as a tiny shaper ------

def location_factor(rec: dict) -> float:
    """Small positive for in-scope Indian metros / relocation willingness.

    Justified by the JD (Pune/Noida hybrid, lists in-scope cities, no visa
    sponsorship outside India). Kept tiny so it never overrides fit. NOT a
    nationality/ethnicity signal — see fairness.py.
    """
    p = rec.get("profile", {})
    loc = (p.get("location", "") + " " + p.get("country", "")).lower()
    country = p.get("country", "").lower()
    relocate = bool(rec.get("redrob_signals", {}).get("willing_to_relocate", False))
    if jd.any_hit(loc, jd.PREFERRED_LOCATIONS):
        return 1.0
    if "india" in country:
        return 0.9 if relocate else 0.8
    # Outside India: case-by-case, no visa sponsorship -> mild discount.
    return 0.7 if relocate else 0.55
