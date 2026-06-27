"""A synthetic keyword-stuffer must rank below a genuine match, and the audit
must flag impossible profiles."""

from datetime import date

from airecruiter import score
from airecruiter.honeypot import audit
from airecruiter.jd import build_job_spec

JD_TEXT = (
    "Senior AI Engineer, 5-9 years. Embeddings retrieval, vector search, ranking, "
    "evaluation NDCG MRR. Product companies, NLP, information retrieval."
)
JOB = build_job_spec(JD_TEXT)
ANCHOR = date(2026, 5, 27)


def _sig(**kw):
    base = dict(
        profile_completeness_score=80,
        signup_date="2022-01-01",
        last_active_date="2026-05-20",
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=3,
        recruiter_response_rate=0.8,
        avg_response_time_hours=5,
        skill_assessment_scores={},
        connection_count=300,
        endorsements_received=100,
        notice_period_days=30,
        expected_salary_range_inr_lpa={"min": 30, "max": 50},
        preferred_work_mode="hybrid",
        willing_to_relocate=True,
        github_activity_score=60,
        search_appearance_30d=20,
        saved_by_recruiters_30d=5,
        interview_completion_rate=0.9,
        offer_acceptance_rate=0.7,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    base.update(kw)
    return base


GENUINE = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "A",
        "headline": "ML Engineer | Search & Ranking",
        "summary": "Applied ML engineer who built and shipped a production recommendation and "
        "search ranking system using embeddings and FAISS vector search at a product company. "
        "Designed NDCG/MRR evaluation pipelines.",
        "location": "Pune", "country": "India", "years_of_experience": 7.0,
        "current_title": "ML Engineer", "current_company": "ProductCo",
        "current_company_size": "201-500", "current_industry": "Software",
    },
    "career_history": [
        {"company": "ProductCo", "title": "ML Engineer", "start_date": "2021-01-01",
         "end_date": None, "duration_months": 60, "is_current": True, "industry": "Software",
         "company_size": "201-500",
         "description": "Built a semantic search and recommendation ranking system with "
         "sentence-transformers embeddings and FAISS; ran A/B tests and tracked NDCG."},
    ],
    "education": [{"institution": "X", "degree": "B.Tech", "field_of_study": "CS",
                   "start_year": 2012, "end_year": 2016, "grade": None, "tier": "tier_2"}],
    "skills": [
        {"name": "NLP", "proficiency": "advanced", "endorsements": 40, "duration_months": 60},
        {"name": "Information Retrieval", "proficiency": "advanced", "endorsements": 20, "duration_months": 48},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
    ],
    "redrob_signals": _sig(skill_assessment_scores={"NLP": 85, "Information Retrieval": 80, "FAISS": 78}),
}

# Keyword stuffer: marketing manager, perfect AI skill list, no supporting career,
# aspirational summary, no assessments.
STUFFER = {
    "candidate_id": "CAND_0000002",
    "profile": {
        "anonymized_name": "B",
        "headline": "Marketing Manager | NLP, LLM, RAG, Vector Search, Embeddings",
        "summary": "Marketing manager interested in transitioning into AI/ML; building "
        "competence in NLP and learning embeddings through self-directed side projects.",
        "location": "Pune", "country": "India", "years_of_experience": 7.0,
        "current_title": "Marketing Manager", "current_company": "AdCo",
        "current_company_size": "201-500", "current_industry": "Marketing",
    },
    "career_history": [
        {"company": "AdCo", "title": "Marketing Manager", "start_date": "2019-01-01",
         "end_date": None, "duration_months": 84, "is_current": True, "industry": "Marketing",
         "company_size": "201-500",
         "description": "Led marketing campaigns, brand strategy, SEO and content calendars."},
    ],
    "education": [{"institution": "Y", "degree": "MBA", "field_of_study": "Marketing",
                   "start_year": 2012, "end_year": 2014, "grade": None, "tier": "tier_2"}],
    "skills": [
        {"name": "NLP", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "RAG", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Vector Search", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
        {"name": "Embeddings", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
    ],
    "redrob_signals": _sig(skill_assessment_scores={}),
}

# Honeypot: impossible skill tenure (skill used 200 months on a 4-year career).
HONEYPOT = {
    "candidate_id": "CAND_0000003",
    "profile": {
        "anonymized_name": "C", "headline": "ML Engineer",
        "summary": "ML engineer.", "location": "Pune", "country": "India",
        "years_of_experience": 4.0, "current_title": "ML Engineer",
        "current_company": "Z", "current_company_size": "51-200", "current_industry": "Software",
    },
    "career_history": [
        {"company": "Z", "title": "ML Engineer", "start_date": "2022-01-01", "end_date": None,
         "duration_months": 48, "is_current": True, "industry": "Software", "company_size": "51-200",
         "description": "Built ML models."},
    ],
    "education": [],
    "skills": [
        {"name": "NLP", "proficiency": "expert", "endorsements": 5, "duration_months": 200},
        {"name": "Retrieval", "proficiency": "expert", "endorsements": 5, "duration_months": 180},
    ],
    "redrob_signals": _sig(),
}


def test_stuffer_ranks_below_genuine():
    rows = score.score_pool([GENUINE, STUFFER], JOB, None, ANCHOR)
    by_id = {r["candidate_id"]: r for r in rows}
    assert by_id["CAND_0000001"]["final_score"] > by_id["CAND_0000002"]["final_score"]


def test_honeypot_is_flagged_and_penalized():
    pen, flags = audit(HONEYPOT)
    assert pen > 0.2
    assert any("impossible skill tenure" in f for f in flags)


def test_stuffer_audit_flags_aspirational_and_unsupported():
    pen, flags = audit(STUFFER)
    assert pen > 0.2
    joined = " ".join(flags)
    assert "aspirational" in joined or "no career role" in joined


def test_parallel_scoring_matches_single_process():
    import numpy as np
    recs = [GENUINE, STUFFER, HONEYPOT]
    sem = np.array([0.5, 0.5, 0.5], dtype="float32")
    a = score.score_pool(recs, JOB, sem, ANCHOR)
    b = score.score_pool_parallel(recs, JOB, sem, ANCHOR, n_workers=2)
    am = {r["candidate_id"]: r["final_score"] for r in a}
    bm = {r["candidate_id"]: r["final_score"] for r in b}
    assert am.keys() == bm.keys()
    assert all(abs(am[k] - bm[k]) < 1e-12 for k in am)
