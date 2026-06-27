"""Job-description decomposition — the "deep job understanding" layer.

There is exactly ONE target JD in this challenge (Redrob's "Senior AI Engineer —
Founding Team"). Rather than run a generic keyword extractor over it and pretend
that is understanding, we read the JD as a human would and encode its *meaning* as
structured requirements: the must-haves, the explicit disqualifiers, the implied
seniority, and a single prose "ideal candidate" description used for embedding.

Every concept list below is traceable to a specific sentence in
data/job_description.txt. The deck can show the JD line next to the rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --- Concept vocabularies -------------------------------------------------
# These are matched (word-boundary, case-insensitive) against titles, career
# descriptions, summaries and skill names. They are the structured half of
# "seeing beyond keywords": a concept fires on any of several surface forms.

# The core of the role: ranking / retrieval / search / recommendation / IR.
CORE_ROLE_CONCEPTS = {
    "ranking": ["ranking", "learning to rank", "learning-to-rank", "ltr", "re-rank", "rerank", "relevance"],
    "retrieval": ["retrieval", "retriever", "information retrieval", "semantic search", "rag"],
    "search": ["search engine", "search system", "search relevance", "query understanding", "elasticsearch", "opensearch", "solr", "lucene"],
    "recommendation": ["recommendation", "recommender", "recsys", "personalization", "candidate generation"],
    "embeddings": ["embedding", "embeddings", "sentence-transformers", "sentence transformers", "bge", "e5", "word2vec", "vector representation"],
    "vector_db": ["vector database", "vector db", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "annoy", "hnsw", "ann index", "nearest neighbor"],
}

# Production applied-ML — the JD wants product-company ML, not pure research.
ML_PRODUCTION_CONCEPTS = {
    "ml_engineering": ["machine learning", "ml engineer", "applied ml", "applied scientist", "ml systems", "ml platform", "model serving", "model deployment", "production ml", "mlops", "feature pipeline", "feature store"],
    "nlp": ["nlp", "natural language", "language model", "text classification", "named entity", "question answering", "transformer", "bert", "llm"],
    "deep_learning": ["deep learning", "pytorch", "tensorflow", "neural network", "fine-tuning", "fine tuning", "lora", "qlora", "peft"],
}

# Evaluation maturity — explicitly a must-have ("designing evaluation frameworks").
EVAL_CONCEPTS = {
    "eval": ["ndcg", "mrr", "map@", "mean average precision", "precision@", "recall@", "a/b test", "ab test", "offline evaluation", "online evaluation", "ranking metric", "eval framework", "evaluation framework"],
}

# Nice-to-haves: won't reject for, but a genuine plus.
NICE_TO_HAVE_CONCEPTS = {
    "ltr_models": ["xgboost", "lightgbm", "gradient boosting", "learning to rank"],
    "hr_tech": ["hr-tech", "hr tech", "recruiting", "recruitment", "talent", "marketplace", "hiring"],
    "scale": ["distributed", "large-scale", "low latency", "high throughput", "inference optimization", "scale"],
    "open_source": ["open source", "open-source", "contributor", "maintainer", "published", "paper", "arxiv"],
}

# --- Negative / disqualifier signals (straight from the JD) ---------------

# "People who have only worked at consulting firms ... in their entire career."
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "l&t infotech",
    "lti", "mphasis", "ibm global", "deloitte", "pwc", "kpmg", "ernst & young",
    "persistent systems", "hexaware", "birlasoft", "nttdata", "ntt data", "dxc",
]

# "primary expertise is computer vision, speech, or robotics without NLP/IR".
CV_SPEECH_ROBOTICS = [
    "computer vision", "image classification", "object detection", "image segmentation",
    "opencv", "speech recognition", "text-to-speech", "tts", "asr", "voice",
    "robotics", "slam", "lidar", "autonomous", "pose estimation", "ocr", "face recognition",
]

# "AI experience consists primarily of recent ... LangChain to call OpenAI".
FRAMEWORK_ENTHUSIAST = [
    "langchain", "llama-index", "llamaindex", "autogen", "crewai", "prompt engineering",
]

# Aspirational language = wannabe / keyword-stuffer summary. Strong negative:
# the summary describes a *future* identity, not demonstrated production work.
ASPIRATIONAL_PHRASES = [
    "transitioning into", "transitioning toward", "transition into", "building competence",
    "building my competence", "learning", "self-directed", "side project", "side-project",
    "interested in", "interested in transitioning", "aspiring", "looking to move", "want to move",
    "hoping to", "passionate about getting into", "break into", "pivoting", "upskilling",
    "trying to get into", "exploring a move", "keen to", "eager to learn", "on the ml side",
]

# Pure research without production deployment — an explicit hard disqualifier.
PURE_RESEARCH = [
    "research scientist", "research-only", "postdoc", "post-doc", "phd candidate",
    "academic researcher", "research fellow", "research assistant", "publications",
]

# Title-chasing / "stopped writing code" titles. Mild negatives, used in audit.
MANAGER_TITLES = [
    "engineering manager", "director", "vice president", " vp ", "head of",
    "tech lead", "technical lead", "architect", "principal architect",
]

# Locations the JD calls out as in-scope (bona fide hybrid-role requirement, not
# a bias proxy — the role is Pune/Noida hybrid). Used as a *mild* positive only.
PREFERRED_LOCATIONS = [
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram", "hyderabad",
    "mumbai", "bangalore", "bengaluru",
]


# Flattened vocabularies (module constants so id()-keyed regex caches are stable).
# Used on the hot path to replace many per-group searches with one combined search.
MUSTHAVE_FLAT: list[str] = [t for g in CORE_ROLE_CONCEPTS.values() for t in g] + \
    [t for g in ML_PRODUCTION_CONCEPTS.values() for t in g]
EVAL_FLAT: list[str] = [t for g in EVAL_CONCEPTS.values() for t in g]
CORE_PLUS_EVAL_FLAT: list[str] = MUSTHAVE_FLAT + EVAL_FLAT
NLP_IR_FLAT: list[str] = (
    ML_PRODUCTION_CONCEPTS["nlp"]
    + CORE_ROLE_CONCEPTS["retrieval"]
    + CORE_ROLE_CONCEPTS["search"]
    + CORE_ROLE_CONCEPTS["ranking"]
)
DEEP_ML_FLAT: list[str] = ML_PRODUCTION_CONCEPTS["ml_engineering"] + ML_PRODUCTION_CONCEPTS["deep_learning"]


@dataclass
class JobSpec:
    """Decomposed, machine-usable view of the target JD."""

    title: str
    min_years: float
    max_years: float
    ideal_years_lo: float
    ideal_years_hi: float
    ideal_text: str
    must_have_concepts: dict = field(default_factory=dict)
    eval_concepts: dict = field(default_factory=dict)
    nice_to_have_concepts: dict = field(default_factory=dict)
    raw_text: str = ""


# The "ideal candidate" prose, written FROM the JD's own "How to read between the
# lines" section. This is what we embed and compare every candidate against. It is
# deliberately buzzword-light in places ("built a recommendation system at a
# product company") so the embedding rewards demonstrated work, not keyword bingo.
IDEAL_PROFILE_TEXT = (
    "Senior AI/ML engineer with roughly six to eight years of experience, four to "
    "five of them in applied machine learning at product companies rather than "
    "services or pure research. Has personally shipped at least one end-to-end "
    "ranking, search, retrieval, or recommendation system to real users at "
    "meaningful scale. Strong production experience with embeddings-based retrieval "
    "using sentence-transformers, BGE, E5 or similar models, and with vector search "
    "or hybrid search infrastructure such as FAISS, Pinecone, Weaviate, Qdrant, "
    "Milvus, Elasticsearch or OpenSearch — including operational concerns like "
    "embedding drift, index refresh and retrieval-quality regression. Strong Python "
    "engineer who cares about code quality and still writes production code. "
    "Designs rigorous evaluation frameworks for ranking systems using NDCG, MRR, "
    "MAP, A/B testing and offline-to-online correlation. Understood retrieval and "
    "ranking before large language models became fashionable, with pre-LLM machine "
    "learning production experience in NLP and information retrieval. Scrappy "
    "product-engineering attitude: ships a working ranker fast and improves it from "
    "real user feedback. Comfortable owning the intelligence layer of a product, "
    "mentoring engineers, and defending design choices about hybrid vs dense "
    "retrieval, when to fine-tune vs prompt, and how to evaluate a ranking system."
)


def build_job_spec(jd_text: str) -> JobSpec:
    """Decompose the JD text into a JobSpec.

    The numeric band and concept vocabularies are fixed for this single known JD;
    we still parse the experience band from the text so the value is traceable and
    a different JD would update it.
    """
    yrs = re.findall(r"(\d+)\s*[-–]\s*(\d+)\s*years", jd_text.lower())
    if yrs:
        min_years, max_years = float(yrs[0][0]), float(yrs[0][1])
    else:
        min_years, max_years = 5.0, 9.0

    must = {}
    must.update(CORE_ROLE_CONCEPTS)
    must.update(ML_PRODUCTION_CONCEPTS)

    return JobSpec(
        title="Senior AI Engineer — Founding Team",
        min_years=min_years,
        max_years=max_years,
        ideal_years_lo=6.0,
        ideal_years_hi=8.0,
        ideal_text=IDEAL_PROFILE_TEXT,
        must_have_concepts=must,
        eval_concepts=EVAL_CONCEPTS,
        nice_to_have_concepts=NICE_TO_HAVE_CONCEPTS,
        raw_text=jd_text,
    )


# --- Concept matching helpers ---------------------------------------------

def _term_regex(t: str) -> str:
    """Regex fragment for one (lowercased) term: word-boundaries for plain tokens,
    escaped substring for phrases / tokens containing punctuation."""
    if " " in t or any(c in t for c in "@-&./+"):
        return re.escape(t)
    return rf"\b{re.escape(t)}\b"


# Memoized caches keyed by term-list identity. The vocabularies are module-level
# constants, so each is compiled exactly once for the whole 100K pool. Patterns are
# built over LOWERCASED terms and matched against text the caller has already
# lowercased — dropping re.IGNORECASE roughly halves match cost at this scale.
_ANY_CACHE: dict[int, re.Pattern] = {}
_COUNT_CACHE: dict[int, list[re.Pattern]] = {}


def _combined(terms: list[str]) -> re.Pattern:
    key = id(terms)
    cached = _ANY_CACHE.get(key)
    if cached is None:
        cached = re.compile("|".join(_term_regex(t.lower()) for t in terms))
        _ANY_CACHE[key] = cached
    return cached


def _per_term(terms: list[str]) -> list[re.Pattern]:
    key = id(terms)
    cached = _COUNT_CACHE.get(key)
    if cached is None:
        cached = [re.compile(_term_regex(t.lower())) for t in terms]
        _COUNT_CACHE[key] = cached
    return cached


# --- low-level helpers: caller guarantees `low` is already lowercased ------

def match_any(low: str, terms: list[str]) -> bool:
    """True if any term appears in the already-lowercased `low` (hot path)."""
    if not low:
        return False
    return _combined(terms).search(low) is not None


def match_count(low: str, terms: list[str]) -> int:
    """Distinct terms present in already-lowercased `low`. Short-circuits to 0 (the
    common case over a noisy pool) on the combined regex before per-term checks."""
    if not low or not _combined(terms).search(low):
        return 0
    return sum(1 for pat in _per_term(terms) if pat.search(low))


# --- convenience wrappers for non-hot callers (tests, reasoning) -----------

def any_hit(text: str, terms: list[str]) -> bool:
    return match_any(text.lower() if text else "", terms)


def count_hits(text: str, terms: list[str]) -> int:
    return match_count(text.lower() if text else "", terms)
