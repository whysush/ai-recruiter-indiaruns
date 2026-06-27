"""Local sentence-embedding wrapper with a precompute/cache split.

Design for the 5-minute, no-network ranking constraint:

  * Computing embeddings for 100K candidates is the ONLY slow step. We treat it as
    a declared ONE-TIME pre-computation (scripts/precompute_embeddings.py): it may
    use the network once to download the local model and may exceed 5 minutes.
  * The cached result is a float16 `.npy` matrix aligned to candidate order, plus
    a parallel id list. The timed ranking step loads these with NUMPY ONLY — no
    torch, no network — so reproduction inside the judges' sandbox is trivial.

If neither cache nor model is available, `semantic_fit` degrades gracefully to a
deterministic lexical proxy (see score.py), so the pipeline still runs offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384


def load_model(model_name: str = MODEL_NAME, max_seq_length: int = 128):
    """Load the local sentence-transformer. Raises if unavailable (caller falls back).

    max_seq_length is capped at 128 tokens: the discriminative signal (summary,
    headline, current/recent titles) sits at the start of each candidate's text, so
    128 tokens keep the semantic quality while roughly halving precompute time.
    """
    from sentence_transformers import SentenceTransformer  # lazy: not needed at rank time

    m = SentenceTransformer(model_name, device="cpu")
    if max_seq_length:
        m.max_seq_length = max_seq_length
    return m


def embed_texts(model, texts: list[str], batch_size: int = 128,
                log_every: int = 20000) -> np.ndarray:
    """Embed and L2-normalize a list of texts -> float32 [N, D].

    Manually chunked so we can print real progress (sentence-transformers' own bar
    is suppressed when stdout isn't a TTY)."""
    import time as _t

    n = len(texts)
    out = np.empty((n, EMB_DIM), dtype=np.float32)
    t0 = _t.time()
    done = 0
    next_log = log_every
    for start in range(0, n, batch_size):
        chunk = texts[start:start + batch_size]
        vecs = model.encode(
            chunk, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        out[start:start + len(chunk)] = vecs.astype(np.float32)
        done += len(chunk)
        if done >= next_log or done == n:
            rate = done / max(1e-9, (_t.time() - t0))
            eta = (n - done) / max(1e-9, rate)
            print(f"[precompute]   {done}/{n} embedded  ({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)
            next_log = done + log_every
    return out


def embed_query(model, text: str) -> np.ndarray:
    v = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return v[0].astype(np.float32)


# --- cache I/O ------------------------------------------------------------

def save_cache(cache_dir: str | Path, ids: list[str], matrix: np.ndarray, ideal_vec: np.ndarray) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "cand_embeddings.npy", matrix.astype(np.float16))
    np.save(cache_dir / "ideal_vec.npy", ideal_vec.astype(np.float16))
    with open(cache_dir / "cand_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ids))


def load_cache(cache_dir: str | Path):
    """Return (ids, matrix_float32, ideal_vec_float32) or None if missing."""
    cache_dir = Path(cache_dir)
    mpath = cache_dir / "cand_embeddings.npy"
    ipath = cache_dir / "ideal_vec.npy"
    idpath = cache_dir / "cand_ids.txt"
    if not (mpath.exists() and ipath.exists() and idpath.exists()):
        return None
    matrix = np.load(mpath).astype(np.float32)
    ideal = np.load(ipath).astype(np.float32)
    ids = idpath.read_text(encoding="utf-8").splitlines()
    return ids, matrix, ideal


def cosine_to_ideal(matrix: np.ndarray, ideal_vec: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row to the ideal vector.

    Rows and ideal are already L2-normalized, so this is a single dot product.
    Returns a [N] array in roughly [-1, 1].
    """
    return matrix @ ideal_vec
