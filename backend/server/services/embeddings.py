from __future__ import annotations

from typing import List, Any
import hashlib
import math
import logging
import os
import threading

from .interfaces import Embeddings
from ..core.config import settings

try:  # pragma: no cover - module availability depends on environment
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # noqa: BLE001 - we intentionally swallow import issues here
    SentenceTransformer = Any  # только для аннотаций/IDE
    _HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

_sbert_cache: Any | None = None
_embeddings_singleton: Embeddings | None = None
_embeddings_lock = threading.Lock()


class HashEmbeddings(Embeddings):
    """Fallback embedding that deterministically hashes tokens into a dense vector."""

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dim

    def _vectorize(self, text: str) -> List[float]:
        buckets = [0.0] * self.dim
        for token in text.lower().split():
            idx = self._bucket(token)
            buckets[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
        return [v / norm for v in buckets]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(t or "") for t in texts]


class SbertEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        global _sbert_cache
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers is unavailable")
        if _sbert_cache is not None and getattr(_sbert_cache, "_model_card", None) == model_name:
            self.model = _sbert_cache
        else:
            model = SentenceTransformer(model_name)
            setattr(model, "_model_card", model_name)
            _sbert_cache = model
            self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        raw_vecs = self.model.encode(texts, normalize_embeddings=True)
        if hasattr(raw_vecs, "tolist"):
            return raw_vecs.tolist()
        return [list(vec) for vec in raw_vecs]


def _build_embeddings() -> Embeddings:
    provider = (settings.EMBED_PROVIDER or "sbert").lower()
    if provider == "hash":
        return HashEmbeddings(settings.EMBED_DIM)
    if provider == "sbert":
        if os.environ.get("PYTEST_CURRENT_TEST"):
            logger.info("Using HashEmbeddings because tests are running")
            return HashEmbeddings(settings.EMBED_DIM)
        try:
            return SbertEmbeddings(settings.EMBED_MODEL)
        except Exception as exc:  # noqa: BLE001 - we want a graceful fallback
            logger.warning("Falling back to HashEmbeddings due to error: %s", exc)
            return HashEmbeddings(settings.EMBED_DIM)
    raise NotImplementedError(f"Unsupported EMBED_PROVIDER={provider}")


def get_embeddings() -> Embeddings:
    global _embeddings_singleton
    if _embeddings_singleton is None:
        with _embeddings_lock:
            if _embeddings_singleton is None:
                _embeddings_singleton = _build_embeddings()
    return _embeddings_singleton
