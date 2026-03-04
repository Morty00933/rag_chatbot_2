from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException

from ...services.embeddings import get_embeddings
from ...services.llm import get_llm
from ...services.retriever import HybridRetriever
from ...services.vectorstore import get_vectorstore
from ...schemas.chat import ChatRequest, ChatResponse, Reference
from ...services.prompting import get_system_instruction, build_user_prompt
from ...db import get_docstore
from ...core.config import settings

if TYPE_CHECKING:
    from ...services.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

_reranker: "CrossEncoderReranker | None" = None

router = APIRouter()


def _get_reranker() -> "CrossEncoderReranker | None":
    """Lazy init for CrossEncoderReranker — graceful degradation if unavailable."""
    global _reranker
    if _reranker is None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return None
        try:
            from ...services.reranker import CrossEncoderReranker

            _reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as exc:
            logger.warning("Reranker unavailable, using base ordering: %s", exc)
            _reranker = None
    return _reranker


def _normalize_candidate(c: Any) -> Tuple[str, Dict[str, Any], float]:
    """
    Normalize a candidate to (chunk_id, payload, score).
    Supports tuple/list and dict formats.
    """
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        chunk_id, payload, score = c[0], c[1], c[2]
        return str(chunk_id), (payload or {}), float(score or 0.0)

    if isinstance(c, dict):
        cid = c.get("id") or c.get("chunk_id") or c.get("point_id") or c.get("uuid") or "unknown"
        payload = c.get("payload") or {}
        score = c.get("score") or 0.0
        return str(cid), payload, float(score)

    return "unknown", {}, 0.0


def _collect_contexts_and_refs(
    candidates: List[Any], max_ctx: int = 6
) -> Tuple[List[str], List[Reference]]:
    """
    Retrieve chunk texts from docstore and build Reference objects.
    """
    docstore = get_docstore()
    contexts: List[str] = []
    refs: List[Reference] = []

    for c in candidates[:max_ctx]:
        chunk_id, payload, score = _normalize_candidate(c)

        text = (payload or {}).get("text")
        meta: Dict[str, Any] = (payload or {}).get("meta") or {}

        if not text:
            rec = docstore.get(chunk_id) if chunk_id and chunk_id != "unknown" else None
            if rec and isinstance(rec, dict):
                text = rec.get("text")
                meta = rec.get("meta") or meta

        if not text:
            continue

        filename = str((meta or {}).get("filename", "unknown"))
        doc_id_raw = (meta or {}).get("document_id", 0)
        try:
            doc_id = int(doc_id_raw)
        except (TypeError, ValueError):
            doc_id = 0
        # Try chunk_index first (from ingest metadata), fall back to chunk_ord
        chunk_ord = int((meta or {}).get("chunk_index", (meta or {}).get("chunk_ord", 0)))

        safe_text = text[:settings.MAX_CTX_LEN]
        contexts.append(safe_text)
        refs.append(
            Reference(
                document_id=doc_id,
                filename=str(filename),
                score=float(score),
                chunk_ord=chunk_ord,
                preview=safe_text[:200],
            )
        )

    return contexts, refs


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest) -> ChatResponse:
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")

    # Use request top_k with fallback to config
    final_k = req.top_k or settings.FINAL_K

    # 1) Hybrid retrieval
    retriever = HybridRetriever(get_embeddings(), get_vectorstore(), top_pool=settings.TOP_POOL)
    try:
        first_hits = retriever.search(q, top_k=settings.FIRST_K)
    except Exception as exc:
        logger.error("Retriever search failed: %s", exc, exc_info=True)
        first_hits = []

    # 2) Collect contexts and references
    contexts_raw, refs_raw = _collect_contexts_and_refs(
        first_hits, max_ctx=len(first_hits) or settings.FIRST_K
    )

    # No contexts — honest fallback
    if not contexts_raw:
        sys_instr = get_system_instruction()
        prompt = build_user_prompt(q, [], sys_instr)
        try:
            answer = await get_llm().generate(prompt)
        except Exception as exc:
            logger.error("LLM generation failed (no-context path): %s", exc, exc_info=True)
            answer = ""
        answer = (answer or "").strip() or "I don't know"
        return ChatResponse(answer=answer, references=[])

    # 3) Reranking attempt
    order = list(range(len(contexts_raw)))
    rr = _get_reranker()
    if rr is not None:
        try:
            scores = rr.score(q, contexts_raw)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        except Exception as exc:
            logger.warning("Reranking failed, using base ordering: %s", exc)

    k = min(final_k, len(order))
    contexts = [contexts_raw[i] for i in order[:k]]
    refs = [refs_raw[i] for i in order[:k]]

    # 4) System instruction + generate
    sys_instr = get_system_instruction()
    prompt = build_user_prompt(q, contexts, sys_instr)
    try:
        answer = await get_llm().generate(prompt)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc, exc_info=True)
        answer = ""
    answer = (answer or "").strip() or "I don't know"

    return ChatResponse(answer=answer, references=refs)
