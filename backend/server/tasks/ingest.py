from __future__ import annotations
import logging
from typing import Dict, List

from .celery_app import celery_app
from ..services.embeddings import get_embeddings
from ..services.vectorstore import get_vectorstore
from ..services.indexing import Indexer
from ..services.chunking import split_with_metadata

logger = logging.getLogger(__name__)


@celery_app.task(name="ingest.text")
def ingest_text(doc_id: int, filename: str, content: str) -> dict[str, int | bool]:
    rich_chunks = split_with_metadata(
        text=content,
        filename=filename,
        document_id=doc_id,
        chunk_size=800,
        overlap=120,
        strip_html=True,
        markdown_aware=True,
    )

    chunks: List[str] = []
    metas: List[Dict[str, object]] = []

    for idx, rc in enumerate(rich_chunks):
        cid = f"{doc_id}:{idx}"
        meta = {
            "chunk_id": cid,
            "chunk_index": idx,
            "filename": filename,
            "document_id": doc_id,
            "heading": rc.get("heading", ""),
            "level": rc.get("level", "0"),
            "span": rc.get("span", [0, 0]),
        }
        chunks.append(rc["text"])
        metas.append(meta)

    indexer = Indexer(get_embeddings(), get_vectorstore())
    indexer.upsert_chunks(chunks, metas)
    logger.info("Ingested doc_id=%d filename=%s chunks=%d", doc_id, filename, len(chunks))
    return {"ok": True, "doc_id": doc_id, "chunks": len(chunks)}
