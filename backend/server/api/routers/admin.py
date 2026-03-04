from __future__ import annotations
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ...db import get_docstore
from ...services.vectorstore import get_vectorstore
from ...services.embeddings import get_embeddings
from ...services.indexing import Indexer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/documents", tags=["admin"])
async def list_documents() -> dict[str, Any]:
    """List all known documents from the docstore."""
    docstore = get_docstore()
    try:
        keys = docstore.keys() if hasattr(docstore, "keys") else []
        # Group chunks by document_id
        documents: dict[int, dict[str, Any]] = {}
        for key in keys:
            rec = docstore.get(key)
            if not rec or not isinstance(rec, dict):
                continue
            meta = rec.get("meta") or {}
            doc_id = meta.get("document_id", 0)
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "chunks": 0,
                }
            documents[doc_id]["chunks"] += 1
        return {"ok": True, "documents": list(documents.values())}
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc, exc_info=True)
        raise HTTPException(500, "Failed to list documents")


@router.delete("/documents/{document_id}", tags=["admin"])
async def delete_document(document_id: int) -> dict[str, Any]:
    """Delete a document and its chunks from docstore and vector store."""
    docstore = get_docstore()
    vs = get_vectorstore()
    deleted_chunks = 0

    try:
        # Find all chunk keys for this document
        keys = docstore.keys() if hasattr(docstore, "keys") else []
        chunk_ids_to_delete = []
        for key in keys:
            rec = docstore.get(key)
            if not rec or not isinstance(rec, dict):
                continue
            meta = rec.get("meta") or {}
            if meta.get("document_id") == document_id:
                chunk_ids_to_delete.append(key)

        if not chunk_ids_to_delete:
            raise HTTPException(404, f"Document {document_id} not found")

        # Delete from docstore
        for cid in chunk_ids_to_delete:
            if hasattr(docstore, "delete"):
                docstore.delete(cid)
            deleted_chunks += 1

        # Delete from vector store if supported
        if hasattr(vs, "delete"):
            try:
                vs.delete(chunk_ids_to_delete)
            except Exception as exc:
                logger.warning("Could not delete from vector store: %s", exc)

        logger.info("Deleted document %d (%d chunks)", document_id, deleted_chunks)
        return {"ok": True, "document_id": document_id, "deleted_chunks": deleted_chunks}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete document %d: %s", document_id, exc, exc_info=True)
        raise HTTPException(500, f"Failed to delete document: {exc}")


@router.post("/reindex", tags=["admin"])
async def reindex() -> dict[str, bool | str | int]:
    """Re-index all documents from docstore into vector store."""
    docstore = get_docstore()
    emb = get_embeddings()
    vs = get_vectorstore()
    indexer = Indexer(emb, vs)

    try:
        keys = docstore.keys() if hasattr(docstore, "keys") else []
        chunks = []
        metas = []
        for key in keys:
            rec = docstore.get(key)
            if not rec or not isinstance(rec, dict):
                continue
            text = rec.get("text")
            meta = rec.get("meta") or {}
            if text:
                chunks.append(text)
                metas.append(meta)

        if not chunks:
            return {"ok": True, "message": "no documents to reindex", "chunks": 0}

        n = indexer.upsert_chunks(chunks, metas)
        logger.info("Reindexed %d chunks", n)
        return {"ok": True, "message": "reindex completed", "chunks": n}
    except Exception as exc:
        logger.error("Reindex failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Reindex failed: {exc}")
