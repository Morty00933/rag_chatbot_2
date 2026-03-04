from __future__ import annotations
import hashlib
import logging
from typing import Dict, List, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ...services.embeddings import get_embeddings
from ...services.vectorstore import get_vectorstore
from ...services.indexing import Indexer
from ...services.chunking import split_with_metadata
from ...db import get_docstore
from ...core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestResponse(BaseModel):
    ok: bool
    document_id: int
    document_hash: str
    filename: str
    chunks: int


def _stable_document_id(filename: str, document_hash: str) -> int:
    """Build a deterministic integer identifier for a document."""
    hasher = hashlib.blake2b(digest_size=16)  # 16 bytes for fewer collisions
    hasher.update(filename.encode("utf-8", errors="replace"))
    hasher.update(b"\0")
    hasher.update(document_hash.encode("ascii"))
    return int.from_bytes(hasher.digest(), "big", signed=False)


@router.post("", response_model=IngestResponse, tags=["ingest"])
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(400, "Empty file")

    # Validate file size
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content_bytes) > max_bytes:
        raise HTTPException(
            413,
            f"File too large: {len(content_bytes)} bytes "
            f"(max {settings.MAX_UPLOAD_MB} MB)",
        )

    try:
        text = content_bytes.decode("utf-8", errors="replace")
        if "\ufffd" in text:
            logger.warning(
                "File %s contains non-UTF-8 bytes (replaced with U+FFFD)",
                file.filename,
            )
    except Exception as e:
        raise HTTPException(400, f"Decode error: {e}")

    document_hash = hashlib.sha256(content_bytes).hexdigest()
    document_size = len(content_bytes)
    content_type = file.content_type or "application/octet-stream"

    filename = file.filename or ""

    document_id = _stable_document_id(filename, document_hash)

    rich_chunks = split_with_metadata(
        text=text,
        filename=file.filename,
        document_id=document_id,
        chunk_size=800,
        overlap=120,
        strip_html=True,
        markdown_aware=True,
    )
    if not rich_chunks:
        raise HTTPException(400, "No chunks produced")

    chunks: List[str] = []
    metas: List[Dict[str, object]] = []
    items_for_store: List[Tuple[str, Dict[str, object]]] = []

    chunk_total = len(rich_chunks)

    for idx, rc in enumerate(rich_chunks):
        cid = f"{document_id}:{idx}"
        meta = {
            "chunk_id": cid,
            "chunk_index": idx,
            "filename": filename,
            "document_id": document_id,
            "document_sha256": document_hash,
            "document_size": document_size,
            "chunk_total": chunk_total,
            "content_type": content_type,
            "heading": rc.get("heading", ""),
            "level": rc.get("level", "0"),
            "span": rc.get("span", [0, 0]),
        }
        metas.append(meta)
        chunks.append(rc["text"])
        items_for_store.append((cid, {"meta": meta, "text": rc["text"]}))

    # Persist raw texts
    get_docstore().bulk_put(items_for_store)

    # Upsert into vector store
    indexer = Indexer(get_embeddings(), get_vectorstore())
    n = indexer.upsert_chunks(chunks, metas)

    return IngestResponse(
        ok=True,
        document_id=document_id,
        document_hash=document_hash,
        filename=filename,
        chunks=n,
    )
