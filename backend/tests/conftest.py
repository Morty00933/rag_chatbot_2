"""Shared pytest fixtures for the RAG chatbot backend tests."""
from __future__ import annotations

import os
import pytest

# Ensure test environment is set before any imports
os.environ.setdefault("ENV", "dev")
os.environ.setdefault("PYTEST_CURRENT_TEST", "1")
os.environ.setdefault("EMBED_PROVIDER", "hash")
os.environ.setdefault("VECTOR_BACKEND", "memory")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all singletons between tests to ensure isolation."""
    yield
    # Reset embeddings singleton
    from server.services import embeddings as emb_mod
    emb_mod._embeddings_singleton = None

    # Reset vectorstore singleton
    from server.services import vectorstore as vs_mod
    vs_mod._vectorstore_singleton = None

    # Reset docstore singleton
    from server.db import reset_docstore
    reset_docstore()


@pytest.fixture()
def test_client():
    """FastAPI TestClient with test settings."""
    from fastapi.testclient import TestClient
    from server.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def embeddings():
    """Return a HashEmbeddings instance for tests."""
    from server.services.embeddings import HashEmbeddings, get_embeddings
    emb = get_embeddings()
    assert isinstance(emb, HashEmbeddings), "Tests must use HashEmbeddings"
    return emb


@pytest.fixture()
def vectorstore():
    """Return an InMemoryVectorStore for tests."""
    from server.services.vectorstore import InMemoryVectorStore, get_vectorstore
    vs = get_vectorstore()
    assert isinstance(vs, InMemoryVectorStore), "Tests must use InMemoryVectorStore"
    return vs
