from __future__ import annotations
import logging
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)
import time

from .core.config import settings
from .api.routers import health, ingest, chat, admin

logger = logging.getLogger(__name__)

# Regex to normalize dynamic path segments for metrics labels
_DYNAMIC_PATH_RE = re.compile(r"/\d+")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    from .db.session import init_db

    await init_db()
    logger.info("Application started (ENV=%s)", settings.ENV)
    yield
    logger.info("Application shutting down")


app = FastAPI(title="RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, list[str] | str]:
    return {"status": "ok", "see": ["/docs", "/healthz", "/metrics"]}


# CORS — never fall back to wildcard
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if not origins:
    logger.warning("CORS_ORIGINS is empty; no origins will be allowed")
    origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
api_requests_total = Counter(
    "api_requests_total", "Total API requests", ["method", "route", "status"]
)
api_latency_hist = Histogram(
    "api_request_latency_seconds", "API request latency in seconds", ["route"]
)


def _normalize_route(path: str) -> str:
    """Replace dynamic path segments with placeholder and strip query params."""
    path = path.split("?")[0]
    path = _DYNAMIC_PATH_RE.sub("/:id", path)
    return path


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response: Response
    try:
        response = await call_next(request)
        return response
    finally:
        route = _normalize_route(request.url.path)
        status = str(response.status_code) if response else "500"
        api_requests_total.labels(request.method, route, status).inc()
        api_latency_hist.labels(route).observe(time.perf_counter() - start)


@app.get("/healthz", tags=["health"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get(settings.API_METRICS_PATH, tags=["metrics"], response_model=None)
async def metrics() -> PlainTextResponse | JSONResponse:
    if not settings.PROMETHEUS_ENABLED:
        return JSONResponse({"detail": "metrics disabled"}, status_code=404)
    data = generate_latest(REGISTRY)
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


# Routers
app.include_router(health.router, prefix="")
app.include_router(ingest.router, prefix="/ingest")
app.include_router(chat.router, prefix="")
app.include_router(admin.router, prefix="/admin")
