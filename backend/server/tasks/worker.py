from __future__ import annotations
import socket
import threading
import time
import uvicorn
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import PlainTextResponse
from ..core.config import settings
from .celery_app import celery_app

# Worker metrics
worker_ingest_total = Counter("worker_ingest_total", "Total ingested documents")
worker_heartbeat = Gauge("worker_heartbeat", "Worker heartbeat", ["name"])

app = FastAPI(title="Worker Metrics", version="0.1.0")


@app.get("/")
async def metrics_root() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


def heartbeat_loop() -> None:
    name = socket.gethostname()
    while True:
        worker_heartbeat.labels(name).set(time.time())
        time.sleep(5)


def start_metrics_server() -> None:
    uvicorn.run(app, host="0.0.0.0", port=settings.WORKER_METRICS_PORT, log_level="warning")


def start_celery() -> None:
    celery_app.worker_main(["worker", "-l", "INFO", "-Q", "celery"])


if __name__ == "__main__":
    t1 = threading.Thread(target=start_metrics_server, daemon=True)
    t1.start()
    t2 = threading.Thread(target=heartbeat_loop, daemon=True)
    t2.start()
    start_celery()
