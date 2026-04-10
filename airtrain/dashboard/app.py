"""Local web dashboard for monitoring AirTrain training."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AirTrain Dashboard", version="0.1.0")


class MetricsCollector:
    """Collects and stores training metrics for the dashboard."""

    def __init__(self, max_points: int = 1000):
        self.metrics: deque[dict[str, Any]] = deque(maxlen=max_points)
        self.peers: dict[str, dict] = {}
        self.checkpoints: list[dict] = []
        self.status: dict[str, Any] = {
            "model_name": "",
            "global_step": 0,
            "loss": 0.0,
            "throughput": 0.0,
            "peer_count": 0,
            "status": "idle",
            "start_time": 0,
        }

    def update(self, step: int, loss: float, throughput: float = 0, peer_count: int = 0):
        self.metrics.append({
            "step": step,
            "loss": loss,
            "throughput": throughput,
            "peer_count": peer_count,
            "timestamp": time.time(),
        })
        self.status.update({
            "global_step": step,
            "loss": loss,
            "throughput": throughput,
            "peer_count": peer_count,
            "status": "training",
        })

    def set_peer(self, peer_id: str, info: dict):
        self.peers[peer_id] = info

    def remove_peer(self, peer_id: str):
        self.peers.pop(peer_id, None)

    def add_checkpoint(self, step: int, loss: float, path: str):
        self.checkpoints.append({
            "step": step,
            "loss": loss,
            "path": path,
            "timestamp": time.time(),
        })


# Global collector instance — set by coordinator
collector = MetricsCollector()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text())


@app.get("/api/status")
async def api_status():
    return collector.status


@app.get("/api/peers")
async def api_peers():
    return list(collector.peers.values())


@app.get("/api/checkpoints")
async def api_checkpoints():
    return collector.checkpoints


@app.get("/api/metrics")
async def api_metrics_sse():
    """Server-Sent Events endpoint for real-time metrics."""

    async def event_generator():
        last_idx = 0
        while True:
            current_len = len(collector.metrics)
            if current_len > last_idx:
                for i in range(last_idx, current_len):
                    data = json.dumps(collector.metrics[i])
                    yield f"data: {data}\n\n"
                last_idx = current_len
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def run_dashboard(port: int = 8471):
    """Run the dashboard server (called from coordinator)."""
    import uvicorn

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
