"""HTTP relay server for internet-based peer discovery.

A lightweight signaling server that peers can register with
to find each other across the internet (not just LAN).
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AirTrain Relay", version="0.1.0")

PEER_TTL = 30.0  # seconds before peer expires


class RelayPeer(BaseModel):
    peer_id: str
    hostname: str
    ip_address: str
    port: int
    chip: str = ""
    memory_gb: float = 0.0
    tflops: float = 0.0
    model_name: str = ""
    session_id: str = ""
    last_seen: float = 0.0


_peers: dict[str, RelayPeer] = {}


def _cleanup():
    """Remove expired peers."""
    now = time.time()
    expired = [pid for pid, p in _peers.items() if now - p.last_seen > PEER_TTL]
    for pid in expired:
        del _peers[pid]


@app.post("/peers")
def register_peer(peer: RelayPeer) -> dict:
    """Register or update a peer."""
    peer.last_seen = time.time()
    _peers[peer.peer_id] = peer
    _cleanup()
    return {"status": "ok", "peer_id": peer.peer_id}


@app.get("/peers")
def list_peers(session_id: Optional[str] = None) -> list[RelayPeer]:
    """List active peers, optionally filtered by session."""
    _cleanup()
    peers = list(_peers.values())
    if session_id:
        peers = [p for p in peers if p.session_id == session_id]
    return peers


@app.delete("/peers/{peer_id}")
def unregister_peer(peer_id: str) -> dict:
    """Unregister a peer."""
    if peer_id not in _peers:
        raise HTTPException(status_code=404, detail="Peer not found")
    del _peers[peer_id]
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    """Health check."""
    _cleanup()
    return {"status": "ok", "active_peers": len(_peers)}
