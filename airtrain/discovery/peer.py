"""Peer management for AirTrain."""

from __future__ import annotations

import logging
import platform
import time
from typing import Optional

from airtrain.config import PeerInfo, PeerRole, PeerStatus

logger = logging.getLogger(__name__)


class PeerManager:
    """Tracks connected peers in the training swarm."""

    def __init__(self):
        self.peers: dict[str, PeerInfo] = {}
        self._connect_times: dict[str, float] = {}

    def add_peer(self, peer: PeerInfo) -> None:
        self.peers[peer.peer_id] = peer
        self._connect_times[peer.peer_id] = time.time()
        logger.info("Peer added: %s (%s, %.1f GB)", peer.peer_id, peer.chip, peer.memory_gb)

    def remove_peer(self, peer_id: str) -> Optional[PeerInfo]:
        peer = self.peers.pop(peer_id, None)
        self._connect_times.pop(peer_id, None)
        if peer:
            logger.info("Peer removed: %s", peer_id)
        return peer

    def get_peer(self, peer_id: str) -> Optional[PeerInfo]:
        return self.peers.get(peer_id)

    def get_peers(self, status: PeerStatus | None = None) -> list[PeerInfo]:
        if status is None:
            return list(self.peers.values())
        return [p for p in self.peers.values() if p.status == status]

    def update_status(self, peer_id: str, status: PeerStatus) -> None:
        if peer_id in self.peers:
            self.peers[peer_id].status = status

    def update_step(self, peer_id: str, step: int) -> None:
        if peer_id in self.peers:
            self.peers[peer_id].current_step = step

    def add_compute_hours(self, peer_id: str, hours: float) -> None:
        if peer_id in self.peers:
            self.peers[peer_id].compute_hours += hours

    @property
    def active_count(self) -> int:
        return len([p for p in self.peers.values() if p.status != PeerStatus.DISCONNECTED])

    @property
    def total_tflops(self) -> float:
        return sum(p.tflops for p in self.peers.values() if p.status != PeerStatus.DISCONNECTED)


def get_local_peer_info(port: int = 7471, role: PeerRole = PeerRole.WORKER) -> PeerInfo:
    """Detect local machine info and create a PeerInfo."""
    import socket

    chip = _detect_chip()
    memory_gb = _detect_memory()
    tflops = _estimate_tflops(chip)

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except Exception:
        hostname = platform.node()
        ip = "127.0.0.1"

    return PeerInfo(
        hostname=hostname,
        ip_address=ip,
        port=port,
        chip=chip,
        memory_gb=memory_gb,
        tflops=tflops,
        role=role,
    )


def _detect_chip() -> str:
    machine = platform.machine()
    system = platform.system()

    if system == "Darwin" and machine == "arm64":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return "Apple Silicon"
    return f"{machine} ({platform.processor() or 'unknown'})"


def _detect_memory() -> float:
    try:
        import os

        if platform.system() == "Darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            return int(result.stdout.strip()) / (1024**3)
        else:
            # Windows/Linux fallback
            import shutil

            total, _, _ = shutil.disk_usage("/")
            # Use a rough heuristic — this isn't memory but better than 0
            return round(os.cpu_count() * 2, 1)  # rough estimate
    except Exception:
        return 0.0


# Rough TFLOPS estimates for Apple Silicon chips
_TFLOPS_MAP = {
    "M1": 1.36,
    "M1 Pro": 4.0,
    "M1 Max": 8.0,
    "M1 Ultra": 16.0,
    "M2": 2.24,
    "M2 Pro": 5.0,
    "M2 Max": 10.0,
    "M2 Ultra": 20.0,
    "M3": 2.47,
    "M3 Pro": 5.5,
    "M3 Max": 11.0,
    "M3 Ultra": 22.0,
    "M4": 2.90,
    "M4 Pro": 6.5,
    "M4 Max": 18.43,
    "M4 Ultra": 36.0,
    "M5": 4.15,
    "M5 Pro": 8.29,
}


def _estimate_tflops(chip: str) -> float:
    for name, tflops in sorted(_TFLOPS_MAP.items(), key=lambda x: -len(x[0])):
        if name in chip:
            return tflops
    return 1.0  # conservative default
