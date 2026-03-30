"""mDNS/Zeroconf peer discovery for AirTrain.

Uses multicast DNS to discover peers on the local network
with zero configuration — the same technology Apple uses for AirDrop discovery.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Callable, Optional

from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

from airtrain.config import PeerInfo, PeerStatus

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_airtrain._tcp.local."


class AirTrainServiceBrowser:
    """Browse for AirTrain peers on the local network."""

    def __init__(
        self,
        on_found: Callable[[PeerInfo], None] | None = None,
        on_removed: Callable[[str], None] | None = None,
    ):
        self.on_found = on_found
        self.on_removed = on_removed
        self.discovered: dict[str, PeerInfo] = {}
        self._zc: Optional[Zeroconf] = None
        self._browser: Optional[ServiceBrowser] = None

    def start(self):
        """Start browsing for peers."""
        self._zc = Zeroconf()
        self._browser = ServiceBrowser(
            self._zc, SERVICE_TYPE, handlers=[self._on_state_change]
        )
        logger.info("Started mDNS browser for %s", SERVICE_TYPE)

    def stop(self):
        """Stop browsing."""
        if self._browser:
            self._browser.cancel()
        if self._zc:
            self._zc.close()
        logger.info("Stopped mDNS browser")

    def _on_state_change(
        self, zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ):
        if state_change == ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                peer = self._service_to_peer(info)
                if peer:
                    self.discovered[peer.peer_id] = peer
                    logger.info("Discovered peer: %s at %s:%d", peer.hostname, peer.ip_address, peer.port)
                    if self.on_found:
                        self.on_found(peer)

        elif state_change == ServiceStateChange.Removed:
            peer_id = name.split(".")[0]
            removed = self.discovered.pop(peer_id, None)
            if removed:
                logger.info("Peer removed: %s", peer_id)
                if self.on_removed:
                    self.on_removed(peer_id)

    @staticmethod
    def _service_to_peer(info: ServiceInfo) -> Optional[PeerInfo]:
        try:
            addresses = info.parsed_addresses()
            if not addresses:
                return None

            props = {}
            if info.properties:
                props = {
                    k.decode(): v.decode() if isinstance(v, bytes) else v
                    for k, v in info.properties.items()
                }

            return PeerInfo(
                peer_id=props.get("peer_id", info.name.split(".")[0]),
                hostname=props.get("hostname", info.server or ""),
                ip_address=addresses[0],
                port=info.port or 7471,
                chip=props.get("chip", ""),
                memory_gb=float(props.get("memory_gb", 0)),
                tflops=float(props.get("tflops", 0)),
            )
        except Exception as e:
            logger.warning("Failed to parse service info: %s", e)
            return None


def register_service(peer: PeerInfo) -> tuple[Zeroconf, ServiceInfo]:
    """Register this peer as an AirTrain service on the local network."""
    properties = {
        "peer_id": peer.peer_id,
        "hostname": peer.hostname,
        "chip": peer.chip,
        "memory_gb": str(peer.memory_gb),
        "tflops": str(peer.tflops),
        "role": peer.role.value,
        "version": "0.1.0",
    }

    info = ServiceInfo(
        SERVICE_TYPE,
        f"{peer.peer_id}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(peer.ip_address)] if peer.ip_address and peer.ip_address != "127.0.0.1" else [socket.inet_aton(socket.gethostbyname(socket.gethostname()))],
        port=peer.port,
        properties=properties,
        server=f"{peer.hostname}.local.",
    )

    zc = Zeroconf()
    zc.register_service(info)
    logger.info("Registered mDNS service: %s on port %d", peer.peer_id, peer.port)

    return zc, info


def discover_peers(timeout: float = 5.0) -> list[PeerInfo]:
    """Discover AirTrain peers on the local network.

    Blocks for `timeout` seconds, then returns all discovered peers.
    """
    peers: list[PeerInfo] = []

    def on_found(peer: PeerInfo):
        peers.append(peer)

    browser = AirTrainServiceBrowser(on_found=on_found)
    browser.start()

    time.sleep(timeout)

    browser.stop()
    return peers
