"""Async TCP transport for AirTrain peer communication."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from airtrain.config import NetworkConfig
from airtrain.network.protocol import Message, MessageType, read_message, write_message

logger = logging.getLogger(__name__)


@dataclass
class PeerConnection:
    """A connected peer with its reader/writer streams."""

    peer_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    address: str = ""
    last_heartbeat: float = 0.0


class TransportServer:
    """Async TCP server for the coordinator."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.connections: dict[str, PeerConnection] = {}
        self.on_message: Callable[[str, Message], Any] | None = None
        self.on_connect: Callable[[str], Any] | None = None
        self.on_disconnect: Callable[[str], Any] | None = None
        self._server: asyncio.Server | None = None
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self):
        """Start the TCP server."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.config.listen_host,
            self.config.listen_port,
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        addr = self._server.sockets[0].getsockname()
        logger.info("Transport server listening on %s:%d", addr[0], addr[1])

    async def stop(self):
        """Stop the server and disconnect all peers."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        for conn in list(self.connections.values()):
            conn.writer.close()
        self.connections.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def broadcast(self, msg: Message, exclude: str | None = None):
        """Send a message to all connected peers."""
        disconnected = []
        for peer_id, conn in self.connections.items():
            if peer_id == exclude:
                continue
            try:
                await write_message(conn.writer, msg)
            except (ConnectionError, OSError):
                disconnected.append(peer_id)

        for peer_id in disconnected:
            await self._disconnect_peer(peer_id)

    async def send_to(self, peer_id: str, msg: Message):
        """Send a message to a specific peer."""
        conn = self.connections.get(peer_id)
        if not conn:
            logger.warning("Peer %s not found", peer_id)
            return
        try:
            await write_message(conn.writer, msg)
        except (ConnectionError, OSError):
            await self._disconnect_peer(peer_id)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new client connection."""
        addr = writer.get_extra_info("peername")
        peer_id = ""

        try:
            # First message must be HANDSHAKE
            msg = await read_message(reader)
            if not msg or msg.msg_type != MessageType.HANDSHAKE:
                writer.close()
                return

            peer_id = msg.sender_id
            import time

            conn = PeerConnection(
                peer_id=peer_id,
                reader=reader,
                writer=writer,
                address=f"{addr[0]}:{addr[1]}",
                last_heartbeat=time.time(),
            )
            self.connections[peer_id] = conn
            logger.info("Peer connected: %s from %s", peer_id, addr)

            if self.on_connect:
                await self._maybe_await(self.on_connect, peer_id)

            # Message loop
            while True:
                msg = await read_message(reader)
                if not msg:
                    break
                if msg.msg_type == MessageType.HEARTBEAT:
                    conn.last_heartbeat = time.time()
                elif self.on_message:
                    await self._maybe_await(self.on_message, peer_id, msg)

        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            if peer_id:
                await self._disconnect_peer(peer_id)

    async def _disconnect_peer(self, peer_id: str):
        conn = self.connections.pop(peer_id, None)
        if conn:
            try:
                conn.writer.close()
            except Exception:
                pass
            logger.info("Peer disconnected: %s", peer_id)
            if self.on_disconnect:
                await self._maybe_await(self.on_disconnect, peer_id)

    async def _heartbeat_loop(self):
        """Check for stale peers periodically."""
        import time

        while True:
            await asyncio.sleep(self.config.heartbeat_interval)
            now = time.time()
            stale = [
                pid
                for pid, conn in self.connections.items()
                if now - conn.last_heartbeat > self.config.heartbeat_timeout
            ]
            for pid in stale:
                logger.warning("Peer %s heartbeat timeout", pid)
                await self._disconnect_peer(pid)

    @staticmethod
    async def _maybe_await(fn, *args):
        result = fn(*args)
        if asyncio.iscoroutine(result):
            await result


class TransportClient:
    """Async TCP client for workers."""

    def __init__(self, peer_id: str):
        self.peer_id = peer_id
        self.on_message: Callable[[Message], Any] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._listen_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self.connected = False

    async def connect(self, host: str, port: int):
        """Connect to the coordinator."""
        self._reader, self._writer = await asyncio.open_connection(host, port)
        self.connected = True

        # Send handshake
        handshake = Message(
            msg_type=MessageType.HANDSHAKE,
            sender_id=self.peer_id,
            metadata={"version": "0.1.0"},
        )
        await write_message(self._writer, handshake)

        self._listen_task = asyncio.create_task(self._listen_loop())
        self._heartbeat_task = asyncio.create_task(self._send_heartbeats())

        logger.info("Connected to coordinator at %s:%d", host, port)

    async def disconnect(self):
        """Disconnect from the coordinator."""
        self.connected = False
        if self._listen_task:
            self._listen_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._writer:
            self._writer.close()

    async def send(self, msg: Message):
        """Send a message to the coordinator."""
        if not self._writer:
            raise ConnectionError("Not connected")
        await write_message(self._writer, msg)

    async def _listen_loop(self):
        """Listen for messages from the coordinator."""
        try:
            while self.connected:
                msg = await read_message(self._reader)
                if not msg:
                    break
                if self.on_message:
                    result = self.on_message(msg)
                    if asyncio.iscoroutine(result):
                        await result
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.connected = False
            logger.info("Disconnected from coordinator")

    async def _send_heartbeats(self):
        """Send periodic heartbeats to the coordinator."""
        try:
            while self.connected:
                await asyncio.sleep(5.0)
                heartbeat = Message(
                    msg_type=MessageType.HEARTBEAT,
                    sender_id=self.peer_id,
                )
                await self.send(heartbeat)
        except (asyncio.CancelledError, ConnectionError):
            pass
