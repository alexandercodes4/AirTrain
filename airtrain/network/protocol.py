"""Message protocol for AirTrain peer communication.

Wire format: [4-byte payload length (big-endian)] [JSON header] [binary payload]
The JSON header is separated from the binary payload by a null byte.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MessageType(IntEnum):
    HANDSHAKE = 1
    HEARTBEAT = 2
    SYNC_REQUEST = 3
    SYNC_GRADIENTS = 4
    PEER_JOIN = 5
    PEER_LEAVE = 6
    CHECKPOINT_META = 7
    MODEL_WEIGHTS = 8
    STATUS = 9
    ACK = 10


@dataclass
class Message:
    """A message exchanged between peers."""

    msg_type: MessageType
    sender_id: str
    payload: bytes = b""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def payload_size(self) -> int:
        return len(self.payload)


def encode_message(msg: Message) -> bytes:
    """Encode a message to bytes for transmission.

    Format: [4-byte total length] [JSON header \\0 binary payload]
    """
    header = {
        "type": int(msg.msg_type),
        "sender": msg.sender_id,
        "payload_size": msg.payload_size,
        "metadata": msg.metadata,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = header_bytes + b"\x00" + msg.payload
    length_prefix = struct.pack("!I", len(body))
    return length_prefix + body


def decode_message(data: bytes) -> Message:
    """Decode a message from bytes.

    Expects the full message including the 4-byte length prefix.
    """
    if len(data) < 4:
        raise ValueError("Message too short: missing length prefix")

    body_length = struct.unpack("!I", data[:4])[0]
    body = data[4 : 4 + body_length]

    null_idx = body.index(b"\x00")
    header_bytes = body[:null_idx]
    payload = body[null_idx + 1 :]

    header = json.loads(header_bytes)

    return Message(
        msg_type=MessageType(header["type"]),
        sender_id=header["sender"],
        metadata=header.get("metadata", {}),
        payload=payload,
    )


async def read_message(reader) -> Message | None:
    """Read a single message from an asyncio StreamReader."""
    length_data = await reader.readexactly(4)
    if not length_data:
        return None

    body_length = struct.unpack("!I", length_data)[0]
    body = await reader.readexactly(body_length)

    return decode_message(length_data + body)


async def write_message(writer, msg: Message) -> None:
    """Write a single message to an asyncio StreamWriter."""
    data = encode_message(msg)
    writer.write(data)
    await writer.drain()
