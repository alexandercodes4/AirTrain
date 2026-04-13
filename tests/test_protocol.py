"""Tests for the network protocol."""

from __future__ import annotations

import pytest


def test_message_encode_decode():
    """Test message round-trip serialization."""
    from airtrain.network.protocol import Message, MessageType, decode_message, encode_message

    msg = Message(
        msg_type=MessageType.HANDSHAKE,
        sender_id="peer123",
        metadata={"chip": "M4", "memory_gb": 32},
        payload=b"hello world",
    )

    encoded = encode_message(msg)
    assert isinstance(encoded, bytes)

    decoded = decode_message(encoded)
    assert decoded.msg_type == MessageType.HANDSHAKE
    assert decoded.sender_id == "peer123"
    assert decoded.metadata["chip"] == "M4"
    assert decoded.payload == b"hello world"


def test_message_no_payload():
    """Test message with no payload."""
    from airtrain.network.protocol import Message, MessageType, decode_message, encode_message

    msg = Message(
        msg_type=MessageType.HEARTBEAT,
        sender_id="peer456",
    )

    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert decoded.msg_type == MessageType.HEARTBEAT
    assert decoded.payload == b""


def test_message_large_payload():
    """Test message with large payload."""
    from airtrain.network.protocol import Message, MessageType, decode_message, encode_message

    payload = b"x" * 1_000_000  # 1MB
    msg = Message(
        msg_type=MessageType.SYNC_GRADIENTS,
        sender_id="peer789",
        payload=payload,
    )

    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert len(decoded.payload) == 1_000_000
