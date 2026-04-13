"""Tests for AirTrain configuration models."""

from __future__ import annotations

from airtrain.config import (
    CheckpointMeta,
    DiLoCoConfig,
    NetworkConfig,
    PeerInfo,
    PeerRole,
    PeerStatus,
    TrainingConfig,
)


def test_peer_info_defaults():
    peer = PeerInfo()
    assert len(peer.peer_id) == 12
    assert peer.status == PeerStatus.IDLE
    assert peer.role == PeerRole.WORKER
    assert peer.port == 7471


def test_peer_info_custom():
    peer = PeerInfo(
        peer_id="abc123",
        chip="Apple M4 Max",
        memory_gb=128.0,
        tflops=18.43,
        role=PeerRole.COORDINATOR,
    )
    assert peer.peer_id == "abc123"
    assert peer.chip == "Apple M4 Max"
    assert peer.tflops == 18.43


def test_diloco_config_defaults():
    config = DiLoCoConfig()
    assert config.inner_steps == 500
    assert config.inner_lr == 3e-4
    assert config.outer_lr == 0.7
    assert config.outer_momentum == 0.9
    assert config.use_nesterov is True


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.model_name == "gpt2-small"
    assert config.batch_size == 8
    assert config.diloco.inner_steps == 500


def test_training_config_serialization():
    config = TrainingConfig(model_name="gpt2-tiny", batch_size=4)
    data = config.model_dump_json()
    restored = TrainingConfig.model_validate_json(data)
    assert restored.model_name == "gpt2-tiny"
    assert restored.batch_size == 4


def test_checkpoint_meta():
    meta = CheckpointMeta(
        model_name="gpt2-small",
        global_step=5000,
        loss=2.5,
        contributors=["alice", "bob"],
        total_compute_hours=12.5,
    )
    assert meta.global_step == 5000
    assert len(meta.contributors) == 2


def test_network_config():
    config = NetworkConfig()
    assert config.listen_port == 7471
    assert config.use_mdns is True
    assert config.mdns_service_type == "_airtrain._tcp.local."
