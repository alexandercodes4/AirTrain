"""Configuration models for AirTrain."""

from __future__ import annotations

import platform
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class PeerStatus(str, Enum):
    IDLE = "idle"
    TRAINING = "training"
    SYNCING = "syncing"
    PAUSED = "paused"
    DISCONNECTED = "disconnected"


class PeerRole(str, Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"


class PeerInfo(BaseModel):
    """Information about a peer in the training swarm."""

    peer_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    hostname: str = Field(default_factory=platform.node)
    ip_address: str = ""
    port: int = 7471
    chip: str = ""
    memory_gb: float = 0.0
    tflops: float = 0.0
    status: PeerStatus = PeerStatus.IDLE
    role: PeerRole = PeerRole.WORKER
    current_step: int = 0
    compute_hours: float = 0.0


class DiLoCoConfig(BaseModel):
    """Configuration for the DiLoCo training algorithm."""

    inner_steps: int = 500
    inner_lr: float = 3e-4
    inner_optimizer: str = "adamw"
    inner_weight_decay: float = 0.1
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    use_nesterov: bool = True
    gradient_compression: bool = True
    compress_to_fp16: bool = True


class TrainingConfig(BaseModel):
    """Top-level training configuration."""

    model_name: str = "gpt2-small"
    dataset_path: str = ""
    batch_size: int = 8
    max_steps: int = 100_000
    seq_length: int = 512
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 1000
    log_every: int = 10
    diloco: DiLoCoConfig = Field(default_factory=DiLoCoConfig)
    dashboard_port: int = 8471
    enable_dashboard: bool = False
    seed: int = 42


class CheckpointMeta(BaseModel):
    """Metadata stored alongside a checkpoint."""

    version: str = "0.1.0"
    model_name: str = ""
    global_step: int = 0
    inner_step: int = 0
    loss: float = 0.0
    total_compute_hours: float = 0.0
    contributors: list[str] = Field(default_factory=list)
    config: Optional[TrainingConfig] = None
    created_at: str = ""
    description: str = ""


class SleepConfig(BaseModel):
    """Configuration for Sleep Swarm — automatic overnight training."""

    window_start: str = "23:00"
    window_end: str = "07:00"
    timezone: str = ""  # auto-detected from system if empty
    max_hours: float = 8.0
    prefer_model: str = "any"
    min_battery: int = 20
    relay_url: str = "https://airtrain.dev/api/relay"
    auto_checkpoint: bool = True
    retry_on_disconnect: bool = True
    max_retries: int = 3


class DreamConfig(BaseModel):
    """Configuration for Dream Training — synthetic data generation during idle time."""

    enabled: bool = True
    samples_per_session: int = 1000
    min_length: int = 64
    max_length: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    quality_threshold: float = 0.7
    mix_ratio: float = 0.15
    dream_dir: str = "./dreams"
    max_cache_mb: int = 500
    dream_interval: int = 60
    share_dreams: bool = True


class MarketplaceConfig(BaseModel):
    """Configuration for Gradient Marketplace — quality-weighted gradient aggregation."""

    enabled: bool = True
    score_magnitude: float = 0.25
    score_alignment: float = 0.35
    score_history: float = 0.25
    score_improvement: float = 0.15
    min_weight: float = 0.1
    history_window: int = 10
    warmup_rounds: int = 3


class AutopsyConfig(BaseModel):
    """Configuration for Model Autopsy reports."""

    enabled: bool = True
    output_dir: str = "./autopsy"
    output_format: str = "html"
    track_per_round: bool = True
    track_dreams: bool = True
    track_contributors: bool = True


class NetworkConfig(BaseModel):
    """Network configuration for peer communication."""

    listen_host: str = "0.0.0.0"
    listen_port: int = 7471
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 15.0
    relay_server_url: Optional[str] = None
    use_mdns: bool = True
    mdns_service_type: str = "_airtrain._tcp.local."
