"""Checkpoint management for AirTrain.

Saves and loads training state: model weights (safetensors),
optimizer state (npz), and metadata (JSON).
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from airtrain.config import CheckpointMeta

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    params: dict[str, np.ndarray],
    optimizer_state: dict[str, Any] | None,
    meta: CheckpointMeta,
) -> Path:
    """Save a training checkpoint.

    Args:
        path: Directory to save the checkpoint.
        params: Model parameters as numpy arrays.
        optimizer_state: Optimizer state dict (optional).
        meta: Checkpoint metadata.

    Returns:
        Path to the saved checkpoint directory.
    """
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights_path = checkpoint_dir / "model.npz"
    np.savez(str(weights_path), **params)

    # Save optimizer state
    if optimizer_state:
        opt_path = checkpoint_dir / "optimizer.npz"
        flat_opt = _flatten_optimizer_state(optimizer_state)
        if flat_opt:
            np.savez(str(opt_path), **flat_opt)

    # Save metadata
    meta.created_at = datetime.now(timezone.utc).isoformat()
    meta_path = checkpoint_dir / "metadata.json"
    meta_path.write_text(meta.model_dump_json(indent=2))

    logger.info(
        "Checkpoint saved: step=%d, loss=%.4f, path=%s",
        meta.global_step,
        meta.loss,
        checkpoint_dir,
    )
    return checkpoint_dir


def load_checkpoint(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], Optional[dict], CheckpointMeta]:
    """Load a training checkpoint.

    Returns:
        Tuple of (model_params, optimizer_state, metadata).
    """
    checkpoint_dir = Path(path)

    # Load weights
    weights_path = checkpoint_dir / "model.npz"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model weights found at {weights_path}")
    npz = np.load(str(weights_path))
    params = {name: npz[name] for name in npz.files}

    # Load optimizer state
    optimizer_state = None
    opt_path = checkpoint_dir / "optimizer.npz"
    if opt_path.exists():
        opt_npz = np.load(str(opt_path))
        optimizer_state = {name: opt_npz[name] for name in opt_npz.files}

    # Load metadata
    meta_path = checkpoint_dir / "metadata.json"
    if meta_path.exists():
        meta = CheckpointMeta.model_validate_json(meta_path.read_text())
    else:
        meta = CheckpointMeta()

    logger.info(
        "Checkpoint loaded: step=%d, loss=%.4f, path=%s",
        meta.global_step,
        meta.loss,
        checkpoint_dir,
    )
    return params, optimizer_state, meta


def export_relay(
    checkpoint_path: str | Path,
    output_path: str | Path,
    description: str = "",
) -> Path:
    """Export a checkpoint as a portable relay bundle.

    Copies the checkpoint and adds relay-specific metadata
    so another trainer can pick it up and continue.
    """
    src = Path(checkpoint_path)
    dst = Path(output_path)

    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    # Copy checkpoint files
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # Update metadata with relay info
    meta_path = dst / "metadata.json"
    if meta_path.exists():
        meta = CheckpointMeta.model_validate_json(meta_path.read_text())
    else:
        meta = CheckpointMeta()

    meta.description = description or f"Relay checkpoint at step {meta.global_step}"
    meta.created_at = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(meta.model_dump_json(indent=2))

    # Add relay marker
    relay_info = {
        "format": "airtrain-relay-v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "description": meta.description,
        "model_name": meta.model_name,
        "global_step": meta.global_step,
        "contributors": meta.contributors,
        "total_compute_hours": meta.total_compute_hours,
    }
    relay_path = dst / "relay.json"
    relay_path.write_text(json.dumps(relay_info, indent=2))

    logger.info("Relay checkpoint exported to %s", dst)
    return dst


def import_relay(relay_path: str | Path) -> CheckpointMeta:
    """Import and validate a relay checkpoint.

    Returns:
        Checkpoint metadata.
    """
    path = Path(relay_path)

    meta_path = path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found in relay checkpoint: {path}")

    meta = CheckpointMeta.model_validate_json(meta_path.read_text())

    weights_path = path / "model.npz"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model weights in relay checkpoint: {path}")

    logger.info(
        "Relay checkpoint imported: %s at step %d (%.1f compute hours, %d contributors)",
        meta.model_name,
        meta.global_step,
        meta.total_compute_hours,
        len(meta.contributors),
    )
    return meta


def _flatten_optimizer_state(state: dict) -> dict[str, np.ndarray]:
    """Flatten optimizer state for serialization."""
    flat = {}
    for i, item in enumerate(state) if isinstance(state, list) else state.items():
        key = str(i) if isinstance(state, list) else str(i)
        if isinstance(item, np.ndarray):
            flat[key] = item
        elif isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    flat[f"{key}.{k}"] = v
    return flat
