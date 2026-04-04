"""Base trainer wrapping an MLX model and optimizer."""

from __future__ import annotations

import logging
import time
from typing import Any

from airtrain.compat import MLX_AVAILABLE
from airtrain.config import TrainingConfig

logger = logging.getLogger(__name__)

if MLX_AVAILABLE:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim


class BaseTrainer:
    """Base trainer for MLX models with DiLoCo support."""

    def __init__(self, model: Any, config: TrainingConfig):
        from airtrain.compat import require_mlx

        require_mlx()

        self.model = model
        self.config = config
        self.step = 0
        self.total_loss = 0.0
        self._step_count_for_avg = 0
        self._last_time = time.time()
        self._tokens_since_last = 0

        self.optimizer = optim.AdamW(
            learning_rate=config.diloco.inner_lr,
            weight_decay=config.diloco.inner_weight_decay,
        )

        self.loss_and_grad_fn = nn.value_and_grad(model, self._loss_fn)

    def _loss_fn(self, model, batch_x, batch_y):
        from airtrain.models.transformer import cross_entropy_loss

        logits = model(batch_x)
        return cross_entropy_loss(logits, batch_y)

    def train_step(self, batch_x, batch_y) -> float:
        """Run a single training step. Returns loss value."""
        loss, grads = self.loss_and_grad_fn(self.model, batch_x, batch_y)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)

        loss_val = loss.item()
        self.step += 1
        self.total_loss += loss_val
        self._step_count_for_avg += 1
        self._tokens_since_last += batch_x.size

        return loss_val

    def get_parameters(self) -> dict[str, Any]:
        """Get model parameters as a flat dict of arrays."""
        return _flatten_params(self.model.parameters())

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load parameters into the model."""
        nested = _unflatten_params(params)
        self.model.load_weights(list(_nested_to_pairs(nested)))
        mx.eval(self.model.parameters())

    def get_parameter_diff(
        self, original: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute pseudo-gradients: original - current."""
        diff = {}
        for key in original:
            if key in current:
                diff[key] = original[key] - current[key]
        return diff

    @property
    def avg_loss(self) -> float:
        if self._step_count_for_avg == 0:
            return 0.0
        return self.total_loss / self._step_count_for_avg

    @property
    def throughput(self) -> float:
        """Tokens per second."""
        now = time.time()
        elapsed = now - self._last_time
        if elapsed == 0:
            return 0.0
        tps = self._tokens_since_last / elapsed
        self._last_time = now
        self._tokens_since_last = 0
        return tps

    def reset_metrics(self):
        self.total_loss = 0.0
        self._step_count_for_avg = 0


def _flatten_params(params: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested parameter dict to dot-separated keys."""
    flat = {}
    for key, val in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            flat.update(_flatten_params(val, full_key))
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    flat.update(_flatten_params(item, f"{full_key}.{i}"))
                else:
                    flat[f"{full_key}.{i}"] = item
        else:
            flat[full_key] = val
    return flat


def _unflatten_params(flat: dict[str, Any]) -> dict:
    """Unflatten dot-separated keys back to nested dict."""
    nested: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = val
    return nested


def _nested_to_pairs(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Convert nested dict to list of (key, value) pairs for load_weights."""
    pairs = []
    for key, val in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            pairs.extend(_nested_to_pairs(val, full_key))
        else:
            pairs.append((full_key, val))
    return pairs
