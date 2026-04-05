"""DiLoCo (Distributed Low-Communication) training engine.

Implements the DiLoCo algorithm: each worker trains independently for H steps,
then pseudo-gradients are averaged and an outer optimizer step is applied.
This reduces communication by ~500x compared to standard distributed training.

Reference: https://arxiv.org/abs/2407.07852
"""

from __future__ import annotations

import logging
from typing import Any

from airtrain.compat import MLX_AVAILABLE
from airtrain.config import DiLoCoConfig

logger = logging.getLogger(__name__)

if MLX_AVAILABLE:
    import mlx.core as mx


class DiLoCoEngine:
    """Orchestrates DiLoCo distributed training."""

    def __init__(self, config: DiLoCoConfig):
        self.config = config
        self.outer_momentum: dict[str, Any] = {}
        self.original_params: dict[str, Any] = {}
        self.outer_step = 0

    def snapshot_params(self, params: dict[str, Any]) -> None:
        """Save a snapshot of parameters before inner training loop."""
        self.original_params = {k: mx.array(v) for k, v in params.items()}
        mx.eval(self.original_params)

    def compute_pseudo_gradients(
        self, original: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute pseudo-gradients: θ_original - θ_current."""
        grads = {}
        for key in original:
            if key in current:
                grads[key] = original[key] - current[key]
        return grads

    def apply_outer_step(
        self, pseudo_gradients_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Average pseudo-gradients and apply outer SGD+Nesterov update.

        Args:
            pseudo_gradients_list: List of pseudo-gradient dicts from each worker.

        Returns:
            Updated parameters.
        """
        if not pseudo_gradients_list:
            raise ValueError("No pseudo-gradients to average")

        n_workers = len(pseudo_gradients_list)
        keys = pseudo_gradients_list[0].keys()

        # Average pseudo-gradients across workers
        avg_grads = {}
        for key in keys:
            stacked = mx.stack([pg[key] for pg in pseudo_gradients_list if key in pg])
            avg_grads[key] = mx.mean(stacked, axis=0)

        # SGD + Nesterov momentum update
        lr = self.config.outer_lr
        beta = self.config.outer_momentum
        new_params = {}

        for key in keys:
            grad = avg_grads[key]

            # Initialize momentum if needed
            if key not in self.outer_momentum:
                self.outer_momentum[key] = mx.zeros_like(grad)

            # Update momentum: m = β * m + g
            self.outer_momentum[key] = beta * self.outer_momentum[key] + grad

            # Nesterov: update = g + β * m
            if self.config.use_nesterov:
                update = grad + beta * self.outer_momentum[key]
            else:
                update = self.outer_momentum[key]

            # Apply update: θ = θ_original - lr * update
            new_params[key] = self.original_params[key] - lr * update

        mx.eval(new_params)
        self.outer_step += 1

        logger.info(
            "Outer step %d: averaged %d workers, lr=%.3f",
            self.outer_step,
            n_workers,
            lr,
        )

        return new_params

    def params_to_numpy(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert MLX arrays to numpy for serialization."""
        import numpy as np

        return {k: np.array(v) for k, v in params.items()}

    def numpy_to_params(self, arrays: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy arrays back to MLX arrays."""
        return {k: mx.array(v) for k, v in arrays.items()}
