"""Pipeline parallel engine — v2 stub.

Splits model layers across multiple peers for training models
that don't fit on a single device. Each peer holds a contiguous
slice of layers and forwards activations to the next peer.

This module defines the interface; implementation is deferred to v2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from airtrain.config import PeerInfo

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """A stage in the pipeline, assigned to a specific peer."""

    peer: PeerInfo
    layer_start: int
    layer_end: int
    estimated_tflops: float = 0.0

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallel training."""

    num_microbatches: int = 4
    interleave: bool = False


@dataclass
class PipelineEngine:
    """Pipeline parallel training engine.

    Assigns model layers to peers proportional to their compute
    capability (TFLOPS), then coordinates activation forwarding
    between stages.

    Status: INTERFACE ONLY — implementation in v2.
    """

    config: PipelineConfig = field(default_factory=PipelineConfig)
    stages: list[PipelineStage] = field(default_factory=list)

    def assign_layers(self, total_layers: int, peers: list[PeerInfo]) -> list[PipelineStage]:
        """Assign model layers to peers proportional to their TFLOPS.

        Args:
            total_layers: Total number of model layers.
            peers: Available peers with their compute capabilities.

        Returns:
            List of PipelineStage assignments.
        """
        if not peers:
            raise ValueError("No peers available for pipeline assignment")

        total_tflops = sum(max(p.tflops, 0.1) for p in peers)
        stages = []
        current_layer = 0

        for i, peer in enumerate(peers):
            peer_tflops = max(peer.tflops, 0.1)
            fraction = peer_tflops / total_tflops

            if i == len(peers) - 1:
                num_layers = total_layers - current_layer
            else:
                num_layers = max(1, round(total_layers * fraction))

            stage = PipelineStage(
                peer=peer,
                layer_start=current_layer,
                layer_end=current_layer + num_layers,
                estimated_tflops=peer_tflops,
            )
            stages.append(stage)
            current_layer += num_layers

        self.stages = stages
        logger.info(
            "Pipeline assignment: %s",
            [(s.peer.hostname, f"layers {s.layer_start}-{s.layer_end}") for s in stages],
        )
        return stages

    async def forward(self, batch):
        """Forward pass through the pipeline. (v2)"""
        raise NotImplementedError("Pipeline parallel forward is not yet implemented (v2)")

    async def backward(self):
        """Backward pass through the pipeline. (v2)"""
        raise NotImplementedError("Pipeline parallel backward is not yet implemented (v2)")
