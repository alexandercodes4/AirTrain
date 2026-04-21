"""Gradient Marketplace — quality-weighted gradient aggregation.

Scores each worker's pseudo-gradient contribution based on magnitude,
alignment with consensus, historical consistency, and loss improvement
correlation. Workers with higher-quality gradients get more influence
in the aggregation step.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from airtrain.config import MarketplaceConfig


@dataclass
class WorkerScore:
    """Score breakdown for a single worker in a sync round."""

    peer_id: str
    magnitude_score: float = 0.0
    alignment_score: float = 0.0
    history_score: float = 0.0
    improvement_score: float = 0.0
    total_score: float = 0.0
    weight: float = 0.0
    rank: int = 0


class GradientMarketplace:
    """Scores and ranks worker gradient contributions."""

    def __init__(self, config: MarketplaceConfig | None = None):
        self.config = config or MarketplaceConfig()
        self.round_num = 0
        # Rolling history: peer_id -> list of (score, loss_delta)
        self.history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # Last round's scores for reporting
        self.last_scores: list[WorkerScore] = []

    def score_gradients(
        self,
        pseudo_gradients: dict[str, dict[str, np.ndarray]],
        round_num: int,
    ) -> dict[str, float]:
        """Score each worker's gradients and return normalized weights.

        Args:
            pseudo_gradients: {peer_id: {param_name: gradient_array}}
            round_num: Current sync round number.

        Returns:
            {peer_id: weight} where weights sum to 1.0
        """
        self.round_num = round_num
        peer_ids = list(pseudo_gradients.keys())
        n_workers = len(peer_ids)

        if n_workers == 0:
            return {}

        # During warmup, use equal weights
        if round_num < self.config.warmup_rounds:
            equal_w = 1.0 / n_workers
            self.last_scores = [
                WorkerScore(peer_id=pid, total_score=1.0, weight=equal_w, rank=i + 1)
                for i, pid in enumerate(peer_ids)
            ]
            return {pid: equal_w for pid in peer_ids}

        # Compute average gradient for alignment scoring
        avg_grads = self._compute_average(pseudo_gradients)

        # Score each worker
        scores: list[WorkerScore] = []
        for pid in peer_ids:
            grads = pseudo_gradients[pid]
            ws = WorkerScore(peer_id=pid)
            ws.magnitude_score = self._score_magnitude(grads, pseudo_gradients)
            ws.alignment_score = self._score_alignment(grads, avg_grads)
            ws.history_score = self._score_history(pid)
            ws.improvement_score = self._score_improvement(pid)

            # Weighted combination
            c = self.config
            ws.total_score = (
                c.score_magnitude * ws.magnitude_score
                + c.score_alignment * ws.alignment_score
                + c.score_history * ws.history_score
                + c.score_improvement * ws.improvement_score
            )
            scores.append(ws)

        # Apply minimum weight floor and normalize
        raw_scores = np.array([s.total_score for s in scores])
        raw_scores = np.clip(raw_scores, 0.01, None)  # avoid zero

        # Normalize to weights that sum to 1.0
        weights = raw_scores / raw_scores.sum()

        # Apply minimum weight floor
        weights = np.clip(weights, self.config.min_weight, None)
        weights = weights / weights.sum()  # re-normalize after clipping

        # Assign weights and ranks
        ranked_indices = np.argsort(-weights)
        for rank, idx in enumerate(ranked_indices):
            scores[idx].weight = float(weights[idx])
            scores[idx].rank = rank + 1

        self.last_scores = sorted(scores, key=lambda s: s.rank)

        return {s.peer_id: s.weight for s in scores}

    def update_history(self, peer_id: str, score: float, loss_delta: float):
        """Record a worker's score and the resulting loss change.

        Args:
            peer_id: Worker identifier.
            score: Worker's total score this round.
            loss_delta: Change in loss after this round (negative = improved).
        """
        history = self.history[peer_id]
        history.append((score, loss_delta))
        # Trim to window
        if len(history) > self.config.history_window:
            self.history[peer_id] = history[-self.config.history_window:]

    def get_rankings(self) -> list[WorkerScore]:
        """Return the latest worker rankings sorted by rank."""
        return list(self.last_scores)

    def get_summary(self) -> dict:
        """Return a summary dict for logging/display."""
        if not self.last_scores:
            return {"round": self.round_num, "workers": 0, "rankings": []}

        return {
            "round": self.round_num,
            "workers": len(self.last_scores),
            "rankings": [
                {
                    "rank": s.rank,
                    "peer_id": s.peer_id,
                    "weight": round(s.weight, 4),
                    "total": round(s.total_score, 4),
                    "magnitude": round(s.magnitude_score, 3),
                    "alignment": round(s.alignment_score, 3),
                    "history": round(s.history_score, 3),
                    "improvement": round(s.improvement_score, 3),
                }
                for s in self.last_scores
            ],
        }

    # ── Scoring Functions ──────────────────────────────────────────

    def _compute_average(
        self, pseudo_gradients: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Compute the simple mean of all workers' gradients."""
        peer_ids = list(pseudo_gradients.keys())
        if not peer_ids:
            return {}

        keys = list(pseudo_gradients[peer_ids[0]].keys())
        avg = {}
        for key in keys:
            arrays = [pseudo_gradients[pid][key] for pid in peer_ids if key in pseudo_gradients[pid]]
            if arrays:
                avg[key] = np.mean(arrays, axis=0)
        return avg

    def _score_magnitude(
        self,
        grads: dict[str, np.ndarray],
        all_grads: dict[str, dict[str, np.ndarray]],
    ) -> float:
        """Score based on gradient magnitude being in a healthy range.

        Too small = stale/not learning. Too large = diverging.
        Score peaks when magnitude is near the median of all workers.
        """
        # Compute this worker's L2 norm
        worker_norm = self._grad_norm(grads)

        # Compute all workers' norms
        all_norms = [self._grad_norm(g) for g in all_grads.values()]
        if not all_norms or worker_norm == 0:
            return 0.5

        median_norm = float(np.median(all_norms))
        if median_norm == 0:
            return 0.5

        # Score: 1.0 at median, drops off as ratio deviates from 1.0
        ratio = worker_norm / median_norm
        # Gaussian-like scoring around ratio=1.0
        score = math.exp(-2.0 * (ratio - 1.0) ** 2)
        return float(np.clip(score, 0.0, 1.0))

    def _score_alignment(
        self, grads: dict[str, np.ndarray], avg_grads: dict[str, np.ndarray]
    ) -> float:
        """Score based on cosine similarity with the average gradient.

        High alignment = gradient agrees with consensus = likely good data.
        """
        # Flatten both to single vectors
        worker_flat = self._flatten(grads)
        avg_flat = self._flatten(avg_grads)

        if len(worker_flat) == 0 or len(avg_flat) == 0:
            return 0.5

        # Cosine similarity
        dot = np.dot(worker_flat, avg_flat)
        norm_w = np.linalg.norm(worker_flat)
        norm_a = np.linalg.norm(avg_flat)

        if norm_w == 0 or norm_a == 0:
            return 0.5

        cosine_sim = float(dot / (norm_w * norm_a))
        # Map from [-1, 1] to [0, 1]
        score = (cosine_sim + 1.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def _score_history(self, peer_id: str) -> float:
        """Score based on historical consistency.

        New workers get neutral score (0.5). Workers with consistent
        high scores get a bonus. Workers with erratic scores get penalized.
        """
        history = self.history.get(peer_id, [])
        if not history:
            return 0.5  # neutral for new workers

        past_scores = [h[0] for h in history]
        avg = np.mean(past_scores)
        return float(np.clip(avg, 0.0, 1.0))

    def _score_improvement(self, peer_id: str) -> float:
        """Score based on correlation with loss improvement.

        Workers whose gradients historically correlated with loss
        decreases get higher scores.
        """
        history = self.history.get(peer_id, [])
        if len(history) < 2:
            return 0.5  # neutral until we have enough data

        # Count how often this worker's high score correlated with loss decrease
        improvements = 0
        total = 0
        for score, loss_delta in history:
            total += 1
            if loss_delta < 0:  # loss decreased = good
                improvements += 1

        improvement_rate = improvements / total
        return float(np.clip(improvement_rate, 0.0, 1.0))

    # ── Utilities ──────────────────────────────────────────────────

    def _grad_norm(self, grads: dict[str, np.ndarray]) -> float:
        """Compute L2 norm of all gradient arrays combined."""
        total = 0.0
        for arr in grads.values():
            total += float(np.sum(arr.astype(np.float64) ** 2))
        return math.sqrt(total)

    def _flatten(self, grads: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten all gradient arrays into a single vector."""
        arrays = []
        for key in sorted(grads.keys()):
            arrays.append(grads[key].ravel().astype(np.float32))
        if not arrays:
            return np.array([], dtype=np.float32)
        return np.concatenate(arrays)
