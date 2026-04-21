"""Dream Training — synthetic data generation during idle time.

Between active training sessions, Macs run low-priority inference to generate
synthetic training data from the current model checkpoint. The Mac "dreams" —
replaying and generating variations of what the model has learned, scoring them
for quality, and caching the best examples. Dream data is shared across the
swarm and mixed into the next training round to accelerate convergence.

Inspired by how the human brain consolidates learning during sleep through replay.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from airtrain.config import DreamConfig

logger = logging.getLogger("airtrain.dream")


@dataclass
class DreamSample:
    """A single synthetic training sample generated during dreaming."""

    text: str
    score: float = 0.0
    source_peer: str = ""
    model_step: int = 0
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": self.score,
            "source_peer": self.source_peer,
            "step": self.model_step,
            "ts": self.timestamp,
            "tokens": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DreamSample:
        return cls(
            text=data["text"],
            score=data.get("score", 0.0),
            source_peer=data.get("source_peer", ""),
            model_step=data.get("step", 0),
            timestamp=data.get("ts", 0.0),
            token_count=data.get("tokens", 0),
        )


class DreamScorer:
    """Evaluates quality of generated dream samples.

    Quality heuristic based on perplexity:
    - Too low perplexity = memorized / repetitive (bad)
    - Too high perplexity = gibberish / incoherent (bad)
    - Sweet spot = novel but coherent (good)

    Also rejects:
    - Highly repetitive samples (n-gram overlap)
    - Very short samples
    - Samples with degenerate patterns
    """

    def __init__(self, config: DreamConfig):
        self.config = config
        self._perplexity_low = 5.0    # below this = memorized
        self._perplexity_high = 100.0  # above this = gibberish

    def score(self, text: str, loss: Optional[float] = None) -> float:
        """Score a dream sample from 0.0 (reject) to 1.0 (excellent).

        Args:
            text: The generated text sample.
            loss: Per-token cross-entropy loss from the model (if available).
                  Used to compute perplexity = exp(loss).
        """
        scores = []

        # Length score — prefer samples in the sweet spot
        token_count = len(text.split())
        if token_count < 10:
            return 0.0  # too short, reject
        length_score = min(1.0, token_count / 50)  # ramp up to 50 tokens
        if token_count > 400:
            length_score *= 0.9  # slight penalty for very long
        scores.append(length_score)

        # Repetition score — penalize n-gram repetition
        rep_score = self._repetition_score(text)
        if rep_score < 0.2:
            return 0.0  # too repetitive, reject
        scores.append(rep_score)

        # Perplexity score (if loss available)
        if loss is not None:
            ppl = math.exp(min(loss, 20.0))  # cap to avoid overflow
            ppl_score = self._perplexity_score(ppl)
            scores.append(ppl_score)

        # Diversity score — penalize degenerate patterns
        div_score = self._diversity_score(text)
        scores.append(div_score)

        # Weighted average
        return sum(scores) / len(scores)

    def _repetition_score(self, text: str) -> float:
        """Score based on n-gram diversity. 1.0 = no repetition, 0.0 = all repeated."""
        words = text.lower().split()
        if len(words) < 4:
            return 0.0

        # Check bigram repetition
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        if not bigrams:
            return 0.0
        unique_ratio = len(set(bigrams)) / len(bigrams)

        # Check trigram repetition
        trigrams = [(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)]
        if trigrams:
            tri_ratio = len(set(trigrams)) / len(trigrams)
            unique_ratio = (unique_ratio + tri_ratio) / 2

        return unique_ratio

    def _perplexity_score(self, ppl: float) -> float:
        """Score based on perplexity. Sweet spot = moderate perplexity."""
        if ppl < self._perplexity_low:
            return ppl / self._perplexity_low  # too easy
        elif ppl > self._perplexity_high:
            return max(0.0, 1.0 - (ppl - self._perplexity_high) / self._perplexity_high)
        else:
            # In the sweet spot — peak at geometric mean
            log_range = math.log(self._perplexity_high) - math.log(self._perplexity_low)
            log_pos = math.log(ppl) - math.log(self._perplexity_low)
            # Bell curve centered in log space
            center = log_range / 2
            return math.exp(-2 * ((log_pos - center) / log_range) ** 2)

    def _diversity_score(self, text: str) -> float:
        """Penalize degenerate patterns like repeated punctuation or single chars."""
        if not text.strip():
            return 0.0

        # Check character diversity
        chars = set(text)
        if len(chars) < 10:
            return 0.2  # very low character diversity

        # Check for repeated lines
        lines = text.strip().split("\n")
        if len(lines) > 1:
            unique_lines = len(set(lines))
            if unique_lines / len(lines) < 0.5:
                return 0.3  # too many repeated lines

        # Check for excessive punctuation
        alpha_count = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha_count / len(text) < 0.4:
            return 0.4  # too much non-alpha content

        return 1.0


class DreamCache:
    """Manages local dream sample storage.

    Dreams are stored as JSONL files in the dream directory.
    Supports quality-weighted sampling for training batch mixing.
    Auto-prunes when exceeding the max cache size.
    """

    def __init__(self, config: DreamConfig, peer_id: str = "local"):
        self.config = config
        self.peer_id = peer_id
        self.dream_dir = Path(config.dream_dir)
        self.dream_dir.mkdir(parents=True, exist_ok=True)
        self._samples: list[DreamSample] = []
        self._loaded = False

    def _load_if_needed(self) -> None:
        """Lazy-load all dream samples from disk."""
        if self._loaded:
            return
        self._samples = []
        for path in sorted(self.dream_dir.glob("dreams_*.jsonl")):
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._samples.append(DreamSample.from_dict(json.loads(line)))
            except Exception as e:
                logger.warning(f"Failed to load dream file {path}: {e}")
        self._loaded = True
        logger.debug(f"Loaded {len(self._samples)} dream samples from cache")

    def add(self, samples: list[DreamSample]) -> int:
        """Add scored samples to the cache. Returns count of samples kept."""
        if not samples:
            return 0

        self._load_if_needed()

        # Filter by quality threshold
        kept = [s for s in samples if s.score >= self.config.quality_threshold]
        if not kept:
            return 0

        # Write to a new JSONL file
        timestamp = int(time.time())
        filename = f"dreams_{self.peer_id}_{timestamp}.jsonl"
        filepath = self.dream_dir / filename

        with open(filepath, "w") as f:
            for sample in kept:
                f.write(json.dumps(sample.to_dict()) + "\n")

        self._samples.extend(kept)
        logger.info(f"Cached {len(kept)}/{len(samples)} dream samples to {filename}")

        # Auto-prune if over size limit
        self._prune_if_needed()

        return len(kept)

    def _prune_if_needed(self) -> None:
        """Remove oldest dream files if cache exceeds max size."""
        total_size = sum(
            f.stat().st_size for f in self.dream_dir.glob("dreams_*.jsonl")
        )
        max_bytes = self.config.max_cache_mb * 1024 * 1024

        if total_size <= max_bytes:
            return

        # Sort files by modification time (oldest first) and delete until under limit
        files = sorted(
            self.dream_dir.glob("dreams_*.jsonl"),
            key=lambda f: f.stat().st_mtime,
        )
        for f in files:
            if total_size <= max_bytes:
                break
            size = f.stat().st_size
            f.unlink()
            total_size -= size
            logger.debug(f"Pruned old dream file: {f.name}")

        # Reload cache
        self._loaded = False

    def get_training_batch(self, n: int) -> list[str]:
        """Sample n dream texts, weighted by quality score.

        Higher-quality dreams are more likely to be selected.
        """
        self._load_if_needed()
        if not self._samples:
            return []

        n = min(n, len(self._samples))
        weights = [s.score for s in self._samples]
        total = sum(weights)
        if total == 0:
            selected = random.sample(self._samples, n)
        else:
            selected = random.choices(self._samples, weights=weights, k=n)

        return [s.text for s in selected]

    @property
    def total_samples(self) -> int:
        self._load_if_needed()
        return len(self._samples)

    @property
    def avg_quality(self) -> float:
        self._load_if_needed()
        if not self._samples:
            return 0.0
        return sum(s.score for s in self._samples) / len(self._samples)

    @property
    def cache_size_mb(self) -> float:
        total = sum(
            f.stat().st_size for f in self.dream_dir.glob("dreams_*.jsonl")
            if f.exists()
        )
        return total / (1024 * 1024)

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "total_samples": self.total_samples,
            "avg_quality": round(self.avg_quality, 3),
            "cache_size_mb": round(self.cache_size_mb, 2),
            "max_cache_mb": self.config.max_cache_mb,
            "dream_files": len(list(self.dream_dir.glob("dreams_*.jsonl"))),
        }

    def export_for_sharing(self, max_samples: int = 500) -> list[dict]:
        """Export top-quality samples for sharing with the swarm."""
        self._load_if_needed()
        sorted_samples = sorted(self._samples, key=lambda s: s.score, reverse=True)
        return [s.to_dict() for s in sorted_samples[:max_samples]]

    def import_shared(self, samples: list[dict], source_peer: str) -> int:
        """Import dream samples shared by another peer."""
        dream_samples = []
        for data in samples:
            sample = DreamSample.from_dict(data)
            sample.source_peer = source_peer
            dream_samples.append(sample)
        return self.add(dream_samples)


class DreamGenerator:
    """Generates synthetic training data by running inference on the current model.

    Each Mac uses a different seed (derived from peer_id) for natural variation,
    ensuring dream diversity across the swarm.
    """

    def __init__(self, config: DreamConfig, peer_id: str = "local"):
        self.config = config
        self.peer_id = peer_id
        # Deterministic but unique seed per peer
        seed_hash = hashlib.md5(peer_id.encode()).hexdigest()
        self.seed = int(seed_hash[:8], 16)

    def generate_dreams(
        self,
        model,
        tokenizer=None,
        num_samples: Optional[int] = None,
        prompts: Optional[list[str]] = None,
    ) -> list[DreamSample]:
        """Generate synthetic samples using the current model.

        Args:
            model: The MLX model to generate from.
            tokenizer: Tokenizer for encoding/decoding. If None, model must
                       have a .generate() method that returns strings.
            num_samples: Number of samples to generate (default: config.samples_per_session).
            prompts: Optional seed prompts. If None, generates unconditionally.

        Returns:
            List of DreamSample with text and metadata (not yet scored).
        """
        try:
            import mlx.core as mx
        except ImportError:
            logger.warning("MLX not available — using mock dream generation")
            return self._generate_mock_dreams(num_samples or self.config.samples_per_session)

        n = num_samples or self.config.samples_per_session
        samples = []

        mx.random.seed(self.seed + int(time.time()) % 10000)

        for i in range(n):
            try:
                # Pick a prompt (or empty for unconditional)
                prompt = ""
                if prompts:
                    prompt = prompts[i % len(prompts)]

                # Generate using the model
                if tokenizer is not None:
                    text, loss = self._generate_with_tokenizer(
                        model, tokenizer, prompt
                    )
                else:
                    text = self._generate_raw(model, prompt)
                    loss = None

                if text and len(text.strip()) > 0:
                    sample = DreamSample(
                        text=text.strip(),
                        source_peer=self.peer_id,
                        model_step=0,
                        token_count=len(text.split()),
                    )
                    samples.append(sample)

            except Exception as e:
                logger.debug(f"Dream generation failed for sample {i}: {e}")
                continue

            if (i + 1) % 100 == 0:
                logger.debug(f"Generated {i + 1}/{n} dream samples")

        logger.info(f"Generated {len(samples)} dream samples")
        return samples

    def _generate_with_tokenizer(self, model, tokenizer, prompt: str) -> tuple[str, Optional[float]]:
        """Generate text using model + tokenizer with temperature sampling."""
        import mlx.core as mx
        import mlx.nn as nn

        tokens = tokenizer.encode(prompt) if prompt else [tokenizer.bos_token_id or 1]
        tokens = mx.array([tokens])

        generated = list(tokens[0].tolist())
        total_loss = 0.0
        num_tokens = 0

        for _ in range(self.config.max_length - len(generated)):
            logits = model(mx.array([generated]))
            logits = logits[:, -1, :]  # last token logits

            # Temperature scaling
            logits = logits / self.config.temperature

            # Top-p (nucleus) sampling
            probs = mx.softmax(logits, axis=-1)
            sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
            cumsum = mx.cumsum(sorted_probs, axis=-1)
            mask = cumsum - sorted_probs <= self.config.top_p
            filtered_probs = sorted_probs * mask
            filtered_probs = filtered_probs / mx.sum(filtered_probs, axis=-1, keepdims=True)

            # Sample
            next_token = mx.random.categorical(mx.log(filtered_probs + 1e-10))
            next_token = sorted_indices[0, next_token.item()].item()

            # Track loss for scoring
            token_prob = probs[0, next_token].item()
            if token_prob > 0:
                total_loss -= math.log(token_prob)
                num_tokens += 1

            generated.append(next_token)

            # Stop on EOS
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and next_token == eos:
                break

        avg_loss = total_loss / max(num_tokens, 1)
        text = tokenizer.decode(generated)
        return text, avg_loss

    def _generate_raw(self, model, prompt: str) -> str:
        """Fallback generation for models with a .generate() method."""
        if hasattr(model, "generate"):
            return model.generate(
                prompt,
                max_tokens=self.config.max_length,
                temperature=self.config.temperature,
            )
        return ""

    def _generate_mock_dreams(self, n: int) -> list[DreamSample]:
        """Generate placeholder dreams when MLX is not available (for testing)."""
        random.seed(self.seed + int(time.time()) % 10000)
        templates = [
            "The distributed training process converges when multiple workers share gradient information across the network, enabling collaborative model optimization without centralized compute.",
            "Apple Silicon's unified memory architecture eliminates the traditional bottleneck of host-to-device memory transfers, making it uniquely suited for machine learning workloads on consumer hardware.",
            "In federated learning scenarios, each participant trains on their local data and shares only model updates, preserving privacy while contributing to a shared global model.",
            "The loss landscape of neural networks contains multiple basins of attraction, and distributed training with diverse initializations can explore more of this landscape simultaneously.",
            "Gradient compression techniques reduce communication overhead by quantizing or sparsifying gradient updates before transmission, with minimal impact on convergence quality.",
            "The DiLoCo algorithm achieves communication efficiency by allowing workers to train independently for hundreds of steps before synchronizing, reducing bandwidth requirements by orders of magnitude.",
            "Model checkpointing enables fault-tolerant training by periodically saving the complete training state, allowing seamless recovery from hardware failures or network interruptions.",
            "Temperature scaling in language model sampling controls the entropy of the output distribution, with higher temperatures producing more diverse but potentially less coherent text.",
        ]
        samples = []
        for i in range(n):
            text = random.choice(templates)
            # Add slight variation
            words = text.split()
            if len(words) > 10:
                # Randomly swap a few words
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, len(words) - 1)
                    words[idx] = random.choice(["efficient", "optimal", "novel", "robust", "scalable", "adaptive", "distributed", "parallel", "concurrent", "asynchronous"])
                text = " ".join(words)

            samples.append(DreamSample(
                text=text,
                source_peer=self.peer_id,
                model_step=0,
                token_count=len(text.split()),
            ))
        return samples


class DreamSession:
    """Orchestrates a single dream session.

    Called during idle time — between DiLoCo sync rounds or during
    Sleep Swarm gaps. Generates samples, scores them, and caches
    the high-quality ones.
    """

    def __init__(
        self,
        config: DreamConfig,
        peer_id: str = "local",
        model=None,
        tokenizer=None,
    ):
        self.config = config
        self.peer_id = peer_id
        self.model = model
        self.tokenizer = tokenizer
        self.generator = DreamGenerator(config, peer_id)
        self.scorer = DreamScorer(config)
        self.cache = DreamCache(config, peer_id)

    def run(self, num_samples: Optional[int] = None, model_step: int = 0) -> dict:
        """Run a dream session. Returns stats dict.

        Args:
            num_samples: Override samples count (e.g., 100 for mini-dreams between sync rounds).
            model_step: Current training step (stored in dream metadata).
        """
        n = num_samples or self.config.samples_per_session
        logger.info(f"Starting dream session: generating {n} samples...")
        start = time.time()

        # Generate
        samples = self.generator.generate_dreams(
            self.model, self.tokenizer, num_samples=n
        )

        # Score
        scored = 0
        for sample in samples:
            sample.model_step = model_step
            sample.score = self.scorer.score(sample.text)
            scored += 1

        # Cache
        kept = self.cache.add(samples)

        elapsed = time.time() - start
        stats = {
            "generated": len(samples),
            "scored": scored,
            "kept": kept,
            "rejected": len(samples) - kept,
            "avg_quality": round(
                sum(s.score for s in samples) / max(len(samples), 1), 3
            ),
            "elapsed_seconds": round(elapsed, 1),
            "samples_per_second": round(len(samples) / max(elapsed, 0.1), 1),
            "cache_total": self.cache.total_samples,
            "cache_size_mb": round(self.cache.cache_size_mb, 2),
        }

        logger.info(
            f"Dream session complete: {kept}/{len(samples)} kept, "
            f"avg quality {stats['avg_quality']}, "
            f"{stats['elapsed_seconds']}s"
        )
        return stats


def mix_dream_batch(
    real_texts: list[str],
    dream_cache: DreamCache,
    mix_ratio: float = 0.15,
) -> list[str]:
    """Mix dream data into a real training batch.

    Args:
        real_texts: The real training batch.
        dream_cache: Cache of dream samples to draw from.
        mix_ratio: Fraction of the batch that should be dream data.

    Returns:
        Mixed batch with (1-mix_ratio) real + mix_ratio dream samples, shuffled.
    """
    batch_size = len(real_texts)
    dream_count = max(1, int(batch_size * mix_ratio))
    real_count = batch_size - dream_count

    # Get dream samples
    dreams = dream_cache.get_training_batch(dream_count)
    if not dreams:
        return real_texts  # no dreams available, use all real data

    # Mix and shuffle
    mixed = real_texts[:real_count] + dreams
    random.shuffle(mixed)

    logger.debug(
        f"Batch mixed: {real_count} real ({100 - int(mix_ratio * 100)}%) + "
        f"{len(dreams)} dream ({int(mix_ratio * 100)}%)"
    )
    return mixed
