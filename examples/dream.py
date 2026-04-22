#!/usr/bin/env python3
"""
AirTrain Dream Training Demo
=============================

Your Mac "dreams" about a model during idle time — generating synthetic
training data, scoring each sample for quality, and caching the best ones.
When training resumes, dream data accelerates convergence.

This demo connects to a local Ollama instance and runs a real dream session.

Requirements:
    1. Install Ollama: https://ollama.com
    2. Pull a model:   ollama pull llama3.2
    3. Run this demo:  python demo_dream.py

Optional flags:
    --model phi3          Use a different Ollama model
    --samples 50          Number of dream samples to generate
    --temperature 0.9     Sampling temperature (higher = more creative dreams)
    --dream-dir ./dreams  Where to cache dream data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Missing dependency: pip install httpx")
    sys.exit(1)


# ── Dream Quality Scoring ──────────────────────────────────────────────

PROMPTS = [
    "Explain how distributed computing works in simple terms.",
    "Write a short story about a group of computers working together.",
    "Describe the process of training a neural network step by step.",
    "What are the benefits of collaborative machine learning?",
    "Write a technical explanation of gradient descent optimization.",
    "Explain why Apple Silicon is efficient for machine learning.",
    "Describe a future where everyone's laptop contributes to AI training.",
    "Write about the mathematics behind averaging model weights.",
    "Explain the concept of federated learning to a beginner.",
    "Describe how peer-to-peer networks discover each other.",
    "Write about the energy efficiency of training AI on consumer hardware.",
    "Explain the trade-offs between centralized and distributed training.",
    "Describe a relay race as a metaphor for checkpoint-based training.",
    "Write about the role of synchronization in distributed systems.",
    "Explain how compression reduces communication overhead in training.",
]


@dataclass
class DreamSample:
    text: str
    prompt: str
    quality_score: float
    scores: dict
    token_count: int
    timestamp: str
    model: str
    dream_id: str


def score_repetition(text: str) -> float:
    """Score 0-1 where 1 = no repetition, 0 = highly repetitive."""
    words = text.lower().split()
    if len(words) < 10:
        return 0.0

    # Check bigram repetition
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bigram_counts = Counter(bigrams)
    if not bigrams:
        return 0.0
    max_repeat = max(bigram_counts.values())
    repeat_ratio = max_repeat / len(bigrams)

    # Check trigram repetition
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
    trigram_counts = Counter(trigrams)
    tri_repeat = max(trigram_counts.values()) / max(len(trigrams), 1) if trigrams else 0

    # Penalize high repetition
    score = 1.0 - min(1.0, repeat_ratio * 3 + tri_repeat * 5)
    return max(0.0, score)


def score_diversity(text: str) -> float:
    """Score 0-1 where 1 = high vocabulary diversity."""
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    unique = len(set(words))
    ratio = unique / len(words)
    return min(1.0, ratio * 1.5)  # boost slightly


def score_coherence(text: str) -> float:
    """Score 0-1 based on structural coherence signals."""
    score = 0.5  # baseline

    # Has sentences (periods, question marks, etc.)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        score += 0.15

    # Sentences have reasonable length
    avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if 8 <= avg_sent_len <= 30:
        score += 0.15

    # Has paragraph structure or list structure
    if '\n' in text or any(text.strip().startswith(c) for c in ['1.', '-', '*', '•']):
        score += 0.1

    # Doesn't end mid-sentence (has terminal punctuation)
    if text.strip() and text.strip()[-1] in '.!?':
        score += 0.1

    return min(1.0, score)


def score_length(text: str, min_len: int = 50, max_len: int = 500) -> float:
    """Score 0-1 based on length being in the sweet spot."""
    word_count = len(text.split())
    if word_count < min_len // 5:
        return 0.0
    if word_count > max_len:
        return max(0.3, 1.0 - (word_count - max_len) / max_len)
    return 1.0


def compute_quality(text: str) -> tuple[float, dict]:
    """Compute overall quality score and component scores."""
    scores = {
        "repetition": score_repetition(text),
        "diversity": score_diversity(text),
        "coherence": score_coherence(text),
        "length": score_length(text),
    }

    # Weighted average — coherence and repetition matter most
    weights = {"repetition": 0.3, "diversity": 0.2, "coherence": 0.35, "length": 0.15}
    quality = sum(scores[k] * weights[k] for k in scores)

    return round(quality, 4), {k: round(v, 3) for k, v in scores.items()}


# ── Ollama Client ──────────────────────────────────────────────────────

def check_ollama(base_url: str) -> bool:
    """Check if Ollama is running."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def list_models(base_url: str) -> list[str]:
    """List available Ollama models."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def generate(base_url: str, model: str, prompt: str, temperature: float) -> tuple[str, float]:
    """Generate text from Ollama and return (text, duration_seconds)."""
    start = time.time()
    r = httpx.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.95,
                "num_predict": 512,
            },
        },
        timeout=120.0,
    )
    elapsed = time.time() - start
    data = r.json()
    return data.get("response", ""), elapsed


# ── Dream Cache ────────────────────────────────────────────────────────

class DreamCache:
    def __init__(self, dream_dir: str):
        self.dream_dir = Path(dream_dir)
        self.dream_dir.mkdir(parents=True, exist_ok=True)

    def save(self, sample: DreamSample):
        """Append a dream sample to the cache."""
        filename = self.dream_dir / f"dreams_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(filename, "a") as f:
            f.write(json.dumps({
                "dream_id": sample.dream_id,
                "text": sample.text,
                "prompt": sample.prompt,
                "quality_score": sample.quality_score,
                "scores": sample.scores,
                "token_count": sample.token_count,
                "timestamp": sample.timestamp,
                "model": sample.model,
            }) + "\n")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = 0
        total_quality = 0.0
        size_bytes = 0

        for f in self.dream_dir.glob("*.jsonl"):
            size_bytes += f.stat().st_size
            for line in open(f):
                try:
                    d = json.loads(line)
                    total += 1
                    total_quality += d.get("quality_score", 0)
                except json.JSONDecodeError:
                    pass

        return {
            "total_samples": total,
            "avg_quality": round(total_quality / max(total, 1), 3),
            "cache_size_mb": round(size_bytes / 1024 / 1024, 2),
            "dream_files": len(list(self.dream_dir.glob("*.jsonl"))),
        }


# ── Display ────────────────────────────────────────────────────────────

GRAY = "\033[90m"
WHITE = "\033[97m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def quality_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    if score >= 0.7:
        color = GREEN
    elif score >= 0.4:
        color = YELLOW
    else:
        color = RED
    bar = f"{color}{'█' * filled}{GRAY}{'░' * (width - filled)}{RESET}"
    return bar


def quality_label(score: float) -> str:
    if score >= 0.8:
        return f"{GREEN}excellent{RESET}"
    elif score >= 0.7:
        return f"{GREEN}good{RESET}"
    elif score >= 0.5:
        return f"{YELLOW}fair{RESET}"
    elif score >= 0.3:
        return f"{YELLOW}poor{RESET}"
    else:
        return f"{RED}rejected{RESET}"


def print_header():
    print(f"""
{BOLD}{WHITE}{'=' * 60}
  AirTrain — Dream Training Demo
{'=' * 60}{RESET}

  {CYAN}Your Mac is about to "dream" about a model.{RESET}
  It generates synthetic text, scores each sample for quality,
  and caches the best ones for future training.

  Inspired by how the brain consolidates learning during sleep.
""")


def print_sample_result(i: int, total: int, sample: DreamSample, elapsed: float, kept: bool):
    status = f"{GREEN}KEPT{RESET}" if kept else f"{RED}SKIP{RESET}"
    print(f"\n  {BOLD}Dream {i}/{total}{RESET}  [{status}]  {DIM}{elapsed:.1f}s{RESET}")
    print(f"  {DIM}Prompt: {sample.prompt[:60]}...{RESET}")
    print(f"  Quality: {quality_bar(sample.quality_score)} {sample.quality_score:.3f} ({quality_label(sample.quality_score)})")

    # Show component scores
    components = []
    for k, v in sample.scores.items():
        icon = "●" if v >= 0.6 else "○"
        components.append(f"{icon} {k}={v:.2f}")
    print(f"  {DIM}{' | '.join(components)}{RESET}")

    # Show preview of generated text
    preview = sample.text[:120].replace('\n', ' ').strip()
    if len(sample.text) > 120:
        preview += "..."
    print(f"  {DIM}Preview: \"{preview}\"{RESET}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AirTrain Dream Training Demo")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name (default: llama3.2)")
    parser.add_argument("--samples", type=int, default=20, help="Number of dream samples (default: 20)")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (default: 0.9)")
    parser.add_argument("--quality-threshold", type=float, default=0.55, help="Min quality to keep (default: 0.55)")
    parser.add_argument("--dream-dir", default="./dreams", help="Dream cache directory")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    args = parser.parse_args()

    print_header()

    # Check Ollama
    print(f"  {BOLD}Connecting to Ollama...{RESET}", end="", flush=True)
    if not check_ollama(args.ollama_url):
        print(f" {RED}FAILED{RESET}")
        print(f"\n  Ollama is not running. Start it with:")
        print(f"    ollama serve")
        print(f"\n  Install Ollama: https://ollama.com")
        sys.exit(1)
    print(f" {GREEN}connected{RESET}")

    # Check model
    models = list_models(args.ollama_url)
    model_base = args.model.split(":")[0]
    matching = [m for m in models if m.startswith(model_base)]
    if not matching:
        print(f"\n  {RED}Model '{args.model}' not found.{RESET}")
        print(f"  Available models: {', '.join(models) if models else '(none)'}")
        print(f"\n  Pull it with: ollama pull {args.model}")
        sys.exit(1)

    actual_model = matching[0]
    print(f"  {BOLD}Model:{RESET} {actual_model}")
    print(f"  {BOLD}Samples:{RESET} {args.samples}")
    print(f"  {BOLD}Temperature:{RESET} {args.temperature}")
    print(f"  {BOLD}Quality threshold:{RESET} {args.quality_threshold}")
    print(f"  {BOLD}Dream cache:{RESET} {args.dream_dir}")

    # Initialize cache
    cache = DreamCache(args.dream_dir)

    print(f"\n  {CYAN}Beginning dream session...{RESET}")
    print(f"  {DIM}{'─' * 56}{RESET}")

    kept_count = 0
    rejected_count = 0
    total_quality = 0.0
    total_time = 0.0
    best_dream = None
    worst_kept = None

    for i in range(1, args.samples + 1):
        prompt = PROMPTS[(i - 1) % len(PROMPTS)]

        try:
            text, elapsed = generate(args.ollama_url, actual_model, prompt, args.temperature)
            total_time += elapsed
        except Exception as e:
            print(f"\n  {RED}Dream {i} failed: {e}{RESET}")
            continue

        if not text.strip():
            print(f"\n  {RED}Dream {i}: empty response, skipping{RESET}")
            rejected_count += 1
            continue

        quality, scores = compute_quality(text)
        total_quality += quality

        dream_id = hashlib.md5(f"{text[:100]}{time.time()}".encode()).hexdigest()[:12]
        sample = DreamSample(
            text=text,
            prompt=prompt,
            quality_score=quality,
            scores=scores,
            token_count=len(text.split()),
            timestamp=datetime.now().isoformat(),
            model=actual_model,
            dream_id=dream_id,
        )

        kept = quality >= args.quality_threshold
        if kept:
            cache.save(sample)
            kept_count += 1
            if best_dream is None or quality > best_dream.quality_score:
                best_dream = sample
            if worst_kept is None or quality < worst_kept.quality_score:
                worst_kept = sample
        else:
            rejected_count += 1

        print_sample_result(i, args.samples, sample, elapsed, kept)

    # ── Summary ────────────────────────────────────────────────────
    total_generated = kept_count + rejected_count
    avg_quality = total_quality / max(total_generated, 1)
    keep_rate = kept_count / max(total_generated, 1) * 100
    cache_stats = cache.get_stats()

    print(f"\n  {DIM}{'─' * 56}{RESET}")
    print(f"""
{BOLD}{WHITE}  Dream Session Complete{RESET}
  {'─' * 40}

  {BOLD}Generated:{RESET}    {total_generated} samples in {total_time:.1f}s
  {BOLD}Kept:{RESET}         {GREEN}{kept_count}{RESET} ({keep_rate:.0f}%)
  {BOLD}Rejected:{RESET}     {RED}{rejected_count}{RESET}
  {BOLD}Avg quality:{RESET}  {quality_bar(avg_quality)} {avg_quality:.3f}
  {BOLD}Avg speed:{RESET}    {total_time / max(total_generated, 1):.1f}s per dream
""")

    if best_dream:
        print(f"  {BOLD}{GREEN}Best dream:{RESET} quality {best_dream.quality_score:.3f}")
        preview = best_dream.text[:100].replace('\n', ' ').strip()
        print(f"  {DIM}\"{preview}...\"{RESET}")

    print(f"""
  {BOLD}Dream Cache:{RESET}
    Total samples:  {cache_stats['total_samples']}
    Avg quality:    {cache_stats['avg_quality']}
    Cache size:     {cache_stats['cache_size_mb']} MB
    Cache files:    {cache_stats['dream_files']}
    Location:       {args.dream_dir}/

  {CYAN}These dreams will be mixed into training batches at a 15%
  ratio to accelerate convergence when training resumes.{RESET}

  {DIM}Run again to generate more dreams:
    python demo_dream.py --samples 50 --temperature 1.0{RESET}
""")


if __name__ == "__main__":
    main()
