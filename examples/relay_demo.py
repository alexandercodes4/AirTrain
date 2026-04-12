#!/usr/bin/env python3
"""Example: Training relay — train, pause, export, and let someone else continue.

This demonstrates the async relay workflow where training can be
handed off between people who aren't online at the same time.

Usage:
    # Person 1: Train for a while
    python relay_demo.py train --steps 5000

    # Person 1: Export checkpoint for relay
    python relay_demo.py export --checkpoint ./checkpoints/step-5000

    # Person 2: Import and continue
    python relay_demo.py import --relay ./relay_checkpoint
    python relay_demo.py train --steps 5000 --resume ./relay_checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(args):
    """Train locally for N steps."""
    from airtrain.config import TrainingConfig
    from airtrain.models.registry import get_model

    print(f"Training {args.model} for {args.steps} steps...")

    config = TrainingConfig(
        model_name=args.model,
        max_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
    )

    model = get_model(config.model_name)
    param_count = sum(p.size for p in model.parameters().values())
    print(f"Model parameters: {param_count:,}")

    # In a real scenario, this would run the full training loop
    # For demo purposes, we just show the workflow
    print(f"Training complete. Checkpoint saved to {args.checkpoint_dir}/")


def export_relay(args):
    """Export a checkpoint for relay handoff."""
    from airtrain.engine.checkpoint import export_relay

    print(f"Exporting relay checkpoint from {args.checkpoint}...")
    export_relay(args.checkpoint, args.output, args.description)
    print(f"Relay checkpoint exported to {args.output}/")
    print("Share this directory with another trainer to continue!")


def import_relay(args):
    """Import a relay checkpoint."""
    from airtrain.engine.checkpoint import import_relay

    meta = import_relay(args.relay)
    print(f"Imported relay checkpoint:")
    print(f"  Model: {meta.model_name}")
    print(f"  Step: {meta.global_step}")
    print(f"  Loss: {meta.loss:.4f}")
    print(f"  Contributors: {meta.contributors}")
    print(f"  Compute hours: {meta.total_compute_hours:.1f}")
    print(f"\nResume with: python relay_demo.py train --resume {args.relay}")


def main():
    parser = argparse.ArgumentParser(description="AirTrain Relay Demo")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train locally")
    train_parser.add_argument("--model", default="gpt2-tiny")
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--checkpoint-dir", default="./checkpoints")
    train_parser.add_argument("--resume", type=str, default=None)

    export_parser = subparsers.add_parser("export", help="Export relay checkpoint")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", default="./relay_checkpoint")
    export_parser.add_argument("--description", default="")

    import_parser = subparsers.add_parser("import", help="Import relay checkpoint")
    import_parser.add_argument("--relay", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "export":
        export_relay(args)
    elif args.command == "import":
        import_relay(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
