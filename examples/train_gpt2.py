#!/usr/bin/env python3
"""Example: Train GPT-2 across multiple Macs with AirTrain.

Usage:
    # Mac 1 (coordinator):
    python train_gpt2.py --role coordinator --dataset ./data/wikitext.txt

    # Mac 2+ (workers):
    python train_gpt2.py --role worker

    # Or use the CLI:
    airtrain start --model gpt2-small --dataset ./data/wikitext.txt
    airtrain join auto
"""

from __future__ import annotations

import argparse
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 with AirTrain")
    parser.add_argument(
        "--role",
        choices=["coordinator", "worker"],
        default="coordinator",
        help="Role in the training swarm",
    )
    parser.add_argument("--dataset", type=str, default="./data/wikitext.txt")
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--port", type=int, default=7471)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    from airtrain.config import DiLoCoConfig, NetworkConfig, TrainingConfig

    training_config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        enable_dashboard=args.dashboard,
        diloco=DiLoCoConfig(inner_steps=args.inner_steps),
    )
    network_config = NetworkConfig(listen_port=args.port)

    if args.role == "coordinator":
        from airtrain.engine.coordinator import run_coordinator

        print(f"\n{'=' * 50}")
        print("  AirTrain — GPT-2 Distributed Training")
        print(f"{'=' * 50}")
        print(f"  Model:       {args.model}")
        print(f"  Dataset:     {args.dataset}")
        print(f"  Inner steps: {args.inner_steps}")
        print(f"  Port:        {args.port}")
        print(f"{'=' * 50}\n")

        asyncio.run(
            run_coordinator(training_config, network_config, resume_path=args.resume)
        )
    else:
        from airtrain.engine.worker import run_worker

        print("Searching for AirTrain coordinator on local network...")
        asyncio.run(run_worker("auto", network_config))


if __name__ == "__main__":
    main()
