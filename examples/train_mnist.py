#!/usr/bin/env python3
"""Example: Simple distributed MNIST training for quick testing.

A minimal example using a small CNN to verify AirTrain works.
MNIST is small enough to train in seconds, making it ideal for testing.

Usage:
    # Mac 1:
    python train_mnist.py --role coordinator

    # Mac 2:
    python train_mnist.py --role worker
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


class SimpleCNN(nn.Module):
    """Tiny CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Distributed MNIST with AirTrain")
    parser.add_argument("--role", choices=["coordinator", "worker"], default="coordinator")
    parser.add_argument("--inner-steps", type=int, default=100)
    parser.add_argument("--port", type=int, default=7471)
    args = parser.parse_args()

    print(f"\n{'=' * 40}")
    print("  AirTrain — MNIST Demo")
    print(f"{'=' * 40}")
    print(f"  Role: {args.role}")
    print(f"  Inner steps: {args.inner_steps}")
    print(f"{'=' * 40}\n")

    model = SimpleCNN()
    param_count = sum(p.size for p in model.parameters().values())
    print(f"Model parameters: {param_count:,}")

    # For a real implementation, this would use the AirTrain engine
    # This is a demonstration of the model architecture
    print("MNIST demo ready. Use 'airtrain start/join' for distributed training.")


if __name__ == "__main__":
    main()
