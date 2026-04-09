"""Compatibility layer for running AirTrain on non-macOS platforms.

MLX is macOS-only. This module provides stub implementations so that
the codebase can be imported, tested, and developed on any platform.
Actual training requires macOS with Apple Silicon.
"""

from __future__ import annotations

import sys

MLX_AVAILABLE = False

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    MLX_AVAILABLE = True
except ImportError:
    pass


def require_mlx():
    """Raise an error if MLX is not available."""
    if not MLX_AVAILABLE:
        raise RuntimeError(
            "MLX is required for training but is not installed. "
            "MLX only runs on macOS with Apple Silicon. "
            "Install with: pip install airtrain[mlx]"
        )


def get_platform_info() -> dict:
    """Get information about the current platform's ML capabilities."""
    import platform as plat

    info = {
        "system": plat.system(),
        "machine": plat.machine(),
        "processor": plat.processor(),
        "mlx_available": MLX_AVAILABLE,
        "apple_silicon": plat.machine() == "arm64" and plat.system() == "Darwin",
    }

    if MLX_AVAILABLE:
        info["mlx_version"] = mx.__version__
        info["default_device"] = str(mx.default_device())

    return info
