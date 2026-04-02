"""Gradient compression for efficient network transfer."""

from __future__ import annotations

import gzip
import io
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compress_gradients(arrays: dict[str, np.ndarray], to_fp16: bool = True) -> bytes:
    """Compress gradient arrays for network transfer.

    Optionally casts to FP16 for 2x size reduction, then gzips.

    Args:
        arrays: Dict of parameter name -> gradient array (numpy).
        to_fp16: Whether to cast float32 to float16.

    Returns:
        Compressed bytes.
    """
    buf = io.BytesIO()

    processed = {}
    for name, arr in arrays.items():
        if to_fp16 and arr.dtype == np.float32:
            processed[name] = arr.astype(np.float16)
        else:
            processed[name] = arr

    np.savez(buf, **processed)
    raw = buf.getvalue()

    compressed = gzip.compress(raw, compresslevel=1)

    ratio = len(raw) / max(len(compressed), 1)
    logger.debug(
        "Compressed gradients: %d params, %.1f MB -> %.1f MB (%.1fx)",
        len(arrays),
        len(raw) / 1e6,
        len(compressed) / 1e6,
        ratio,
    )

    return compressed


def decompress_gradients(data: bytes, to_fp32: bool = True) -> dict[str, np.ndarray]:
    """Decompress gradient arrays from network transfer.

    Args:
        data: Compressed bytes from compress_gradients.
        to_fp32: Whether to cast FP16 back to FP32.

    Returns:
        Dict of parameter name -> gradient array.
    """
    raw = gzip.decompress(data)
    buf = io.BytesIO(raw)

    npz = np.load(buf)
    arrays = {}
    for name in npz.files:
        arr = npz[name]
        if to_fp32 and arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        arrays[name] = arr

    return arrays
