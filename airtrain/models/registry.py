"""Model registry for AirTrain."""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str, factory: Callable[..., Any]) -> None:
    """Register a model factory function."""
    _REGISTRY[name] = factory


def get_model(name: str, **kwargs) -> Any:
    """Create a model by name."""
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)

    # Fall back to built-in transformer models
    from airtrain.models.transformer import PRESETS, create_model

    if name in PRESETS:
        return create_model(name)

    available = list(_REGISTRY.keys()) + list(PRESETS.keys())
    raise ValueError(f"Unknown model: {name}. Available: {available}")


def list_models() -> list[str]:
    """List all available model names."""
    from airtrain.models.transformer import PRESETS

    return sorted(set(list(_REGISTRY.keys()) + list(PRESETS.keys())))
