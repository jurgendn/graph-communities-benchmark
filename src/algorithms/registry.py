"""
Algorithm registry for community detection methods.

The registry is the single source of truth for algorithm identity and metadata.
YAML config only selects which registered algorithms to run and optionally
overrides default parameters.

Usage::

    from src.algorithms.registry import register

    @register(
        name="my_algorithm",
        algo_type="static",
        clustering_type="overlapping",
        default_params={"threshold": 0.5},
        description="My overlapping community detection method",
    )
    class MyAlgorithm(CommunityDetectionAlgorithm):
        ...

    # Or for plain functions:
    @register(
        name="my_func",
        algo_type="static",
        clustering_type="overlapping",
        default_params={"alpha": 0.8},
        description="Simple static function",
    )
    def my_func(graph, alpha=0.8):
        ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AlgorithmSpec:
    """Metadata for a registered algorithm."""

    name: str
    target: Callable
    algo_type: str  # "static" | "dynamic"
    clustering_type: str  # "crisp" | "overlapping"
    default_params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


# Global registry: name -> AlgorithmSpec
ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {}


def register(
    *,
    name: str,
    algo_type: str,
    clustering_type: str,
    default_params: dict[str, Any] | None = None,
    description: str = "",
) -> Callable:
    """
    Decorator that registers an algorithm (class or function) in the global registry.

    Args:
        name: Unique algorithm name (must match YAML references).
        algo_type: ``"static"`` or ``"dynamic"``.
        clustering_type: ``"crisp"`` or ``"overlapping"``.
        default_params: Default keyword arguments for the algorithm.
        description: Short human-readable description.

    Raises:
        ValueError: If ``name`` is already registered or ``algo_type`` is invalid.
    """
    if algo_type not in ("static", "dynamic"):
        raise ValueError(f"algo_type must be 'static' or 'dynamic', got '{algo_type}'")
    if clustering_type not in ("crisp", "overlapping"):
        raise ValueError(
            f"clustering_type must be 'crisp' or 'overlapping', got '{clustering_type}'"
        )

    def decorator(target: Callable) -> Callable:
        if name in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Algorithm '{name}' is already registered "
                f"(target: {ALGORITHM_REGISTRY[name].target})"
            )

        ALGORITHM_REGISTRY[name] = AlgorithmSpec(
            name=name,
            target=target,
            algo_type=algo_type,
            clustering_type=clustering_type,
            default_params=default_params or {},
            description=description,
        )
        return target

    return decorator
