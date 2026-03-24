"""
Factory for loading and instantiating community detection algorithms.

Loads algorithms from the registry (populated by ``@register`` decorators)
and uses YAML only for selecting which algorithms to run and optionally
overriding default parameters.

Algorithm modules must be imported before ``load_algorithms`` is called so
that their ``@register`` decorators execute. The ``_ensure_registrations``
helper handles this automatically.
"""
from __future__ import annotations

import importlib
from typing import Dict

import yaml

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.registry import ALGORITHM_REGISTRY
from src.algorithms.wrappers import DynamicMethodWrapper, StaticMethodWrapper


# Modules that contain @register calls.  Importing them populates the registry.
_REGISTRATION_MODULES = [
    # Custom algorithms
    "src.models.static.overlap.big_clam",
    "src.models.static.overlap.copra",
    "src.models.static.overlap.cosine_overlap",
    "src.models.static.overlap.ndocd",
    "src.models.static.overlap.vast_pmo",
    "src.models.dynamic.overlap.tiles",
    "src.models.dynamic.crisp.df_louvain",
    "src.models.static.crisp.static_louvain",
    # CDlib adapters
    "src.algorithms.cdlib_adapters",
]


def _ensure_registrations() -> None:
    """Import all algorithm modules so their ``@register`` decorators fire."""
    for module_path in _REGISTRATION_MODULES:
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            print(f"Warning: Could not import '{module_path}': {exc}. Skipping.")


def load_algorithms(config_path: str = "config/algorithms.yaml") -> Dict[str, dict]:
    """
    Load and instantiate all target algorithms listed in the YAML config.

    The YAML config selects which registered algorithms to run and optionally
    overrides their default parameters.  Algorithm identity and metadata live
    in the registry (populated via ``@register``).

    YAML shape::

        target_snapshot_algorithms:
          - big_clam
          - angel

        target_temporal_algorithms:
          - tiles

        algorithm_params:          # optional overrides
          big_clam:
            num_communities: 10

    Raises:
        ValueError: If a YAML-referenced algorithm name is not in the registry.
    """
    # Step 1: ensure all @register decorators have executed
    _ensure_registrations()

    # Step 2: read YAML run config
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    snapshot_targets = config.get("target_snapshot_algorithms") or []
    temporal_targets = config.get("target_temporal_algorithms") or []
    overrides = config.get("algorithm_params") or {}

    algorithms: Dict[str, dict] = {}

    all_targets = [
        *[(name, "static") for name in snapshot_targets],
        *[(name, "dynamic") for name in temporal_targets],
    ]

    for name, expected_type in all_targets:
        # Validate against registry
        if name not in ALGORITHM_REGISTRY:
            available = ", ".join(sorted(ALGORITHM_REGISTRY.keys()))
            raise ValueError(
                f"Unknown algorithm '{name}' in YAML config. "
                f"Not found in registry. Available: {available}"
            )

        spec = ALGORITHM_REGISTRY[name]

        # Warn if YAML puts it in the wrong target list
        if spec.algo_type != expected_type:
            print(
                f"Warning: Algorithm '{name}' is registered as '{spec.algo_type}' "
                f"but listed under target_{expected_type}_algorithms in YAML. "
                f"Using registered type '{spec.algo_type}'."
            )

        # Merge default params with YAML overrides
        params = {**spec.default_params, **overrides.get(name, {})}
        target = spec.target

        # Instantiate wrapper
        if isinstance(target, type):
            instance = target(**params)
            if isinstance(instance, CommunityDetectionAlgorithm):
                wrapper = instance
            else:
                wrapper = (
                    DynamicMethodWrapper(instance, {})
                    if spec.algo_type == "dynamic"
                    else StaticMethodWrapper(instance, {})
                )
        else:
            wrapper = (
                DynamicMethodWrapper(target, params)
                if spec.algo_type == "dynamic"
                else StaticMethodWrapper(target, params)
            )

        algorithms[name] = {
            "wrapper": wrapper,
            "type": spec.algo_type,
            "clustering_type": spec.clustering_type,
            "params": params,
            "description": spec.description,
        }

    return algorithms
