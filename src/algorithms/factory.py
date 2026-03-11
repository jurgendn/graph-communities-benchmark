"""Factory for loading and instantiating community detection algorithms."""
import importlib
from typing import Dict, Tuple

import yaml

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.wrappers import DynamicMethodWrapper, StaticMethodWrapper


def _read_algorithm_sections(config: dict) -> Tuple[list[str], dict, list[str], dict]:
    if "snapshot_algorithms" in config or "temporal_algorithms" in config:
        return (
            config.get("target_snapshot_algorithms", []),
            config.get("snapshot_algorithms", {}),
            config.get("target_temporal_algorithms", []),
            config.get("temporal_algorithms", {}),
        )

    if "static_algorithms" in config or "dynamic_algorithms" in config:
        return (
            config.get("target_static_algorithms", []),
            config.get("static_algorithms", {}),
            config.get("target_dynamic_algorithms", []),
            config.get("dynamic_algorithms", {}),
        )

    target = config.get("target_algorithms", [])
    all_algorithms = config.get("algorithms", {})

    static_target = [
        name for name in target if all_algorithms.get(name, {}).get("type", "static") == "static"
    ]
    dynamic_target = [
        name for name in target if all_algorithms.get(name, {}).get("type") == "dynamic"
    ]
    static_algorithms = {
        name: {k: v for k, v in algo.items() if k != "type"}
        for name, algo in all_algorithms.items()
        if algo.get("type", "static") == "static"
    }
    dynamic_algorithms = {
        name: {k: v for k, v in algo.items() if k != "type"}
        for name, algo in all_algorithms.items()
        if algo.get("type") == "dynamic"
    }
    return static_target, static_algorithms, dynamic_target, dynamic_algorithms


def load_algorithms(config_path: str = "config/algorithms.yaml") -> Dict[str, dict]:
    """Load and instantiate all target algorithms listed in the config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    snapshot_target, snapshot_algorithms, temporal_target, temporal_algorithms = _read_algorithm_sections(config)

    algorithms: Dict[str, dict] = {}

    for algo_type, target_names, algo_defs in (
        ("static", snapshot_target, snapshot_algorithms),
        ("dynamic", temporal_target, temporal_algorithms),
    ):
        for name in target_names:
            if name not in algo_defs:
                section_name = "snapshot_algorithms" if algo_type == "static" else "temporal_algorithms"
                print(f"Warning: '{name}' not found in {section_name} config, skipping.")
                continue

            algo_config = algo_defs[name]
            module_path = algo_config.get("module", "cdlib.algorithms")
            func_name = algo_config["function"]
            params = algo_config.get("params", {}) or {}
            clustering_type = algo_config.get("clustering_type", "overlap")

            try:
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
            except (ImportError, AttributeError) as exc:
                print(
                    f"Warning: Could not load '{func_name}' from '{module_path}': {exc}. "
                    f"Skipping '{name}'."
                )
                continue

            if isinstance(func, type):
                if issubclass(func, CommunityDetectionAlgorithm):
                    wrapper = func
                else:
                    instance = func(**params)
                    wrapper = DynamicMethodWrapper(instance, {}) if algo_type == "dynamic" else StaticMethodWrapper(instance, {})
            else:
                wrapper = DynamicMethodWrapper(func, params) if algo_type == "dynamic" else StaticMethodWrapper(func, params)

            algorithms[name] = {
                "wrapper": wrapper,
                "type": algo_type,
                "clustering_type": clustering_type,
                "params": params,
                "config": algo_config,
            }

    return algorithms
