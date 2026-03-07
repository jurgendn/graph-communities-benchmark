"""
Factory for loading and instantiating community detection algorithms
from ``config/algorithms.yaml``.
"""
import importlib
from typing import Dict

import yaml

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.wrappers import DynamicMethodWrapper, StaticMethodWrapper


def load_algorithms(config_path: str = "config/algorithms.yaml") -> Dict[str, dict]:
    """
    Load and instantiate all target algorithms listed in the config file.

    Each algorithm entry in the YAML must have:
    - ``module`` dotted import path (e.g. ``"cdlib.algorithms"``)
    - ``function`` name of the callable to import from ``module``
    - ``params`` keyword arguments passed to the callable (may be empty)
    - ``type`` ``"static"`` or ``"dynamic"``

    If ``function`` resolves to a *class* (e.g. ``BigClam``), it is
    instantiated with ``**params`` and wrapped with empty params so that
    wrapper does not double-pass arguments.  If it resolves to a plain
    function, the function itself is wrapped along with its params.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict mapping algorithm name → entry dict with keys:
        - ``"wrapper"`` — a ``StaticMethodWrapper`` or ``DynamicMethodWrapper``
        - ``"type"`` — ``"static"`` or ``"dynamic"``
        - ``"params"`` — the params dict from config (for logging)
        - ``"config"`` — the full algorithm config dict
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    target = config.get("target_algorithms", [])
    all_algos = config.get("algorithms", {})

    algorithms: Dict[str, dict] = {}
    for name in target:
        if name not in all_algos:
            print(f"Warning: '{name}' not found in algorithms config, skipping.")
            continue

        algo_config = all_algos[name]
        module_path = algo_config.get("module", "cdlib.algorithms")
        func_name = algo_config["function"]
        params = algo_config.get("params", {}) or {}
        algo_type = algo_config.get("type", "static")
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

        # Handle classes vs functions
        if isinstance(func, type):
            if issubclass(func, CommunityDetectionAlgorithm):
                # CDA classes implement __call__(tg) -> List[NodeClustering].
                # Wrap the class directly; let __call__ handle algorithm logic.
                wrapper = func
            else:
                # Non-CDA class: instantiate with params and wrap.
                instance = func(**params)
                if algo_type == "dynamic":
                    wrapper = DynamicMethodWrapper(instance, {})
                else:
                    wrapper = StaticMethodWrapper(instance, {})
        else:
            # Plain function: wrap with its params.
            if algo_type == "dynamic":
                wrapper = DynamicMethodWrapper(func, params)
            else:
                wrapper = StaticMethodWrapper(func, params)

        algorithms[name] = {
            "wrapper": wrapper,
            "type": algo_type,
            "clustering_type": clustering_type,
            "params": params,
            "config": algo_config,
        }

    return algorithms
