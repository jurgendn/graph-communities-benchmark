"""Configuration loading helpers for visualization tooling."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_MODE = "dynamic"


class ConfigManager:
    """Single configuration manager for visualization workflows."""

    def __init__(self, root: Path = Path(".")):
        self.root = Path(root)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _cfg_path(self, benchmark_type: str) -> Path:
        return self.root / "config" / f"visualization_{benchmark_type}.yaml"

    def _load_yaml(self, benchmark_type: str) -> Dict[str, Any]:
        cfg_path = self._cfg_path(benchmark_type)
        if not cfg_path.exists():
            return {}
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def get(self, benchmark_type: str = DEFAULT_MODE) -> Dict[str, Any]:
        if benchmark_type not in self._cache:
            self._cache[benchmark_type] = self._load_yaml(benchmark_type)
        return self._cache[benchmark_type]

    def available_modes(self) -> List[str]:
        available = []
        for benchmark_type in ("dynamic", "static"):
            if self._cfg_path(benchmark_type).exists():
                available.append(benchmark_type)
        return available or [DEFAULT_MODE]

    def mode_config(self, benchmark_type: str = DEFAULT_MODE) -> Dict[str, Any]:
        config = self.get(benchmark_type)
        if config:
            return config
        return self.get(DEFAULT_MODE)

    def directories(self, benchmark_type: str = DEFAULT_MODE) -> Dict[str, Path]:
        mode_cfg = self.mode_config(benchmark_type)
        directories = mode_cfg.get("directories", {})
        return {
            "raw_dir": Path(directories.get("raw_dir", f"experiments/{benchmark_type}/raw")),
            "merge_dir": Path(directories.get("merge_dir", f"experiments/{benchmark_type}/merged")),
            "output_dir": Path(directories.get("output_dir", f"assets/{benchmark_type}")),
        }

    def workspace(self) -> Optional[str]:
        for benchmark_type in self.available_modes():
            workspace = self.get(benchmark_type).get("workspace")
            if workspace:
                return workspace
        return None

    def metric_keys(self, benchmark_type: str = DEFAULT_MODE) -> List[str]:
        return self.mode_config(benchmark_type).get("metric_keys", [])

    def projects(self, benchmark_type: str = DEFAULT_MODE) -> List[str]:
        return self.mode_config(benchmark_type).get("projects", [])

    def hyperparameters(self, benchmark_type: str = DEFAULT_MODE) -> List[str]:
        return self.mode_config(benchmark_type).get("hyperparameters", [])

    def plotter(self, benchmark_type: str = DEFAULT_MODE) -> Dict[str, Any]:
        return self.mode_config(benchmark_type).get("plotter", {})

    def uses_batch_ranges(self, benchmark_type: str = DEFAULT_MODE) -> bool:
        return bool(self.mode_config(benchmark_type).get("use_batch_ranges", False))

    def dataset_info(self, benchmark_type: str = DEFAULT_MODE) -> Dict[str, Any]:
        plotter_cfg = self.plotter(benchmark_type)
        common = plotter_cfg.get("common_plotter_settings", {})
        return {
            "synthetic": common.get("synthetic_datasets", []),
            "real_world": common.get("real_world_datasets", []),
            "name_mapping": common.get("projects_name_mapping", {}),
        }

    def algorithm_info(self, benchmark_type: str = DEFAULT_MODE) -> Dict[str, Any]:
        plotter_cfg = self.plotter(benchmark_type)
        common = plotter_cfg.get("common_plotter_settings", {})
        return {
            "selected": common.get("selected_algorithms", []),
            "names": common.get("methods_name", {}),
            "colors": common.get("colors", {}),
            "markers": common.get("markers", {}),
            "order": common.get("orders", []),
        }

    def batch_range_categories(self, benchmark_type: str = DEFAULT_MODE) -> Dict[str, float]:
        plotter_cfg = self.plotter(benchmark_type)
        common = plotter_cfg.get("common_plotter_settings", {})
        return common.get("batch_range_categories", {})
