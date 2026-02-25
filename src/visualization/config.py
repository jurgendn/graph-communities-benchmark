"""
Simple unified configuration loading.
Parses config/metadata.yaml and caches results.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigManager:
    """Single configuration manager for the entire project."""

    def __init__(self, root: Path = Path(".")):
        self.root = Path(root)
        self.cfg_path = self.root / "config" / "visualization.yaml"
        self._cache: Dict[str, Any] = {}
        self._loaded = False

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML with fallback to basic parsing."""
        if not self.cfg_path.exists():
            return {}

        # Try pyyaml first
        try:
            import yaml
            with open(self.cfg_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except ImportError:
            pass

        # Fallback: basic manual parsing for simple YAML
        return self._parse_yaml_basic()

    def _parse_yaml_basic(self) -> Dict[str, Any]:
        """Basic YAML parser for simple structures."""
        result = {}
        try:
            with open(self.cfg_path, "r", encoding="utf-8") as fh:
                lines = [line.rstrip("\n") for line in fh]

            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                # Skip empty/comment lines
                if not stripped or stripped.startswith("#"):
                    i += 1
                    continue

                # Top-level key with list value
                if not line.startswith(" ") and stripped.endswith(":"):
                    key = stripped[:-1]
                    items = []
                    i += 1

                    # Collect list items
                    while i < len(lines):
                        lline = lines[i]
                        if not lline.strip() or lline.strip().startswith("#"):
                            i += 1
                            continue
                        if not lline.startswith(" "):
                            break
                        if lline.strip().startswith("- "):
                            items.append(lline.strip()[2:].strip().strip('"').strip("'"))
                            i += 1
                        else:
                            break

                    result[key] = items if items else None
                else:
                    i += 1

            return result
        except Exception:
            return {}

    def get(self) -> Dict[str, Any]:
        """Get full config (cached)."""
        if not self._loaded:
            self._cache = self._load_yaml()
            self._loaded = True
        return self._cache

    def directories(self) -> Dict[str, Path]:
        """Get directory settings."""
        return {
            "raw_dir": Path(self.get().get("raw_dir", "experiments/raw")),
            "merge_dir": Path(self.get().get("merge_dir", "experiments/merged")),
            "output_dir": Path(self.get().get("output_dir", "assets/grouped")),
        }

    def workspace(self) -> Optional[str]:
        """Get workspace name."""
        return self.get().get("workspace")

    def metric_keys(self) -> List[str]:
        """Get list of metrics to process."""
        return self.get().get("metric_keys", [])

    def projects(self) -> List[str]:
        """Get list of projects."""
        return self.get().get("projects", [])

    def hyperparameters(self) -> List[str]:
        """Get list of hyperparameters."""
        return self.get().get("hyperparameters", [])

    def plotter(self) -> Dict[str, Any]:
        """Get plotter configuration."""
        return self.get().get("plotter", {})

    def dataset_info(self) -> Dict[str, Any]:
        """Get synthetic/real-world dataset info."""
        plotter_cfg = self.plotter()
        common = plotter_cfg.get("common_plotter_settings", {})

        return {
            "synthetic": common.get("synthetic_datasets", []),
            "real_world": common.get("real_world_datasets", []),
            "name_mapping": common.get("projects_name_mapping", {}),
        }

    def algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm styling/ordering info."""
        plotter_cfg = self.plotter()
        common = plotter_cfg.get("common_plotter_settings", {})

        return {
            "selected": common.get("selected_algorithms", []),
            "names": common.get("methods_name", {}),
            "colors": common.get("colors", {}),
            "markers": common.get("markers", {}),
            "order": common.get("orders", []),
        }

    def batch_range_categories(self) -> Dict[str, float]:
        """Get batch range categories (name -> threshold)."""
        plotter_cfg = self.plotter()
        common = plotter_cfg.get("common_plotter_settings", {})
        return common.get("batch_range_categories", {})
