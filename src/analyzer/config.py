"""Configuration helpers for analyzer tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


DEFAULT_ANALYZER_CONFIG = Path("config/analyzer.yaml")


class AnalyzerConfigManager:
    """Load and access analyzer YAML config with safe defaults."""

    def __init__(self, config_path: str | Path = DEFAULT_ANALYZER_CONFIG):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] | None = None

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}

    def get(self) -> Dict[str, Any]:
        if self._config is None:
            self._config = self._load()
        return self._config

    def workspace(self) -> str | None:
        value = self.get().get("workspace")
        return value if isinstance(value, str) and value else None

    def directories(self) -> Dict[str, Path]:
        dirs = self.get().get("directories", {})
        if not isinstance(dirs, dict):
            dirs = {}
        return {
            "artifact_download_dir": Path(
                dirs.get("artifact_download_dir", "experiments/analyzer/artifacts")
            ),
            "report_dir": Path(dirs.get("report_dir", "experiments/analyzer/reports")),
            "plot_output_dir": Path(dirs.get("plot_output_dir", "assets/analysis")),
        }

    def analysis(self) -> Dict[str, Any]:
        section = self.get().get("analysis", {})
        return section if isinstance(section, dict) else {}

    def selection(self) -> Dict[str, Any]:
        section = self.get().get("selection", {})
        return section if isinstance(section, dict) else {}

    def plotting(self) -> Dict[str, Any]:
        section = self.get().get("plotting", {})
        return section if isinstance(section, dict) else {}

    def explicit_artifacts(self) -> List[str]:
        value = self.selection().get("explicit_artifacts", [])
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item]

    def artifact_group(self, name: str) -> List[str]:
        groups = self.selection().get("artifact_groups", {})
        if not isinstance(groups, dict):
            return []
        value = groups.get(name, [])
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item]

    def report_group(self, name: str) -> List[str]:
        groups = self.plotting().get("report_groups", {})
        if not isinstance(groups, dict):
            return []
        value = groups.get(name, [])
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item]
