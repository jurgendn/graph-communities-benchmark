"""
Visualization module - plots for benchmark metrics.
Includes all plot types (line, box, bar) and grouped plotter.
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.visualization.config import ConfigManager
from src.visualization.utils import (
    aggregate_series,
    algorithm_color,
    algorithm_display_name,
    algorithm_marker,
    get_algorithm_info,
    load_metric_file,
    project_display_name,
    sort_and_filter_algorithms,
    summarize_run,
)


class Plot:
    """Plot metrics with line, box, and bar chart options."""

    def __init__(self, project: str, metric_file: Path, out_dir: Path, aggregate: str = "last", batch_range: Optional[str] = None):
        self.project = project
        self.metric_file = Path(metric_file)
        self.batch_range = batch_range
        # Include batch_range in output path if provided
        out_path = Path(out_dir) / "per_datasets" / Path(project).name
        if batch_range:
            out_path = out_path / batch_range
        self.out_dir = out_path
        self.aggregate = aggregate
        self.metric_name = self.metric_file.stem
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> Dict:
        """Load and prepare metric data."""
        return load_metric_file(self.metric_file)

    def _get_algorithms(self) -> List[str]:
        """Get sorted and filtered algorithm list."""
        data = self._load_data()
        algs = list(data.keys())
        return sort_and_filter_algorithms(algs)

    def _save_fig(self, fig, suffix: str) -> Optional[Path]:
        """Save figure to disk."""
        out_path = self.out_dir / f"{self.metric_name}_{suffix}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _add_labels(self, ax):
        """Add project name and labels to axes."""
        proj_name = project_display_name(Path(self.project).name)
        ax.text(0.02, 0.95, proj_name, transform=ax.transAxes,
                fontsize=9, va="top", ha="left", weight="bold")
        ax.set_ylabel(self.metric_name)

    def plot_line(self) -> Optional[Path]:
        """Plot line chart with multiple runs aggregated by step."""
        data = self._load_data()
        algs = self._get_algorithms()

        if not algs:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        any_plotted = False

        for alg in algs:
            runs = data.get(alg, [])
            if not runs:
                continue

            series = aggregate_series(runs)
            if not series:
                continue

            steps = sorted(series.keys())
            means = [np.mean(series[s]) for s in steps]

            color = algorithm_color(alg)
            marker = algorithm_marker(alg) or "o"
            label = algorithm_display_name(alg)

            ax.plot(steps, means, marker=marker, label=label,
                   color=color, linewidth=2, markersize=6)
            any_plotted = True

        if not any_plotted:
            return None

        ax.set_xlabel("step")
        ax.legend(loc="best")
        self._add_labels(ax)
        plt.tight_layout()
        return self._save_fig(fig, "line")

    def plot_box(self) -> Optional[Path]:
        """Plot box chart of aggregated runs."""
        data = self._load_data()
        algs = self._get_algorithms()

        if not algs:
            return None

        groups = []
        labels = []

        for alg in algs:
            runs = data.get(alg, [])
            vals = [summarize_run(run, self.aggregate) for run in runs]
            vals_clean: List[float] = [v for v in vals if v is not None and not np.isnan(v)]

            if vals_clean:
                groups.append(vals_clean)
                labels.append(alg)

        if not groups:
            return None

        w = max(6, len(labels) * 0.7)
        fig, ax = plt.subplots(figsize=(w, 6))

        labels_disp = [algorithm_display_name(alg) for alg in labels]
        bp = ax.boxplot(groups, tick_labels=labels_disp, patch_artist=True, showfliers=True)

        # Color boxes
        for patch, alg in zip(bp.get("boxes", []), labels):
            color = algorithm_color(alg)
            if color:
                try:
                    patch.set_facecolor(color)
                except Exception:
                    pass

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        self._add_labels(ax)
        plt.tight_layout()
        return self._save_fig(fig, "box")

    def plot_bar(self) -> Optional[Path]:
        """Plot bar chart with error bars."""
        data = self._load_data()
        algs = self._get_algorithms()

        if not algs:
            return None

        means: List[float] = []
        stds: List[float] = []
        labels: List[str] = []

        for alg in algs:
            runs = data.get(alg, [])
            vals = [summarize_run(run, self.aggregate) for run in runs]
            vals_clean = [v for v in vals if v is not None and not np.isnan(v)]

            if vals_clean:
                means.append(float(np.mean(vals_clean)))
                stds.append(float(np.std(vals_clean, ddof=0)))
                labels.append(alg)

        if not labels:
            return None

        w = max(6, len(labels) * 0.7)
        fig, ax = plt.subplots(figsize=(w, 6))

        x = np.arange(len(labels))
        labels_disp = [algorithm_display_name(alg) for alg in labels]

        # Draw bars with individual colors
        for i, (mean, std, alg) in enumerate(zip(means, stds, labels)):
            color = algorithm_color(alg) or "C0"
            bar = ax.bar(i, mean, yerr=std, capsize=5, color=color, alpha=0.8)
            # Add value on top of bar
            ax.text(i, mean + std + 0.01 * max(means), f"{mean:.2f}", 
                   ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_disp)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        self._add_labels(ax)
        plt.tight_layout()
        return self._save_fig(fig, "bar")


class ModularityPlot(Plot):
    """Box, bar, and line plots for modularity metrics."""
    pass


class CommunitiesPlot(Plot):
    """Line plot for communities metrics."""
    pass


class RuntimePlot(Plot):
    """Line and bar plots for runtime metrics."""
    pass


class GroupedPlotter:
    """Create grouped multi-dataset figures."""

    def __init__(self, merge_root: Path = Path("data/merge"), out_base: Path = Path("assets/grouped")):
        self.merge_root = Path(merge_root)
        self.out_base = Path(out_base)
        self.out_base.mkdir(parents=True, exist_ok=True)

    def plot(self, metric: str = "num_communities") -> List[Path]:
        """Generate grouped plots for a metric across datasets and batch ranges."""
        out_paths = []

        # Get configuration
        cfg = ConfigManager()
        plotter_cfg = cfg.plotter()
        grouped = plotter_cfg.get("grouped_plotter", {}) or {}
        group_by = grouped.get("group_by", ["real_world_datasets"])
        batch_range_categories = cfg.batch_range_categories()

        # Build legend once
        alg_info = get_algorithm_info()
        legend_handles, legend_labels = self._build_legend(alg_info)

        # Determine plot modes
        modes: List[Optional[str]] = ["box", "bar", "line"] if metric in {"cdlib_modularity", "customize_q0_modularity"} else [None]

        # Get dataset info
        dataset_info = cfg.dataset_info()

        # Generate plots for each group and batch range
        for group_name in group_by:
            # Get datasets for this group
            datasets = dataset_info.get(group_name.replace("_datasets", ""), [])
            if not datasets:
                continue
            
            # Get batch ranges available for this specific group
            batch_ranges = self._get_batch_ranges_for_group(group_name, datasets)
            
            for batch_range in batch_ranges:
                print(f"  Batch Range: {batch_range}, Group: {group_name}")
                for mode in modes:
                    path = self._plot_group(
                        metric=metric,
                        group_name=group_name,
                        mode=mode,
                        legend_handles=legend_handles,
                        legend_labels=legend_labels,
                        batch_range=batch_range,
                    )
                    if path:
                        out_paths.append(path)

        return out_paths

    def _get_available_batch_ranges(self) -> List[str]:
        """Get list of available batch ranges from merge directory."""
        batch_ranges = set()
        if self.merge_root.exists():
            for project_dir in self.merge_root.iterdir():
                if project_dir.is_dir():
                    for item in project_dir.iterdir():
                        if item.is_dir() and item.name not in ['.', '..']:
                            batch_ranges.add(item.name)
        return sorted(batch_ranges) if batch_ranges else [""]

    def _get_batch_ranges_for_group(self, group_name: str, datasets: List[str]) -> List[str]:
        """Get batch ranges available for a specific group of datasets."""
        batch_ranges = set()
        for dataset in datasets:
            dataset_path = self.merge_root / dataset
            if dataset_path.exists():
                for item in dataset_path.iterdir():
                    if item.is_dir() and item.name not in ['.', '..'] and item.name != "unknown":
                        batch_ranges.add(item.name)
        return sorted(batch_ranges) if batch_ranges else [""]

    def _plot_group(self, metric: str, group_name: str, mode: Optional[str],
                   legend_handles: List, legend_labels: List[str], batch_range: str = "") -> Optional[Path]:
        """Plot a single group of datasets for a specific batch range."""
        # Get datasets for this group
        cfg = ConfigManager()
        dataset_info = cfg.dataset_info()

        datasets = dataset_info.get(group_name.replace("_datasets", ""), [])
        if not datasets:
            return None

        # Get layout
        rows, cols = self._get_layout(group_name, len(datasets))
        datasets = datasets[: rows * cols]

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=False, sharey=False)
        fig.subplots_adjust(top=0.95, bottom=0.18, hspace=0.28, wspace=0.18)

        if rows * cols == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten() # type: ignore

        # Plot each subplot
        for ax, dataset_name in zip(axes_flat, datasets):
            # Build path with batch range
            if batch_range:
                metric_file = self.merge_root / dataset_name / batch_range / f"{metric}.json"
            else:
                metric_file = self.merge_root / dataset_name / f"{metric}.json"
            if not metric_file.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                ax.axis("off")
                continue

            data = load_metric_file(metric_file)
            algs = sort_and_filter_algorithms(list(data.keys()))

            # Check if we have any data
            if not algs or not data:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                ax.axis("off")
                continue

            # Draw plot
            if metric in {"cdlib_modularity", "customize_q0_modularity"}:
                self._plot_modularity(ax, data, algs, metric, mode)
            elif metric == "runtime":
                self._plot_runtime(ax, data, algs, metric)
            else:
                self._plot_timeseries(ax, data, algs, metric)

            # Remove ylabel except for leftmost column
            if ax != axes_flat[0]:
                ax.set_ylabel("")

            # Set title
            disp_name = project_display_name(dataset_name)
            ax.set_title(disp_name, fontsize=9, fontweight="normal", pad=6)
            ax.tick_params(axis="y", labelsize=8)

        # Hide empty subplots
        for ax in axes_flat[len(datasets):]:
            ax.axis("off")

        # Add legend
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.045),
            ncol=min(5, len(legend_handles)),
            fontsize=10,
        )

        # Save with batch range in path
        suffix = mode if mode else "_"
        batch_folder = batch_range if batch_range else "all"
        parent_folder = self.out_base / f"{metric}" / batch_folder / f"{group_name}"
        parent_folder.mkdir(parents=True, exist_ok=True)
        out_path = parent_folder / f"{suffix}.png"
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

        return out_path

    def _plot_modularity(self, ax, data: Dict, algs: List[str], metric: str, mode: Optional[str]) -> None:
        labels, means, stds = self._aggregate_modularity(data, algs)

        if mode == "box":
            groups = [data.get(alg, []) for alg in labels]
            groups = [[summarize_run(run, "last") for run in runs] for runs in groups]
            groups = [[v for v in g if v is not None and not np.isnan(v)] for g in groups]
            labels_disp = [algorithm_display_name(alg) for alg in labels]
            ax.boxplot(groups, tick_labels=labels_disp, patch_artist=True)

            # Color boxes
            for patch, alg in zip(ax.findobj(Patch), labels):
                color = algorithm_color(alg)
                if color:
                    patch.set_facecolor(color)

        elif mode == "bar":
            x = np.arange(len(labels))
            for i, (mean, std, alg) in enumerate(zip(means, stds, labels)):
                color = algorithm_color(alg) or "C0"
                ax.bar(i, mean, yerr=std, capsize=4, color=color, alpha=0.9)
                # Add value on top of bar
                ax.text(i, mean + std + 0.01 * max(means), f"{mean:.2f}", 
                       ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_xticks([])

        elif mode == "line":
            x = np.arange(len(labels))
            ax.plot(x, means, color="black", linestyle="--", linewidth=2, marker="o", markersize=4)
            ax.set_xticks([])

        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

    def _plot_runtime(self, ax, data: Dict, algs: List[str], metric: str):
        """Plot runtime as bars."""
        labels, means, stds = self._aggregate_modularity(data, algs)

        x = np.arange(len(labels))
        for i, (mean, std, alg) in enumerate(zip(means, stds, labels)):
            color = algorithm_color(alg) or "C0"
            ax.bar(i, mean, yerr=std, color=color, alpha=0.9)
            # Add value on top of bar
            ax.text(i, mean + std + 0.01 * max(means), f"{mean:.2f}", 
                   ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks([])
        ax.set_ylabel("seconds (s)")
        ax.grid(axis="y", alpha=0.3)

    def _plot_timeseries(self, ax, data: Dict, algs: List[str], metric: str):
        """Plot time series."""
        for alg in algs:
            runs = data.get(alg, [])
            if not runs:
                continue

            series = aggregate_series(runs)
            if not series:
                continue

            steps = sorted(series.keys())
            means = [np.mean(series[s]) for s in steps]

            color = algorithm_color(alg)
            marker = algorithm_marker(alg) or "o"

            ax.plot(steps, means, marker=marker, color=color, linewidth=2, markersize=4)

        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)

    def _build_legend(self, alg_info: Dict) -> tuple:
        """Build legend patches and labels for selected algorithms only."""
        handles = []
        labels = []
        selected = set(alg_info.get("selected", []))

        for alg in alg_info.get("order", []):
            if selected and alg not in selected:
                continue
            color = alg_info.get("colors", {}).get(alg)
            if not color:
                continue

            handles.append(Patch(facecolor=color, edgecolor="black"))
            labels.append(algorithm_display_name(alg))

        return handles, labels

    def _aggregate_modularity(self, data: Dict, algs: List[str]) -> tuple[List[str], List[float], List[float]]:
        """Aggregate modularity stats (labels, means, stds)."""
        labels, means, stds = [], [], []

        for alg in algs:
            runs = data.get(alg, [])
            vals = [summarize_run(run, "last") for run in runs]
            vals_clean: List[float] = [v for v in vals if v is not None and not np.isnan(v)]

            if vals_clean:
                labels.append(alg)
                means.append(float(np.mean(vals_clean)))
                stds.append(float(np.std(vals_clean, ddof=0)))

        return labels, means, stds

    def _get_layout(self, group_name: str, num_datasets: int) -> tuple:
        """Get subplot layout (rows, cols)."""
        cfg = ConfigManager()
        plotter_cfg = cfg.plotter()
        grouped = plotter_cfg.get("grouped_plotter", {})
        layouts = grouped.get("subplots_layout", {}) or {}

        if group_name in layouts:
            layout = layouts[group_name]
            return int(layout[0]), int(layout[1])

        # Default layouts
        if "synthetic" in group_name:
            return 1, 2
        return 2, 5
