"""Plot utilities for analyzer JSON reports.

This module focuses on plotting output produced by ``tools/analyze.py --save-json``.
It intentionally stays separate from ``src/visualization`` because report JSON
shape differs from merged benchmark metric files.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

from src.analyzer.models import MetricComparison, OverlapQualityReport


METRIC_CHOICES = (
    "participation_coefficient",
    "max_embeddedness",
    "betweenness_centrality",
)


def resolve_report_paths(inputs: Sequence[str]) -> List[Path]:
    """Resolve report file paths from files, directories, and glob patterns."""
    paths: List[Path] = []

    for item in inputs:
        candidate = Path(item)

        if candidate.exists() and candidate.is_file():
            paths.append(candidate)
            continue

        if candidate.exists() and candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
            continue

        for match in sorted(glob.glob(item)):
            path_match = Path(match)
            if path_match.is_file():
                paths.append(path_match)

    # Preserve order, drop duplicates
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)

    return deduped


def load_report(path: Path) -> OverlapQualityReport:
    """Load one overlap-quality report JSON into a typed model."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return OverlapQualityReport.model_validate(payload)


def load_reports(paths: Sequence[Path]) -> List[Tuple[Path, OverlapQualityReport]]:
    """Load many report files.

    Files that do not match ``OverlapQualityReport`` raise validation errors.
    """
    loaded: List[Tuple[Path, OverlapQualityReport]] = []
    for path in paths:
        loaded.append((path, load_report(path)))
    return loaded


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())


def _report_label(report: OverlapQualityReport) -> str:
    return f"{report.algorithm} | {report.dataset} ({report.benchmark_mode})"


def _metric_from_snapshot(
    report: OverlapQualityReport,
    snapshot_index: int,
    metric_name: str,
) -> MetricComparison | None:
    for snapshot in report.snapshot_reports:
        if snapshot.snapshot_index != snapshot_index:
            continue
        if metric_name == "participation_coefficient":
            return snapshot.participation_coefficient
        if metric_name == "max_embeddedness":
            return snapshot.max_embeddedness
        if metric_name == "betweenness_centrality":
            return snapshot.betweenness_centrality
    return None


def plot_overlap_fraction(
    reports: Sequence[Tuple[Path, OverlapQualityReport]],
    out_dir: Path,
) -> Path | None:
    """Plot overlap fraction per snapshot for one or many reports."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    for _, report in reports:
        snapshots = sorted(report.snapshot_reports, key=lambda s: s.snapshot_index)
        if not snapshots:
            continue

        x = [s.snapshot_index for s in snapshots]
        y = [s.overlap_fraction for s in snapshots]
        ax.plot(x, y, marker="o", linewidth=2, label=_report_label(report))
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title("Overlap Fraction by Snapshot")
    ax.set_xlabel("snapshot")
    ax.set_ylabel("overlap fraction")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_path = out_dir / "overlap_fraction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_temporal_stability(
    reports: Sequence[Tuple[Path, OverlapQualityReport]],
    out_dir: Path,
) -> Path | None:
    """Plot consecutive-snapshot ONMI for one or many reports."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    for _, report in reports:
        entries = sorted(report.temporal_stability, key=lambda t: t.snapshot_to)
        if not entries:
            continue

        x = [entry.snapshot_to for entry in entries]
        y = [entry.onmi for entry in entries]
        ax.plot(x, y, marker="o", linewidth=2, label=_report_label(report))
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title("Temporal Stability (ONMI t-1 -> t)")
    ax.set_xlabel("snapshot t")
    ax.set_ylabel("ONMI")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_path = out_dir / "temporal_stability_onmi.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_structural_effect_size(
    reports: Sequence[Tuple[Path, OverlapQualityReport]],
    out_dir: Path,
    metric_name: str,
) -> List[Path]:
    """Plot rank-biserial effect size per snapshot for each report."""
    out_paths: List[Path] = []

    for path, report in reports:
        snapshots = sorted(report.snapshot_reports, key=lambda s: s.snapshot_index)
        points = []
        for snapshot in snapshots:
            metric = _metric_from_snapshot(report, snapshot.snapshot_index, metric_name)
            if metric is None or metric.rank_biserial is None or metric.error:
                continue
            points.append((snapshot.snapshot_index, metric.rank_biserial))

        if not points:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y, marker="o", linewidth=2)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(
            f"Effect Size by Snapshot ({metric_name})\n{_report_label(report)}"
        )
        ax.set_xlabel("snapshot")
        ax.set_ylabel("rank-biserial")
        ax.grid(alpha=0.3)

        suffix = _safe_name(path.stem)
        out_path = out_dir / f"effect_size_{metric_name}_{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def plot_structural_medians(
    reports: Sequence[Tuple[Path, OverlapQualityReport]],
    out_dir: Path,
    metric_name: str,
) -> List[Path]:
    """Plot overlap vs non-overlap medians per snapshot for each report."""
    out_paths: List[Path] = []

    for path, report in reports:
        snapshots = sorted(report.snapshot_reports, key=lambda s: s.snapshot_index)
        points = []
        for snapshot in snapshots:
            metric = _metric_from_snapshot(report, snapshot.snapshot_index, metric_name)
            if metric is None or metric.error:
                continue
            if metric.group_a_median is None or metric.group_b_median is None:
                continue
            points.append((snapshot.snapshot_index, metric.group_a_median, metric.group_b_median))

        if not points:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        x = [p[0] for p in points]
        overlap_vals = [p[1] for p in points]
        non_overlap_vals = [p[2] for p in points]

        ax.plot(x, overlap_vals, marker="o", linewidth=2, label="overlap median")
        ax.plot(
            x,
            non_overlap_vals,
            marker="o",
            linewidth=2,
            label="non-overlap median",
        )
        ax.set_title(
            f"Medians by Snapshot ({metric_name})\n{_report_label(report)}"
        )
        ax.set_xlabel("snapshot")
        ax.set_ylabel("metric median")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

        suffix = _safe_name(path.stem)
        out_path = out_dir / f"medians_{metric_name}_{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def plot_all(
    reports: Sequence[Tuple[Path, OverlapQualityReport]],
    out_dir: Path,
    metric_name: str,
) -> List[Path]:
    """Generate all available analysis plots."""
    out_paths: List[Path] = []

    overlap_path = plot_overlap_fraction(reports, out_dir)
    if overlap_path is not None:
        out_paths.append(overlap_path)

    temporal_path = plot_temporal_stability(reports, out_dir)
    if temporal_path is not None:
        out_paths.append(temporal_path)

    out_paths.extend(plot_structural_effect_size(reports, out_dir, metric_name))
    out_paths.extend(plot_structural_medians(reports, out_dir, metric_name))

    return out_paths
