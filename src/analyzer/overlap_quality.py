"""Overlap quality analyzer — validates whether overlapping nodes occupy
structurally meaningful boundary positions.

Since v2, each ``NodeClustering`` carries its own ``graph``, so structural
analysis works for **both static AND dynamic** benchmarks — no need to
reload graphs from config files.

Entry point: ``run_overlap_quality(payload, clusterings, **kwargs) -> OverlapQualityReport``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import networkx as nx
import numpy as np
from cdlib import NodeClustering

from src.analyzer.metrics_accuracy import omega_index, onmi_consecutive
from src.analyzer.metrics_structural import (
    betweenness_centrality,
    max_embeddedness,
    participation_coefficient,
)
from src.analyzer.models import (
    ClusteringArtifactPayload,
    MetricComparison,
    OverlapQualityReport,
    SnapshotOverlapReport,
    TemporalStabilityEntry,
)
from src.analyzer.stats import mann_whitney_compare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_overlapping_nodes(communities: List[List[Any]]) -> set:
    """Return nodes that appear in more than one community."""
    seen: set = set()
    overlap: set = set()
    for comm in communities:
        for node in comm:
            if node in seen:
                overlap.add(node)
            else:
                seen.add(node)
    return overlap


def _split_metric_by_overlap(
    metric_values: Dict[Any, float],
    overlap_nodes: set,
) -> tuple[np.ndarray, np.ndarray]:
    """Split metric values into overlap / non-overlap arrays."""
    overlap_vals = []
    non_overlap_vals = []
    for node, val in metric_values.items():
        if node in overlap_nodes:
            overlap_vals.append(val)
        else:
            non_overlap_vals.append(val)
    return np.array(overlap_vals, dtype=float), np.array(
        non_overlap_vals, dtype=float
    )


def _build_comparison(
    metric_name: str,
    metric_values: Dict[Any, float],
    overlap_nodes: set,
) -> MetricComparison:
    """Run Mann-Whitney U on a structural metric, overlap vs non-overlap."""
    ov, non_ov = _split_metric_by_overlap(metric_values, overlap_nodes)

    if len(ov) == 0 or len(non_ov) == 0:
        return MetricComparison(
            metric_name=metric_name,
            group_a_n=len(ov),
            group_b_n=len(non_ov),
            error="No overlapping or non-overlapping nodes to compare",
        )

    raw = mann_whitney_compare(ov, non_ov)
    return MetricComparison(
        metric_name=metric_name,
        group_a_label=raw.get("group_a_label", "overlap"),
        group_b_label=raw.get("group_b_label", "non_overlap"),
        group_a_n=raw["group_a_n"],
        group_b_n=raw["group_b_n"],
        group_a_median=raw.get("group_a_median"),
        group_b_median=raw.get("group_b_median"),
        group_a_mean=raw.get("group_a_mean"),
        group_b_mean=raw.get("group_b_mean"),
        u_statistic=raw.get("u_statistic"),
        p_value=raw.get("p_value"),
        rank_biserial=raw.get("rank_biserial"),
        error=raw.get("error"),
    )


# ---------------------------------------------------------------------------
# Per-snapshot structural analysis
# ---------------------------------------------------------------------------


def analyze_snapshot(
    graph: nx.Graph,
    communities: List[List[Any]],
    snapshot_index: int,
    ground_truth: List[List[Any]] | None = None,
    betweenness_k: int | None = None,
) -> SnapshotOverlapReport:
    """Full structural overlap analysis for a single snapshot.

    Called when the snapshot's ``NodeClustering`` has a graph attached.
    """
    overlap_nodes = _find_overlapping_nodes(communities)

    # --- structural metrics ---
    pc_vals = participation_coefficient(graph, communities)
    me_vals = max_embeddedness(graph, communities)
    bc_vals = betweenness_centrality(graph, k=betweenness_k)

    pc_cmp = _build_comparison("participation_coefficient", pc_vals, overlap_nodes)
    me_cmp = _build_comparison("max_embeddedness", me_vals, overlap_nodes)
    bc_cmp = _build_comparison("betweenness_centrality", bc_vals, overlap_nodes)

    # --- ground-truth accuracy ---
    omega = None
    if ground_truth is not None:
        omega = omega_index(communities, ground_truth, graph)

    return SnapshotOverlapReport(
        snapshot_index=snapshot_index,
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        num_communities=len(communities),
        num_overlapping_nodes=len(overlap_nodes),
        overlap_fraction=(
            len(overlap_nodes) / graph.number_of_nodes()
            if graph.number_of_nodes() > 0
            else 0.0
        ),
        participation_coefficient=pc_cmp,
        max_embeddedness=me_cmp,
        betweenness_centrality=bc_cmp,
        omega_index=omega,
    )


# ---------------------------------------------------------------------------
# Temporal stability
# ---------------------------------------------------------------------------


def analyze_temporal_stability(
    snapshots_communities: List[List[List[Any]]],
) -> List[TemporalStabilityEntry]:
    """ONMI between consecutive snapshot community assignments."""
    entries: List[TemporalStabilityEntry] = []
    for i in range(1, len(snapshots_communities)):
        score = onmi_consecutive(
            snapshots_communities[i - 1],
            snapshots_communities[i],
        )
        entries.append(
            TemporalStabilityEntry(
                snapshot_from=i - 1,
                snapshot_to=i,
                onmi=score,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Report formatting (plain text)
# ---------------------------------------------------------------------------


def _fmt_comparison(cmp: MetricComparison, indent: str = "    ") -> str:
    """Format a single MetricComparison as readable text lines."""
    lines: List[str] = []
    lines.append(f"{indent}{cmp.metric_name}:")
    if cmp.error:
        lines.append(f"{indent}  (skipped: {cmp.error})")
        return "\n".join(lines)

    lines.append(
        f"{indent}  overlap    (n={cmp.group_a_n:>5}): "
        f"median={cmp.group_a_median:.4f}  mean={cmp.group_a_mean:.4f}"
    )
    lines.append(
        f"{indent}  non-overlap(n={cmp.group_b_n:>5}): "
        f"median={cmp.group_b_median:.4f}  mean={cmp.group_b_mean:.4f}"
    )
    if cmp.u_statistic is not None:
        lines.append(
            f"{indent}  U={cmp.u_statistic:.1f}  "
            f"p={cmp.p_value:.2e}  "
            f"r_rb={cmp.rank_biserial:.4f}"
        )
    return "\n".join(lines)


def format_report(report: OverlapQualityReport) -> str:
    """Format an OverlapQualityReport as human-readable text."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("OVERLAP QUALITY REPORT")
    lines.append("=" * 70)
    lines.append(f"Algorithm:       {report.algorithm}")
    lines.append(f"Dataset:         {report.dataset}")
    lines.append(f"Mode:            {report.benchmark_mode}")
    lines.append(f"Clustering type: {report.clustering_type}")
    lines.append(f"Snapshots:       {report.num_snapshots}")
    lines.append("")

    # Membership summary
    lines.append("--- Membership summary ---")
    lines.append(f"Communities/snapshot: {report.community_counts}")
    lines.append(f"Overlap nodes/snapshot: {report.overlap_node_counts}")
    lines.append("")

    # Per-snapshot structural reports
    for sr in report.snapshot_reports:
        lines.append(f"--- Snapshot {sr.snapshot_index} ---")
        lines.append(
            f"  Nodes: {sr.num_nodes}  Edges: {sr.num_edges}  "
            f"Communities: {sr.num_communities}"
        )
        lines.append(
            f"  Overlapping nodes: {sr.num_overlapping_nodes} "
            f"({sr.overlap_fraction:.1%})"
        )

        if sr.participation_coefficient:
            lines.append(_fmt_comparison(sr.participation_coefficient))
        if sr.max_embeddedness:
            lines.append(_fmt_comparison(sr.max_embeddedness))
        if sr.betweenness_centrality:
            lines.append(_fmt_comparison(sr.betweenness_centrality))
        if sr.omega_index is not None:
            lines.append(f"    omega_index: {sr.omega_index:.4f}")
        lines.append("")

    # Temporal stability
    if report.temporal_stability:
        lines.append("--- Temporal stability (ONMI consecutive) ---")
        for ts in report.temporal_stability:
            lines.append(
                f"  t{ts.snapshot_from} -> t{ts.snapshot_to}: "
                f"ONMI = {ts.onmi:.4f}"
            )
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_overlap_quality(
    payload: ClusteringArtifactPayload,
    clusterings: Optional[List[NodeClustering]] = None,
    betweenness_k: int | None = None,
) -> OverlapQualityReport:
    """Run the full overlap quality analysis.

    With v2 pickle artifacts, ``clusterings`` carries graphs and communities
    directly — structural analysis works for every snapshot that has a graph.

    Falls back to metadata-only analysis (membership distributions +
    temporal ONMI) when ``clusterings`` is ``None`` (v1 backward compat).

    Args:
        payload: The clustering artifact metadata.
        clusterings: Unpickled ``List[NodeClustering]`` (``None`` for v1).
        betweenness_k: Sample size for approximate betweenness centrality.
            ``None`` = exact (can be slow for large graphs).

    Returns:
        Populated ``OverlapQualityReport``.
    """
    # --- Extract communities from either source ---
    communities_per_snap: List[List[List[Any]]]
    community_counts: List[int]
    overlap_node_counts: List[int]

    if clusterings is not None and len(clusterings) > 0:
        communities_per_snap = [
            [list(c) for c in nc.communities] for nc in clusterings
        ]
        community_counts = [len(nc.communities) for nc in clusterings]
        overlap_node_counts = [
            len(_find_overlapping_nodes([list(c) for c in nc.communities]))
            for nc in clusterings
        ]
    else:
        # v1 fallback: no communities in payload (v2) — report what we can
        communities_per_snap = []
        community_counts = [s.num_communities for s in payload.snapshots]
        overlap_node_counts = []

    # --- Temporal stability (multi-snapshot) ---
    temporal: List[TemporalStabilityEntry] = []
    if len(communities_per_snap) > 1:
        temporal = analyze_temporal_stability(communities_per_snap)

    # --- Structural analysis per snapshot (v2: use nc.graph) ---
    snapshot_reports: List[SnapshotOverlapReport] = []

    if clusterings is not None:
        for idx, nc in enumerate(clusterings):
            graph: nx.Graph | None = cast(Optional[nx.Graph], nc.graph)
            if graph is None:
                continue  # skip snapshots without a graph
            communities = [list(c) for c in nc.communities]
            sr = analyze_snapshot(
                graph=graph,
                communities=communities,
                snapshot_index=idx,
                ground_truth=None,  # TODO: support ground truth from pickle
                betweenness_k=betweenness_k,
            )
            snapshot_reports.append(sr)

    report = OverlapQualityReport(
        algorithm=payload.algorithm,
        dataset=payload.dataset,
        benchmark_mode=payload.benchmark_mode,
        clustering_type=payload.clustering_type,
        num_snapshots=payload.num_snapshots,
        snapshot_reports=snapshot_reports,
        temporal_stability=temporal,
        community_counts=community_counts,
        overlap_node_counts=overlap_node_counts,
    )

    # Print the report
    print(format_report(report))

    return report
