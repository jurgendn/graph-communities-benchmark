"""Orchestrate artifact download and analysis execution.

Keeps I/O (Comet download, pickle loading) separated from analysis logic
so that analyzer modules only receive plain data objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from cdlib import NodeClustering

from src.analyzer.artifacts import download_artifact, load_clusterings, load_payload
from src.analyzer.models import ClusteringArtifactPayload
from src.analyzer.overlap_quality import run_overlap_quality


# ---------------------------------------------------------------------------
# Built-in analyzers
# ---------------------------------------------------------------------------


def default_summary(
    payload: ClusteringArtifactPayload,
    clusterings: Optional[List[NodeClustering]] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Print and return a basic summary of the clustering artifact."""
    summary: Dict[str, Any] = {
        "algorithm": payload.algorithm,
        "algorithm_type": payload.algorithm_type,
        "clustering_type": payload.clustering_type,
        "dataset": payload.dataset,
        "benchmark_mode": payload.benchmark_mode,
        "num_snapshots": payload.num_snapshots,
        "avg_runtime": payload.avg_runtime,
        "avg_modularity": payload.avg_modularity,
        "avg_nmi": payload.avg_nmi,
    }

    # Per-snapshot community counts
    if clusterings is not None:
        community_counts = [len(nc.communities) for nc in clusterings]
    else:
        community_counts = [s.num_communities for s in payload.snapshots]
    summary["community_counts"] = community_counts

    has_pickle = clusterings is not None
    summary["pickle_available"] = has_pickle

    print(f"Algorithm:       {payload.algorithm}")
    print(f"Dataset:         {payload.dataset}")
    print(f"Mode:            {payload.benchmark_mode}")
    print(f"Snapshots:       {payload.num_snapshots}")
    print(f"Clustering type: {payload.clustering_type}")
    print(f"Pickle data:     {'yes' if has_pickle else 'no (v1 JSON-only)'}")
    if payload.avg_runtime is not None:
        print(f"Avg runtime:     {payload.avg_runtime:.4f}s")
    if payload.avg_modularity is not None:
        print(f"Avg modularity:  {payload.avg_modularity:.4f}")
    if payload.avg_nmi is not None:
        print(f"Avg NMI:         {payload.avg_nmi:.4f}")
    print(f"Communities/snapshot: {community_counts}")

    return summary


def _overlap_quality_wrapper(
    payload: ClusteringArtifactPayload,
    clusterings: Optional[List[NodeClustering]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Thin wrapper so run_overlap_quality matches the analyzer signature."""
    betweenness_k = kwargs.get("betweenness_k")
    report = run_overlap_quality(
        payload, clusterings=clusterings, betweenness_k=betweenness_k
    )
    return report.model_dump()


# Analyzer registry: name -> callable(payload, clusterings, **kwargs) -> dict
ANALYZERS: Dict[str, Any] = {
    "summary": default_summary,
    "overlap-quality": _overlap_quality_wrapper,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    workspace: str,
    artifact_name: str,
    output_dir: str | Path | None = None,
    api_key: str | None = None,
    version_or_alias: str | None = None,
    analyzer_name: str = "summary",
    **analyze_kwargs: Any,
) -> Dict[str, Any]:
    """Download an artifact and run a named analyzer on it.

    Args:
        workspace: Comet workspace.
        artifact_name: Full artifact name.
        output_dir: Local download directory.
        api_key: Comet API key.
        version_or_alias: Artifact version or alias.
        analyzer_name: Which analyzer to run (``"summary"`` or
            ``"overlap-quality"``).  Defaults to ``"summary"``.
        **analyze_kwargs: Extra keyword arguments forwarded to the analyzer
            (e.g. ``betweenness_k=500``).

    Returns:
        Whatever the analyzer returns (typically a dict).
    """
    if analyzer_name not in ANALYZERS:
        available = ", ".join(sorted(ANALYZERS.keys()))
        raise ValueError(
            f"Unknown analyzer '{analyzer_name}'. Available: {available}"
        )

    analyze_fn = ANALYZERS[analyzer_name]

    print(f"Downloading artifact: {artifact_name} ...")
    artifact_dir = download_artifact(
        workspace=workspace,
        artifact_name=artifact_name,
        output_dir=output_dir,
        api_key=api_key,
        version_or_alias=version_or_alias,
    )
    print(f"Downloaded to: {artifact_dir}")

    payload = load_payload(artifact_dir)

    # Try loading pickle (v2); None if absent (v1 backward compat)
    clusterings = load_clusterings(artifact_dir)
    if clusterings is not None:
        print(f"Loaded {len(clusterings)} clusterings from pickle")
    else:
        print("No pickle file found — using JSON-only metadata (v1)")

    print()

    return analyze_fn(payload, clusterings=clusterings, **analyze_kwargs)
