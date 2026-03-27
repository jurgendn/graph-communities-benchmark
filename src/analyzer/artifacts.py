"""Serialize NodeClustering results to/from Comet ML artifacts.

**Artifact format (v2)**:

Each Comet Artifact contains TWO files:

- ``clusterings.pkl`` — ``List[NodeClustering]`` with graphs attached
- ``clustering_payload.json`` — lightweight metadata (no communities/graph)

Upload side — called from ``src.core.pipeline.log_results``:
    enrich_clusterings()  → fills graph/method_name/method_parameters/overlap
    build_payload()       → ClusteringArtifactPayload (metadata only)
    log_artifact()        → uploads pickle + JSON as a Comet Artifact

Download side — called from ``src.analyzer.runner``:
    download_artifact()   → downloads artifact to local dir
    load_payload()        → ClusteringArtifactPayload from local JSON
    load_clusterings()    → List[NodeClustering] from local pickle
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from cdlib import NodeClustering
from comet_ml import API, Artifact, Experiment

from src.analyzer.models import ClusteringArtifactPayload, SnapshotClustering
from src.core.results import MethodDynamicResults
from src.core.temporal_graph import TemporalGraph


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_clusterings(
    clusterings: List[NodeClustering],
    tg: TemporalGraph,
    algo_name: str,
    algo_params: dict,
    clustering_type: str,
    dataset_config: Dict[str, Any],
) -> None:
    """Fill in missing fields on each NodeClustering **in-place**.

    Ensures every NC has ``graph``, ``method_name``, ``method_parameters``,
    and ``overlap`` set before pickling.

    Args:
        clusterings: The list produced by ``run_algorithm()``.
        tg: TemporalGraph whose snapshots supply graphs for NCs missing one.
        algo_name: Algorithm name (e.g. ``"coach"``).
        algo_params: Algorithm hyperparameters dict.
        clustering_type: ``"crisp"`` or ``"overlapping"``.
        dataset_config: Dataset metadata dict to embed in ``method_parameters``.
    """
    snapshots = list(tg.iter_snapshots())

    for idx, nc in enumerate(clusterings):
        # Graph — fill from snapshot if missing (e.g. VastPMO)
        if nc.graph is None and idx < len(snapshots):
            nc.graph = snapshots[idx]

        # Method name — prefer existing (cdlib adapters set this)
        if not getattr(nc, "method_name", None):
            nc.method_name = algo_name

        # Method parameters — merge algo params + dataset config
        if not getattr(nc, "method_parameters", None):
            nc.method_parameters = {
                **algo_params,
                "dataset_config": dataset_config,
            }

        # Overlap flag
        nc.overlap = clustering_type == "overlapping"


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def build_payload(
    results: MethodDynamicResults,
    algo_name: str,
    algo_type: str,
    clustering_type: str,
    algo_params: dict,
    dataset: str,
    benchmark_mode: str,
    dataset_config: Dict[str, Any] | None = None,
) -> ClusteringArtifactPayload:
    """Convert benchmark results into a JSON-serialisable metadata payload.

    v2: No communities stored in JSON — they live in the pickle file.
    """
    snapshots: List[SnapshotClustering] = []
    for idx, nc in enumerate(results.clusterings):
        snapshots.append(
            SnapshotClustering(
                snapshot_index=idx,
                num_communities=len(nc.communities),
                overlap=getattr(nc, "overlap", False),
            )
        )

    return ClusteringArtifactPayload(
        algorithm=algo_name,
        algorithm_type=algo_type,
        clustering_type=clustering_type,
        algorithm_params=algo_params,
        dataset_config=dataset_config or {},
        dataset=dataset,
        benchmark_mode=benchmark_mode,
        num_snapshots=len(snapshots),
        snapshots=snapshots,
        avg_runtime=float(results.avg_runtime) if results.runtimes else None,
        avg_modularity=(
            float(results.avg_cdlib_modularity_overlap)
            if results.cdlib_modularity_overlap_trace
            else None
        ),
        avg_nmi=float(results.avg_nmi) if results.nmi_trace else None,
    )


def log_artifact(
    experiment: Experiment,
    payload: ClusteringArtifactPayload,
    clusterings: List[NodeClustering],
) -> None:
    """Write pickle + JSON to temp files and log them as a Comet Artifact.

    Artifact naming: ``clustering-{algorithm}-{dataset}``
    """
    artifact_name = f"clustering-{payload.algorithm}-{payload.dataset}"
    artifact = Artifact(
        name=artifact_name,
        artifact_type="clustering-result",
        metadata={
            "algorithm": payload.algorithm,
            "dataset": payload.dataset,
            "benchmark_mode": payload.benchmark_mode,
            "num_snapshots": payload.num_snapshots,
        },
    )

    tmp_json = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="clustering_", delete=False
    )
    tmp_pkl = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pkl", prefix="clusterings_", delete=False
    )
    try:
        # JSON metadata
        tmp_json.write(payload.model_dump_json(indent=2))
        tmp_json.close()
        artifact.add(tmp_json.name, logical_path="clustering_payload.json")

        # Pickle — full List[NodeClustering] with graphs
        pickle.dump(clusterings, tmp_pkl)
        tmp_pkl.close()
        artifact.add(tmp_pkl.name, logical_path="clusterings.pkl")

        experiment.log_artifact(artifact)
    finally:
        for path in (tmp_json.name, tmp_pkl.name):
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_artifact(
    workspace: str,
    artifact_name: str,
    output_dir: str | Path | None = None,
    api_key: str | None = None,
    version_or_alias: str | None = None,
) -> Path:
    """Download a clustering artifact from Comet ML and return the local path.

    Args:
        workspace: Comet workspace name.
        artifact_name: Artifact name (e.g. ``clustering-coach-CollegeMsg``).
        output_dir: Where to download.  Defaults to a temp directory.
        api_key: Comet API key (falls back to ``COMET_API_KEY`` env var).
        version_or_alias: Optional version or alias string.

    Returns:
        Path to the directory containing the downloaded artifact files.
    """
    api_key = api_key or os.getenv("COMET_API_KEY")
    if not api_key:
        raise ValueError("COMET_API_KEY not set")

    full_name = f"{workspace}/{artifact_name}"
    if version_or_alias:
        full_name = f"{full_name}:{version_or_alias}"

    # Use an online experiment to download via the SDK helper
    experiment = Experiment(api_key=api_key, auto_output_logging="false")
    try:
        logged = experiment.get_artifact(full_name)
        dest = str(output_dir) if output_dir else None
        downloaded = logged.download(path=dest, overwrite_strategy="OVERWRITE")
        local_path = Path(str(downloaded.download_local_path))
    finally:
        experiment.end()

    return local_path


def load_payload(artifact_dir: Path) -> ClusteringArtifactPayload:
    """Load a ``ClusteringArtifactPayload`` from a downloaded artifact directory."""
    payload_file = artifact_dir / "clustering_payload.json"
    if not payload_file.exists():
        # Try finding any JSON file in the directory
        json_files = list(artifact_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No clustering payload found in {artifact_dir}"
            )
        payload_file = json_files[0]

    with open(payload_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return ClusteringArtifactPayload.model_validate(data)


def load_clusterings(artifact_dir: Path) -> Optional[List[NodeClustering]]:
    """Load ``List[NodeClustering]`` from the pickle file.

    Returns ``None`` if the pickle file doesn't exist (backward compat
    with v1 JSON-only artifacts).
    """
    pkl_file = artifact_dir / "clusterings.pkl"
    if not pkl_file.exists():
        return None

    with open(pkl_file, "rb") as fh:
        clusterings = pickle.load(fh)

    return clusterings


def list_artifacts(
    workspace: str,
    api_key: str | None = None,
    artifact_type: str = "clustering-result",
) -> List[Dict[str, Any]]:
    """List all clustering artifacts in a workspace.

    Returns a list of dicts with artifact metadata.
    """
    api_key = api_key or os.getenv("COMET_API_KEY")
    if not api_key:
        raise ValueError("COMET_API_KEY not set")

    api = API(api_key=api_key)
    result = api.get_artifact_list(workspace, artifact_type=artifact_type)

    artifacts = result.get("artifacts", []) if isinstance(result, dict) else []
    return artifacts
