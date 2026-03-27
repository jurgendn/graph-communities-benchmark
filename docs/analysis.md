# Post-Hoc Analysis

The analyzer package (`src/analyzer/`) provides offline analysis of clustering artifacts logged during benchmark runs. Artifacts are self-contained: each includes the full `NodeClustering` objects with graphs attached, so analysis does not require re-running benchmarks or loading datasets from config files.

## Artifact Format

Each benchmark run logs a Comet ML artifact named `clustering-{algorithm}-{dataset}` containing two files:

| File | Format | Contents |
|------|--------|----------|
| `clusterings.pkl` | pickle | `List[NodeClustering]` — one per snapshot, with `.graph`, `.communities`, `.method_name`, `.method_parameters`, `.overlap` |
| `clustering_payload.json` | JSON | Lightweight metadata for filtering — algorithm name, dataset, benchmark mode, per-snapshot community counts |

The pickle file stores the exact snapshot graphs used during the benchmark. For dynamic benchmarks this is important because graph reconstruction is non-deterministic (random edge deletions).

## CLI Usage

The CLI tool is `tools/analyze.py`:

```bash
python tools/analyze.py --workspace <workspace> --artifact <artifact-name> [options]
```

### Options

| Flag | Description |
|------|-------------|
| `--workspace` | Comet ML workspace name (required) |
| `--artifact` | Artifact name, e.g. `clustering-coach-CollegeMsg` (required) |
| `--output-dir` | Local directory to download artifact into (default: temp dir) |
| `--version` | Artifact version or alias (default: latest) |
| `--analyzer` | Which analyzer to run: `summary` or `overlap-quality` (default: `summary`) |
| `--betweenness-k` | Sample size for approximate betweenness centrality (default: exact) |
| `--save-json` | Save analysis result as JSON to this path |

### Examples

```bash
# Basic summary (default analyzer)
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg

# Overlap quality analysis
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg \
    --analyzer overlap-quality

# With approximate betweenness and JSON export
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg \
    --analyzer overlap-quality --betweenness-k 500 --save-json report.json
```

## Available Analyzers

### `summary`

Prints basic metadata from the artifact payload:

- Algorithm, dataset, benchmark mode, clustering type
- Number of snapshots
- Average runtime, modularity, NMI (if available)
- Per-snapshot community counts
- Whether pickle data is available

### `overlap-quality`

Validates whether overlapping nodes occupy structurally meaningful boundary positions. Computes the following per snapshot (when the graph is available from pickle):

**Structural metrics (per node):**

| Metric | Description |
|--------|-------------|
| Participation coefficient | Fraction of a node's neighbors that belong to communities other than its own. Higher values indicate boundary nodes. |
| Max embeddedness | Maximum fraction of a node's neighbors in any single community. Lower values indicate boundary nodes. |
| Betweenness centrality | Standard betweenness centrality. Higher values indicate bridge positions. |

**Statistical testing:**

For each structural metric, nodes are split into two groups:
- **Overlap nodes**: appear in 2 or more communities
- **Non-overlap nodes**: appear in exactly 1 community

A **Mann-Whitney U test** compares the distributions. The report includes:
- Sample sizes, medians, and means for each group
- U statistic, p-value, and rank-biserial correlation (effect size)

If overlapping nodes are structurally meaningful, you would expect:
- Higher participation coefficient for overlap nodes (more cross-community neighbors)
- Lower max embeddedness for overlap nodes (less embedded in any single community)
- Higher betweenness centrality for overlap nodes (more central/bridge-like)

**Accuracy metric:**

| Metric | Description |
|--------|-------------|
| Omega index | Agreement between detected and ground-truth community assignments. Only computed when ground truth is provided. |

**Temporal stability:**

| Metric | Description |
|--------|-------------|
| ONMI (consecutive) | Overlapping NMI between community assignments at snapshot t-1 and t. Measures how stable community structure is across temporal steps. |

**Membership summary (always available):**

- Per-snapshot community counts
- Per-snapshot overlap node counts

## Report Format

The overlap quality analyzer prints a human-readable text report:

```
======================================================================
OVERLAP QUALITY REPORT
======================================================================
Algorithm:       coach
Dataset:         CollegeMsg
Mode:            dynamic
Clustering type: overlapping
Snapshots:       10

--- Membership summary ---
Communities/snapshot: [12, 14, 13, ...]
Overlap nodes/snapshot: [45, 52, 48, ...]

--- Snapshot 0 ---
  Nodes: 1899  Edges: 13838  Communities: 12
  Overlapping nodes: 45 (2.4%)
    participation_coefficient:
      overlap    (n=   45): median=0.7231  mean=0.6984
      non-overlap(n= 1854): median=0.4521  mean=0.4102
      U=55231.0  p=1.23e-04  r_rb=-0.3241
    max_embeddedness:
      ...
    betweenness_centrality:
      ...

--- Temporal stability (ONMI consecutive) ---
  t0 -> t1: ONMI = 0.8234
  t1 -> t2: ONMI = 0.7891
  ...
======================================================================
```

When `--save-json` is used, the full report is exported as a JSON object matching the `OverlapQualityReport` Pydantic model.

## Backward Compatibility

Artifacts logged before the pickle format was introduced (v1, JSON-only) are still supported:

- The `summary` analyzer works normally using JSON metadata.
- The `overlap-quality` analyzer falls back to metadata-only analysis: community counts are reported from JSON, but structural metrics and temporal stability are unavailable because they require the full `NodeClustering` objects.

## Architecture

```
tools/analyze.py          CLI entrypoint
    └── src/analyzer/
        ├── runner.py             Orchestration: download + dispatch
        ├── artifacts.py          Enrich, serialize, upload/download artifacts
        ├── models.py             Pydantic models (payloads + reports)
        ├── overlap_quality.py    Main overlap quality analyzer
        ├── metrics_structural.py Participation coefficient, embeddedness, betweenness
        ├── metrics_accuracy.py   Omega index, ONMI consecutive
        └── stats.py              Mann-Whitney U with rank-biserial
```

The pipeline side (`src/core/pipeline.py`) calls `enrich_clusterings()` during `log_results()` to fill in any missing fields on `NodeClustering` objects before pickling. CDlib adapter algorithms (angel, demon, coach, etc.) already set most fields correctly; the enrichment step is primarily needed for custom algorithms like VastPMO that construct `NodeClustering` with `graph=None`.

## Adding A New Analyzer

1. Implement a function in `src/analyzer/` with the signature:

   ```python
   def my_analyzer(
       payload: ClusteringArtifactPayload,
       clusterings: Optional[List[NodeClustering]] = None,
       **kwargs,
   ) -> dict:
       ...
   ```

2. Register it in `ANALYZERS` in `src/analyzer/runner.py`:

   ```python
   ANALYZERS = {
       "summary": default_summary,
       "overlap-quality": _overlap_quality_wrapper,
       "my-analyzer": my_analyzer,
   }
   ```

3. The CLI picks it up automatically. Users can select it with `--analyzer my-analyzer`.

## Prerequisites

- `COMET_API_KEY` and `COMET_WORKSPACE` must be set in `.env` (for downloading artifacts).
- All dependencies are in `requirements.txt` (cdlib, scipy, pydantic, comet_ml, networkx, numpy).
