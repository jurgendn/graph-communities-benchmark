# Graph Communities Benchmark

A benchmarking framework for dynamic and static community detection. The project evaluates overlapping and crisp methods on temporal graphs, static graphs, and labeled LFR snapshots, logs runs to Comet ML, and provides a plotting pipeline for cross-dataset comparisons.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What It Does

- Build temporal snapshots from edge-list datasets.
- Treat static graphs as one-snapshot `TemporalGraph` objects.
- Run static algorithms snapshot-by-snapshot and dynamic algorithms on the full temporal graph.
- Evaluate crisp and overlapping outputs with the appropriate modularity metric.
- Optionally evaluate against ground truth on LFR-style `.gml` snapshots and supported static graphs.
- Log per-run and per-step metrics to Comet ML.
- Log full clustering results as Comet ML artifacts (pickle + JSON metadata) for post-hoc analysis.
- Analyze overlap quality offline: structural metrics, statistical tests, accuracy, and temporal stability.
- Fetch, merge, and plot experiment results into grouped comparison figures.

## Documentation

Start with the [Documentation Index](docs/README.md).

| Document | Description |
| --- | --- |
| [Documentation Index](docs/README.md) | Organized navigation for guides and reference docs |
| [Quick Start](docs/quick_start.md) | Common commands for benchmarks and plots |
| [Dataset Preparation Guide](docs/dataset_preparation.md) | Supported file formats and config examples |
| [Configuration Guide](docs/configuration.md) | YAML config files and key fields |
| [Metrics Documentation](docs/reference/metrics.md) | Logged metrics and ground-truth evaluation |
| [Visualization Guide](docs/visualization.md) | Fetching Comet runs and generating plots |
| [Post-Hoc Analysis](docs/analysis.md) | Overlap quality analyzer and artifact format |
| [Development Guide](docs/reference/development_guide.md) | Adding algorithms, datasets, and metrics |

## Algorithm Layout

Algorithms are registered via `@register` decorators in their source files. The decorator declares identity and metadata (name, type, clustering type, default params, description). [`config/algorithms.yaml`](config/algorithms.yaml) is a minimal run config that selects which registered algorithms to execute and optionally overrides default parameters.

### Snapshot algorithms

Registered with `algo_type="static"` and executed once per snapshot through the wrapper layer. These can run in both the temporal benchmark and the static benchmark.

Available examples include: `coach`, `graph_entropy`, `core_expansion`, `angel`, `slpa`, `percomvc`, `big_clam`, `demon`, `copra`, `vast-pmo`, `vast-cosine-overlap`, `ndocd`, `dpclus`, `ipca`, `dcs`, `lfm`, `ebgc`, `lais2`, `walkscan`, `umstmo`.

### Temporal algorithms

Registered with `algo_type="dynamic"` and consume the full `TemporalGraph`:

- `tiles`
- `df_louvain`

### Clustering types

- `crisp`: one community per node; evaluated with Newman-Girvan modularity.
- `overlapping`: multiple communities per node; evaluated with CDlib overlap modularity plus custom Q0 modularity.

The execution style is controlled by which target list an algorithm appears in (`target_snapshot_algorithms` or `target_temporal_algorithms`) in [`config/algorithms.yaml`](config/algorithms.yaml). The `clustering_type` comes from the `@register` decorator.

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Comet ML

```bash
cp .env.example .env
```

Set at least:

```ini
COMET_API_KEY=your_api_key_here
COMET_WORKSPACE=your_workspace_here
```

### 3. Run a dynamic benchmark

If you are bringing your own input files, see [Dataset Preparation Guide](docs/dataset_preparation.md) for the expected edge-list and LFR layouts.

```bash
python main.py \
  --dataset-path ./data/CollegeMsg.txt \
  --dataset CollegeMsg \
  --source-idx 0 \
  --target-idx 1 \
  --batch-range 1e-4 \
  --initial-fraction 0.4 \
  --max-steps 10
```

Or use the dataset-config wrapper:

```bash
./scripts/benchmark.sh --list
./scripts/benchmark.sh college-msg main.py
```

### 4. Run a static benchmark

From a file:

```bash
python main_static.py \
  --dataset-path ./data/karate.txt \
  --dataset karate \
  --source-idx 0 \
  --target-idx 1
```

From static dataset config:

```bash
./scripts/benchmark_static.sh --list
./scripts/benchmark_static.sh karate 1
./scripts/benchmark_static.sh synthetic-n-5000-1 1
```

From a built-in graph with ground truth:

```bash
python main_static.py --builtin karate --num-runs 1
```

### 5. Generate plots

```bash
./scripts/plot.sh
```

Or target one benchmark mode:

```bash
./scripts/plot.sh dynamic
./scripts/plot.sh static
```

Manual equivalents:

```bash
PYTHONPATH=. python tools/fetch_and_merge.py --benchmark-type dynamic
PYTHONPATH=. python tools/plots.py --benchmark-type dynamic
```

See [Quick Start](docs/quick_start.md) for more examples, including LFR input and static datasets from config.

## Post-Hoc Overlap Analysis

Every benchmark run logs a Comet ML artifact containing the full `NodeClustering` objects (with graphs) as a pickle file alongside JSON metadata. These artifacts can be downloaded and analyzed offline.

### Run the analyzer

```bash
# Basic summary
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg

# Overlap quality analysis (structural metrics + statistical tests)
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg \
    --analyzer overlap-quality

# With approximate betweenness and JSON export
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg \
    --analyzer overlap-quality --betweenness-k 500 --save-json report.json

# Analyze all dynamic artifacts at once (YAML defaults + filtering)
python tools/analyze.py --config config/analyzer.yaml --all-artifacts --benchmark-mode dynamic
```

The overlap quality analyzer computes per-snapshot:

- **Participation coefficient** — fraction of a node's neighbors in other communities
- **Max embeddedness** — max fraction of neighbors in any single community
- **Betweenness centrality** — bridge position in the graph
- **Mann-Whitney U tests** — statistical comparison of overlap vs non-overlap node groups
- **Temporal stability** — ONMI between consecutive snapshots (multi-snapshot runs)

Plot saved analysis reports:

```bash
python tools/plot_analysis.py --input report.json --plot all
```

See [Post-Hoc Analysis](docs/analysis.md) for the full reference.

## CLI Highlights

`main.py` accepts these commonly used options:

- `--dataset-path`
- `--dataset`
- `--source-idx`
- `--target-idx`
- `--batch-range`
- `--initial-fraction`
- `--max-steps`
- `--delete-insert-ratio`
- `--num-runs`
- `--lfr-folder`
- `--ground-truth-attr`

`main_static.py` accepts these commonly used options:

- `--dataset-path`
- `--dataset`
- `--config`
- `--builtin`
- `--source-idx`
- `--target-idx`
- `--delimiter`
- `--preload-fraction`
- `--ground-truth-attr`
- `--num-runs`
- `--list-datasets`
- `--list-builtins`

Show the full help with:

```bash
python main.py --help
```

## Project Layout

```text
graph-communities-benchmark/
|- main.py
|- main_static.py
|- config/
|  |- algorithms.yaml
|  |- analyzer.yaml
|  |- dynamic_dataset_config.yaml
|  |- static_dataset_config.yaml
|  |- visualization_dynamic.yaml
|  `- visualization_static.yaml
|- docs/
|- scripts/
|  |- benchmark.sh
|  `- plot.sh
|- templates/
|  |- static_algorithm_template.py
|  `- dynamic_algorithm_template.py
|- tools/
|  |- analyze.py
|  |- fetch_and_merge.py
|  |- plot_analysis.py
|  `- plots.py
`- src/
   |- algorithms/
   |  |- registry.py
   |  |- cdlib_adapters.py
   |  |- factory.py
   |  |- base.py
   |  `- wrappers.py
   |- analyzer/
   |  |- artifacts.py
   |  |- models.py
   |  |- runner.py
   |  |- overlap_quality.py
   |  |- metrics_structural.py
   |  |- metrics_accuracy.py
   |  `- stats.py
   |- core/
   |  |- pipeline.py
   |  |- results.py
   |  `- temporal_graph.py
   |- dataloader/
   |- evaluations/
   |- factory/
   |- models/
   |  |- common/
   |  |- dynamic/
   |  `- static/
   |- utils/
   `- visualization/
```

## Metrics Logged

Per run, the pipeline logs summary and per-step metrics to Comet ML, including:

- runtime
- `cdlib_modularity`
- `customize_q0_modularity`
- `num_communities`
- `nmi` when ground truth is available

Details are in [docs/reference/metrics.md](docs/reference/metrics.md).

In addition to metrics, each run logs a **clustering artifact** to Comet ML containing the full `List[NodeClustering]` objects (with graphs attached) as `clusterings.pkl`, plus lightweight JSON metadata as `clustering_payload.json`. These artifacts enable post-hoc analysis without re-running benchmarks.

## Data And Outputs

- Raw datasets are expected under `data/` by default.
- See [docs/dataset_preparation.md](docs/dataset_preparation.md) for supported dataset formats and config examples.
- Raw Comet exports are written under `experiments/dynamic/raw/` and `experiments/static/raw/`.
- Merged plot-ready data is written under `experiments/dynamic/merged/` and `experiments/static/merged/`.
- Generated figures are written under `assets/dynamic/` and `assets/static/`.
- Plot settings live in `config/visualization_dynamic.yaml` and `config/visualization_static.yaml`.

Note: `data/`, `experiments/`, and `assets/` are gitignored in this repository.

## Notes

- `main.py` is the temporal and LFR entry point; it can run both snapshot and temporal algorithms.
- `main_static.py` is the static entry point and can also load one labeled LFR snapshot from a configured `type: lfr` folder.
- Static graphs reuse the same evaluation and logging pipeline by loading as `TemporalGraph(base_graph=G, steps=[])`.
- In static mode, configured LFR datasets load `snapshot_t0.gml` when present; otherwise the earliest `snapshot_t*.gml` file is used as the single benchmark snapshot.
- Comet projects are now mode-specific, using `graph-community-detection-dynamic-*` and `graph-community-detection-static-*` names.
- Shared model code lives under `src/models/common/`; algorithm implementations live under `src/models/static/` and `src/models/dynamic/`.
- Some CDlib algorithms may require optional third-party packages beyond `requirements.txt`.
- The `.env.example` file includes `COMET_PROJECT_NAME`, but current benchmark runs derive the Comet project name from the dataset automatically.

## Synthetic LFR Naming

For generated LFR datasets, prefer short keys that expose the main structural parameters:

- Recommended pattern: `synthetic-n-<nodes>-k<avg_degree>-mu<mixing>-c<min>-<max>`
- Example: `synthetic-n-10000-k4-mu0.1-c50-200`
- Use the hyphenated key in YAML config and shell commands; use the underscore version for folder names when needed.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

## Contributing

Contributions are welcome. For algorithm integration and general development guidance, see [docs/reference/development_guide.md](docs/reference/development_guide.md).
