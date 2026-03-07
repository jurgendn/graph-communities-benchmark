# Graph Communities Benchmark

A benchmarking framework for temporal graph community detection. The project evaluates both overlapping and crisp methods on evolving networks, logs runs to Comet ML, and provides a plotting pipeline for cross-dataset comparisons.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What It Does

- Build temporal snapshots from edge-list datasets.
- Run static algorithms snapshot-by-snapshot and dynamic algorithms on the full temporal graph.
- Evaluate crisp and overlapping outputs with the appropriate modularity metric.
- Optionally evaluate against ground truth on LFR-style GraphML snapshots.
- Log per-run and per-step metrics to Comet ML.
- Fetch, merge, and plot experiment results into grouped comparison figures.

## Documentation

| Document | Description |
| --- | --- |
| [Installation Guide](docs/installation.md) | Environment setup and dependencies |
| [Quick Start](docs/quick_start.md) | Common commands for benchmarks and plots |
| [Configuration Guide](docs/configuration.md) | YAML config files and key fields |
| [Metrics Documentation](docs/metrics.md) | Logged metrics and ground-truth evaluation |
| [Visualization Guide](docs/visualization.md) | Fetching Comet runs and generating plots |
| [Development Guide](docs/development_guide.md) | Adding algorithms, datasets, and metrics |

## Supported Modes

### Static algorithms

Configured in [`config/algorithms.yaml`](config/algorithms.yaml) and executed once per snapshot through the unified wrapper layer.

Common configured examples:

- `coach`
- `graph_entropy`
- `core_expansion`
- `angel`
- `slpa`
- `percomvc`
- `big_clam`
- `demon`

### Dynamic algorithms

Algorithms that consume the full `TemporalGraph`:

- `tiles`
- `df_louvain`

### Clustering types

- `crisp`: one community per node; evaluated with Newman-Girvan modularity.
- `overlapping`: multiple communities per node; evaluated with CDlib overlap modularity plus custom Q0 modularity.

The mode is controlled by each algorithm's `type` and `clustering_type` in [`config/algorithms.yaml`](config/algorithms.yaml).

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

### 3. Run a benchmark

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

Or use the dataset config wrapper:

```bash
./scripts/benchmark.sh --list
./scripts/benchmark.sh college-msg main.py
```

### 4. Generate plots

```bash
./scripts/plot.sh
```

See [Quick Start](docs/quick_start.md) for more examples, including LFR input.

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

Show the full help with:

```bash
python main.py --help
```

## Project Layout

```text
graph-communities-benchmark/
|- main.py
|- main_static.py
|- main_dynamic.py
|- config/
|  |- algorithms.yaml
|  |- dataset_config.yaml
|  `- visualization.yaml
|- docs/
|- scripts/
|  |- benchmark.sh
|  `- plot.sh
|- tools/
|  |- fetch_and_merge.py
|  `- plots.py
`- src/
   |- algorithms/
   |- dataloader/
   |- evaluations/
   |- factory/
   |- models/
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

Details are in [docs/metrics.md](docs/metrics.md).

## Data And Outputs

- Raw datasets are expected under `data/` by default.
- Raw Comet exports are written under `experiments/raw/`.
- Merged plot-ready data is written under `experiments/merged/`.
- Generated figures are written under `assets/grouped/`.

Note: `data/`, `experiments/`, and `assets/` are gitignored in this repository.

## Notes

- The unified path is `main.py`; `main_static.py` and `main_dynamic.py` remain for compatibility.
- Some CDlib algorithms may require optional third-party packages beyond `requirements.txt`.
- The `.env.example` file includes `COMET_PROJECT_NAME`, but current benchmark runs derive the Comet project name from the dataset automatically.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

## Contributing

Contributions are welcome. Start with [docs/development_guide.md](docs/development_guide.md).
