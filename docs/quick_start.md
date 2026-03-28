# Quick Start Guide

This guide covers the fastest path to running benchmarks and generating plots.

## Benchmark Entry Points

- `main.py`: temporal edge-list and LFR `.gml` entry point.
- `main_static.py`: static graph entry point.
- `scripts/benchmark.sh`: dataset-config-driven wrapper around the entry points.
- `scripts/benchmark_static.sh`: dataset-config-driven wrapper for static graphs.

Use `main.py` for temporal/LFR runs and `main_static.py` for static runs.

If you need to stage a new local dataset first, see the [Dataset Preparation Guide](dataset_preparation.md).

## Run A Benchmark With `main.py`

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

What happens:

1. The dataset is converted into a `TemporalGraph`.
2. Target algorithms are loaded from the algorithm registry based on the names in [`config/algorithms.yaml`](../config/algorithms.yaml).
3. Each run is evaluated and logged to Comet ML.

## Useful CLI Options

```bash
python main.py --help
```

Common options:

| Argument | Description |
| --- | --- |
| `--dataset-path` | Input edge-list path |
| `--dataset` | Dataset label used in logs |
| `--source-idx` | Source column index |
| `--target-idx` | Target column index |
| `--delimiter` | File delimiter |
| `--batch-range` | Fraction of edges per temporal batch |
| `--initial-fraction` | Fraction of edges placed in the base graph |
| `--max-steps` | Maximum number of temporal updates |
| `--delete-insert-ratio` | Ratio of deletions to insertions for generated updates |
| `--num-runs` | Number of runs per algorithm |
| `--lfr-folder` | Folder of `snapshot_t*.gml` files |
| `--ground-truth-attr` | Ground-truth node attribute for LFR evaluation |

## Run From Dataset Config

List datasets from [`config/dynamic_dataset_config.yaml`](../config/dynamic_dataset_config.yaml):

```bash
./scripts/benchmark.sh --list
```

Run one configured dataset with the unified runner:

```bash
./scripts/benchmark.sh college-msg main.py
```

Run all configured target datasets:

```bash
./scripts/benchmark.sh --all main.py
```

Run the same benchmark multiple times:

```bash
./scripts/benchmark.sh college-msg main.py 3
```

Notes:

- `benchmark.sh` reads dataset values from [`config/dynamic_dataset_config.yaml`](../config/dynamic_dataset_config.yaml).
- If you need a custom `delete_insert_ratio`, call `main.py` directly.

## Run A Static Benchmark

Run a static graph directly from a file:

```bash
python main_static.py \
  --dataset-path ./data/karate.txt \
  --dataset karate \
  --source-idx 0 \
  --target-idx 1
```

Run a configured static dataset:

```bash
./scripts/benchmark_static.sh --list
./scripts/benchmark_static.sh karate 1
./scripts/benchmark_static.sh synthetic-n-5000-1 1
```

Use the config-defined preload fraction, or override it from the CLI:

```bash
python main_static.py --config karate
python main_static.py --config karate --preload-fraction 0.25
```

Run a built-in graph with ground truth:

```bash
python main_static.py --builtin karate --num-runs 1
```

Notes:

- Static graphs are loaded as `TemporalGraph(base_graph=G, steps=[])`.
- `main_static.py` automatically skips temporal algorithms such as `tiles`.
- `benchmark_static.sh` reads dataset values from [`config/static_dataset_config.yaml`](../config/static_dataset_config.yaml).
- `--preload-fraction` overrides the config value and benchmarks a partial static graph without creating temporal steps.
- When a static config entry has `type: lfr`, `main_static.py` loads a single labeled snapshot from the folder and benchmarks it as one static graph.

## Run On LFR `.gml` Snapshots

If you have temporal snapshots named `snapshot_t0.gml`, `snapshot_t1.gml`, and so on:

```bash
python main.py \
  --lfr-folder ./data/synthetic_n_5000_1 \
  --ground-truth-attr communities \
  --max-steps 10
```

The loader uses the first snapshot as the base graph, derives temporal changes from later snapshots, and precomputes ground-truth clusterings.

For static evaluation of the same dataset:

```bash
./scripts/benchmark_static.sh synthetic-n-5000-1 1
```

This loads one labeled snapshot from the configured folder and evaluates static algorithms only.

## Generate Plots

Recommended:

```bash
./scripts/plot.sh
```

To process only one benchmark mode:

```bash
./scripts/plot.sh dynamic
./scripts/plot.sh static
```

Manual equivalent:

```bash
PYTHONPATH=. python tools/fetch_and_merge.py --benchmark-type all
PYTHONPATH=. python tools/plots.py --benchmark-type all
```

The plot pipeline reads [`config/visualization_dynamic.yaml`](../config/visualization_dynamic.yaml) and [`config/visualization_static.yaml`](../config/visualization_static.yaml), fetches Comet runs into `experiments/dynamic/raw/` and `experiments/static/raw/`, merges them into `experiments/dynamic/merged/` and `experiments/static/merged/`, and writes figures to `assets/dynamic/` and `assets/static/`.

## Analyze Clustering Artifacts

After benchmark runs, clustering results are stored as Comet ML artifacts. You can download and analyze them offline:

```bash
# List what's available (basic summary)
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg

# Run overlap quality analysis
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg \
    --analyzer overlap-quality

# With approximate betweenness and JSON export
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg \
    --analyzer overlap-quality --betweenness-k 500 --save-json report.json

# Analyze all dynamic artifacts in one command (YAML-driven)
python tools/analyze.py --config config/analyzer.yaml --all-artifacts --benchmark-mode dynamic
```

The overlap quality analyzer prints a human-readable report and optionally saves it as JSON. It works for both static and dynamic benchmark artifacts.

You can plot saved analyzer reports directly:

```bash
python tools/plot_analysis.py --input report.json --plot all
```

See [Post-Hoc Analysis](analysis.md) for the full reference.

## Typical Workflow

```bash
cp .env.example .env
python main.py --dataset-path ./data/CollegeMsg.txt --dataset CollegeMsg --max-steps 10
python main_static.py --builtin karate --num-runs 1
./scripts/plot.sh
python tools/analyze.py --workspace my-ws --artifact clustering-coach-CollegeMsg --analyzer overlap-quality
```

## Common Issues

### Dataset not found

- Verify the file exists under `data/`.
- Verify the path and dataset key in [`config/dynamic_dataset_config.yaml`](../config/dynamic_dataset_config.yaml) or [`config/static_dataset_config.yaml`](../config/static_dataset_config.yaml).

### Comet credentials missing

- Add `COMET_API_KEY` and `COMET_WORKSPACE` to `.env`.

### `src` import errors in tools

- Run tools with `PYTHONPATH=.` or use `./scripts/plot.sh`.

## Next Steps

- See [Configuration Guide](configuration.md) to change target datasets and algorithms.
- See [Development Guide](reference/development_guide.md) to integrate your own community detection method.
- See [Metrics Documentation](reference/metrics.md) for the meaning of the logged values.
- See [Visualization Guide](visualization.md) for plot details.
- See [Post-Hoc Analysis](analysis.md) for the overlap quality analyzer reference.
