# Quick Start Guide

This guide covers the fastest path to running benchmarks and generating plots.

## Benchmark Entry Points

- `main.py`: temporal edge-list and LFR `.gml` entry point.
- `main_static.py`: static graph entry point.
- `scripts/benchmark.sh`: dataset-config-driven wrapper around the entry points.
- `scripts/benchmark_static.sh`: dataset-config-driven wrapper for static graphs.

Use `main.py` for temporal/LFR runs and `main_static.py` for static runs.

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
2. Snapshot and temporal target algorithms are loaded from [`config/algorithms.yaml`](../config/algorithms.yaml).
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

List datasets from [`config/dataset_config.yaml`](../config/dataset_config.yaml):

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

- `benchmark.sh` reads dataset values from [`config/dataset_config.yaml`](../config/dataset_config.yaml).
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
```

Run a built-in graph with ground truth:

```bash
python main_static.py --builtin karate --num-runs 1
```

Notes:

- Static graphs are loaded as `TemporalGraph(base_graph=G, steps=[])`.
- `main_static.py` automatically skips temporal algorithms such as `tiles`.
- `--preload-fraction` benchmarks a partial static graph without creating temporal steps.

## Run On LFR `.gml` Snapshots

If you have temporal snapshots named `snapshot_t0.gml`, `snapshot_t1.gml`, and so on:

```bash
python main.py \
  --lfr-folder ./data/synthetic_n_5000_1 \
  --ground-truth-attr communities \
  --max-steps 10
```

The loader uses the first snapshot as the base graph, derives temporal changes from later snapshots, and precomputes ground-truth clusterings.

## Generate Plots

Recommended:

```bash
./scripts/plot.sh
```

Manual equivalent:

```bash
PYTHONPATH=. python tools/fetch_and_merge.py
PYTHONPATH=. python tools/plots.py
```

The plot pipeline reads [`config/visualization.yaml`](../config/visualization.yaml), fetches Comet runs into `experiments/raw/`, merges them into `experiments/merged/`, and writes figures to `assets/grouped/`.

## Typical Workflow

```bash
cp .env.example .env
python main.py --dataset-path ./data/CollegeMsg.txt --dataset CollegeMsg --max-steps 10
python main_static.py --builtin karate --num-runs 1
./scripts/plot.sh
```

## Common Issues

### Dataset not found

- Verify the file exists under `data/`.
- Verify the path and dataset key in [`config/dataset_config.yaml`](../config/dataset_config.yaml).

### Comet credentials missing

- Add `COMET_API_KEY` and `COMET_WORKSPACE` to `.env`.

### `src` import errors in tools

- Run tools with `PYTHONPATH=.` or use `./scripts/plot.sh`.

## Next Steps

- See [Configuration Guide](configuration.md) to change target datasets and algorithms.
- See [Metrics Documentation](metrics.md) for the meaning of the logged values.
- See [Visualization Guide](visualization.md) for plot details.
