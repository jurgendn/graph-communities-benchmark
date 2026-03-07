# Quick Start Guide

This guide covers the fastest path to running benchmarks and generating plots.

## Benchmark Entry Points

- `main.py`: unified entry point for static and dynamic algorithms.
- `main_static.py`: legacy static runner.
- `main_dynamic.py`: legacy dynamic runner.
- `scripts/benchmark.sh`: dataset-config-driven wrapper around the entry points.

For new work, use `main.py` unless you specifically need the legacy runners.

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
2. Target algorithms are loaded from [`config/algorithms.yaml`](../config/algorithms.yaml).
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
| `--lfr-folder` | Folder of `snapshot_t*.graphml` files |
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
- The script only forwards `delete_insert_ratio` automatically for `main_dynamic.py`; if you need a custom value with `main.py`, call `main.py` directly.

## Run On LFR GraphML Snapshots

If you have temporal GraphML snapshots named `snapshot_t0.graphml`, `snapshot_t1.graphml`, and so on:

```bash
python main.py \
  --lfr-folder ./data/lfr_benchmark/my_lfr \
  --ground-truth-attr label \
  --max-steps 10
```

The loader uses the first snapshot as the base graph and derives temporal changes from the remaining snapshots.

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
