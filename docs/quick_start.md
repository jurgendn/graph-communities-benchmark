# Quick Start Guide

This guide helps you get started with running benchmarks using the Graph Communities Benchmark framework.

## Overview

The benchmark framework provides two ways to run experiments:

1. **Configuration-based scripts**: Read from YAML configuration files (recommended)
2. **Direct command-line**: Pass parameters directly to Python scripts

## Method 1: Using Configuration-Based Scripts (Recommended)

The benchmark provides configuration-based scripts that read from [`config/dataset_config.yaml`](../config/dataset_config.yaml) and [`config/algorithms.yaml`](../config/algorithms.yaml).

### List Available Datasets

```bash
./scripts/benchmark.sh --list
```

This will display all datasets configured in [`config/dataset_config.yaml`](../config/dataset_config.yaml).

### Run Benchmarks for All Target Datasets

```bash
# Run static benchmarks
./scripts/benchmark.sh --all main_static.py

# Run dynamic benchmarks
./scripts/benchmark.sh --all main_dynamic.py
```

### Run Benchmarks for a Specific Dataset

```bash
./scripts/benchmark.sh college-msg main_static.py
./scripts/benchmark.sh bio-wormnet-v3 main_static.py
```

### Run Benchmarks with Multiple Runs

```bash
# Run 5 times for each dataset
./scripts/benchmark.sh college-msg main_static.py 5

# Run 3 times for all datasets
./scripts/benchmark.sh --all main_static.py 3
```

### Script Options

| Option | Description |
|--------|-------------|
| `--all` | Run benchmarks for all target datasets |
| `--list` | List all available datasets |
| `<dataset_name>` | Run benchmarks for a specific dataset |
| `<main_script>` | Entry point script (`main_static.py` or `main_dynamic.py`) |
| `<num_runs>` | Number of times to run the benchmark (default: 1) |

## Method 2: Direct Command-Line Execution

### Running Static Benchmarks

Run overlapping community detection algorithms on each temporal snapshot:

```bash
PYTHONPATH=. python main_static.py \
    --dataset-path ./data/CollegeMsg.txt \
    --dataset CollegeMsg \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 10
```

### Running Dynamic Benchmarks

Run streaming/dynamic community detection algorithms:

```bash
PYTHONPATH=. python main_dynamic.py \
    --dataset-path ./data/CollegeMsg.txt \
    --dataset CollegeMsg \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 10
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-path` | str | `data/CollegeMsg.txt` | Path to the dataset file |
| `--dataset` | str | `CollegeMsg` | Name of the dataset (used for tagging) |
| `--source-idx` | int | `0` | Column index for source node |
| `--target-idx` | int | `1` | Column index for target node |
| `--batch-range` | float | `1e-4` | Fraction of edges per temporal batch |
| `--initial-fraction` | float | `0.4` | Fraction of edges used for base graph |
| `--max-steps` | int | `10` | Maximum number of temporal snapshots |
| `--load-full-nodes` | flag | `False` | Pre-load all nodes into the base graph |
| `--delimiter` | str | `" "` | Delimiter used in the dataset file |

## Example Output

```
Loaded temporal graph: 11 snapshots
Base graph: 1893 nodes, 7048 edges

Algorithms:   0%|          | 0/3 [00:00<?, ?it/s]
coach: 0it [00:00, ?it/s]
Steps: 0%|          | 0/11 [00:00<?, ?it/s]

coach:
  steps: 11
  avg_runtime: 0.2341s
  avg_modularity: 0.4215
```

## Generating Visualizations

After running benchmarks, you can generate visualization plots:

```bash
# Using the plot script (recommended)
./scripts/plot.sh

# Or manually
PYTHONPATH=. python3 tools/fetch_and_merge.py
PYTHONPATH=. python3 tools/plots.py
```

See [Visualization Guide](visualization.md) for more details.

## Configuration Files

### Algorithms Configuration ([`config/algorithms.yaml`](../config/algorithms.yaml))

This file contains all available algorithms and their parameters. To change which algorithms run, modify the `target_algorithms` list:

```yaml
target_algorithms:
  - coach
  - percomvc
  - core_expansion
```

### Dataset Configuration ([`config/dataset_config.yaml`](../config/dataset_config.yaml))

This file contains all available datasets and their parameters. To change which datasets run, modify the `target_datasets` list:

```yaml
target_datasets:
  - college-msg
  - bio-wormnet-v3
```

## Troubleshooting

### Dataset Not Found

If you get a "Dataset not found" error:

1. Check that the dataset file exists in the `data/` directory
2. Verify the dataset name matches the configuration in [`config/dataset_config.yaml`](../config/dataset_config.yaml)

### Algorithm Not Found

If you get an "Algorithm not found" error:

1. Check that the algorithm is defined in [`config/algorithms.yaml`](../config/algorithms.yaml)
2. Verify the algorithm is in the `target_algorithms` list

### Comet ML Connection Issues

If you encounter Comet ML connection issues:

1. Verify your credentials in `.env` file
2. Check your internet connection
3. Ensure your API key is valid

## Next Steps

- See [Development Guide](development_guide.md) for adding new algorithms
- See [Metrics Documentation](metrics.md) for understanding evaluation metrics
- See [Visualization Guide](visualization.md) for generating plots
