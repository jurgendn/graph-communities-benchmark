# Configuration Guide

The repository uses YAML files under `config/` to define algorithms, datasets, and visualization settings.

## Files

| File | Purpose |
| --- | --- |
| [`config/algorithms.yaml`](../config/algorithms.yaml) | Which algorithms run and how they are loaded |
| [`config/dynamic_dataset_config.yaml`](../config/dynamic_dataset_config.yaml) | Named temporal and LFR datasets |
| [`config/static_dataset_config.yaml`](../config/static_dataset_config.yaml) | Named static graph datasets |
| [`config/visualization_dynamic.yaml`](../config/visualization_dynamic.yaml) | Dynamic benchmark visualization settings |
| [`config/visualization_static.yaml`](../config/visualization_static.yaml) | Static benchmark visualization settings |

## `config/algorithms.yaml`

This file drives both [`main.py`](../main.py) and [`main_static.py`](../main_static.py) through [`src/algorithms/factory.py`](../src/algorithms/factory.py).

### Structure

```yaml
target_snapshot_algorithms:
  - coach
  - core_expansion
  - graph_entropy

target_temporal_algorithms:
  - tiles

snapshot_algorithms:
  coach:
    module: "cdlib.algorithms"
    function: "coach"
    params: {}
    clustering_type: "overlapping"
    metadata: {}
    description: "COACH"
```

### Important fields

| Field | Meaning |
| --- | --- |
| `target_snapshot_algorithms` | Ordered list of per-snapshot algorithms to execute |
| `target_temporal_algorithms` | Ordered list of full-temporal-graph algorithms to execute |
| `snapshot_algorithms` | Definitions for snapshot-by-snapshot algorithms |
| `temporal_algorithms` | Definitions for full-temporal-graph algorithms |
| `module` | Python import path |
| `function` | Function or class loaded from that module |
| `params` | Keyword args passed to the function or constructor |
| `clustering_type` | `crisp` or `overlapping` |
| `description` | Human-readable label |

### Behavior

- Snapshot algorithms are wrapped and called once per snapshot; they can be used in both `main.py` and `main_static.py`.
- Temporal algorithms consume the full `TemporalGraph`; they are only meaningful in `main.py`.
- The `module` field usually points either to `cdlib.algorithms` or to an implementation under `src/models/`.

### Current defaults

At the moment, the default target lists are:

```yaml
target_snapshot_algorithms:
  - coach
  - core_expansion
  - graph_entropy

target_temporal_algorithms:
  - tiles
```

Change those lists to choose what runs. `main.py` loads both sections; `main_static.py` only executes snapshot algorithms.

## `config/dynamic_dataset_config.yaml`

This file is consumed by `scripts/benchmark.sh` and `scripts/parse_config.py`.

### Structure

```yaml
target_datasets:
  - college-msg

common: &common_settings
  max_steps: 9
  initial_fraction: 0.4
  batch_range: 0.00001

datasets:
  college-msg:
    path: ./data/CollegeMsg.txt
    dataset_name: CollegeMsg
    source_idx: 0
    target_idx: 1
    delimiter: " "
    <<: *common_settings
```

### Important fields

| Field | Meaning |
| --- | --- |
| `target_datasets` | Datasets shown by `./scripts/benchmark.sh --list` and used by `--all` |
| `path` | Local dataset file path |
| `dataset_name` | Label sent to logs and CLI output |
| `type` | `edge_list` or `lfr` for temporal config entries |
| `source_idx` | Source column index |
| `target_idx` | Target column index |
| `delimiter` | Field separator |
| `max_steps` | Maximum number of temporal updates |
| `initial_fraction` | Fraction of rows used to form the base graph |
| `batch_range` | Fraction of total edges assigned to each temporal batch |
| `ground_truth_attr` | Node attribute used to build ground truth when available |

### Dataset section

- `datasets`: temporal edge-list and LFR benchmarks used by `main.py`

### Temporal dataset types

Temporal entries use an explicit `type` field:

- `edge_list`: one file converted into a temporal graph through batching
- `lfr`: folder of `snapshot_t*.gml` files diffed into temporal steps

Example:

```yaml
datasets:
  synthetic-n-5000-1:
    dataset_name: synthetic_n_5000_1
    path: data/synthetic_n_5000_1/
    type: "lfr"
    ground_truth_attr: "communities"
    source_idx: 0
    target_idx: 1
    delimiter: " "
```

### Notes

- `scripts/benchmark.sh` reads `datasets` through `scripts/parse_config.py`.
- Values in `common` are reused through YAML anchors.
- Paths under `data/` refer to local files; the repository does not track the data directory.

## `config/static_dataset_config.yaml`

This file is consumed by `main_static.py --config` and `scripts/benchmark_static.sh`.

### Structure

```yaml
target_datasets:
  - karate

common: &common_settings
  preload_fraction: 1.0

datasets:
  karate:
    <<: *common_settings
    dataset_name: karate
    path: data/karate.txt
    delimiter: " "
    source_idx: 0
    target_idx: 1
```

### Important fields

| Field | Meaning |
| --- | --- |
| `target_datasets` | Datasets shown by `./scripts/benchmark_static.sh --list` and used by static `--all` |
| `path` | Local dataset file path |
| `dataset_name` | Label sent to logs and CLI output |
| `source_idx` | Source column index |
| `target_idx` | Target column index |
| `delimiter` | Field separator |
| `preload_fraction` | Fraction of edges loaded into the single static snapshot |
| `ground_truth_attr` | Node attribute used to build ground truth when available |

### Notes

- Static entries do not need a `type` because they are always loaded through `main_static.py` as one-snapshot temporal graphs.
- `scripts/benchmark_static.sh` reads `datasets` from this file.
- Values in `common` are reused through YAML anchors.
- Paths under `data/` refer to local files; the repository does not track the data directory.

## `config/visualization_dynamic.yaml` and `config/visualization_static.yaml`

These files control the Comet fetch/merge/plot pipeline for each benchmark mode.

### Top-level keys used in code

| Key | Used for |
| --- | --- |
| `workspace` | Default Comet workspace |
| `directories` | Raw, merged, and output locations for the selected mode |
| `hyperparameters` | Hyperparameter keys inspected during merging |
| `metric_keys` | Metrics fetched from Comet |
| `projects` | Comet project names to fetch |
| `use_batch_ranges` | Whether merged outputs are grouped by batch range |
| `plotter` | Dataset grouping and algorithm styling |

### Shared structure

Both files use the same schema:

| Key | Meaning |
| --- | --- |
| `directories.raw_dir` | Raw Comet export location |
| `directories.merge_dir` | Merged metric location |
| `directories.output_dir` | Plot output location |
| `hyperparameters` | Hyperparameter keys to inspect during merging |
| `metric_keys` | Metrics fetched from Comet |
| `projects` | Comet project names to fetch |
| `use_batch_ranges` | Whether merged outputs are grouped by batch range |
| `plotter` | Dataset grouping and algorithm styling |

### Metric keys

The current visualization pipeline expects metric names such as:

```yaml
metric_keys:
  - cdlib_modularity
  - customize_q0_modularity
  - num_communities
  - runtime
```

### Plotter config

The `plotter.common_plotter_settings` section defines:

- batch-range categories
- dataset groups
- display-name mappings
- selected algorithms
- colors
- markers
- plotting order

Those values are read by the visualization helpers in [`src/visualization/config.py`](../src/visualization/config.py) and [`src/visualization/utils.py`](../src/visualization/utils.py).

## Validation Tips

Check YAML syntax before running long jobs:

```bash
python -c "import yaml; yaml.safe_load(open('config/algorithms.yaml')); yaml.safe_load(open('config/dynamic_dataset_config.yaml')); yaml.safe_load(open('config/static_dataset_config.yaml')); yaml.safe_load(open('config/visualization_dynamic.yaml')); yaml.safe_load(open('config/visualization_static.yaml')); print('ok')"
```

## Common Changes

### Run fewer algorithms

Edit `target_snapshot_algorithms` and `target_temporal_algorithms` in [`config/algorithms.yaml`](../config/algorithms.yaml).

### Run different datasets with the shell wrapper

Edit `target_datasets` in [`config/dynamic_dataset_config.yaml`](../config/dynamic_dataset_config.yaml).

### Run different static datasets with the static shell wrapper

Edit `target_datasets` in [`config/static_dataset_config.yaml`](../config/static_dataset_config.yaml).

### Plot different algorithms or projects

Edit [`config/visualization_dynamic.yaml`](../config/visualization_dynamic.yaml) or [`config/visualization_static.yaml`](../config/visualization_static.yaml), especially `projects`, `directories`, `selected_algorithms`, `methods_name`, and `colors`.
