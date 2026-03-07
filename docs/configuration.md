# Configuration Guide

The repository uses YAML files under `config/` to define algorithms, datasets, and visualization settings.

## Files

| File | Purpose |
| --- | --- |
| [`config/algorithms.yaml`](../config/algorithms.yaml) | Which algorithms run and how they are loaded |
| [`config/dataset_config.yaml`](../config/dataset_config.yaml) | Named datasets and their temporal graph parameters |
| [`config/visualization.yaml`](../config/visualization.yaml) | Comet projects, metric keys, and plot styling |

## `config/algorithms.yaml`

This file drives `main.py` through [`src/algorithms/factory.py`](../src/algorithms/factory.py).

### Structure

```yaml
target_algorithms:
  - coach
  - core_expansion
  - graph_entropy
  - df_louvain

algorithms:
  coach:
    module: "cdlib.algorithms"
    function: "coach"
    params: {}
    type: "static"
    clustering_type: "overlapping"
    metadata: {}
    description: "COACH"
```

### Important fields

| Field | Meaning |
| --- | --- |
| `target_algorithms` | Ordered list of algorithms to execute |
| `module` | Python import path |
| `function` | Function or class loaded from that module |
| `params` | Keyword args passed to the function or constructor |
| `type` | `static` or `dynamic` |
| `clustering_type` | `crisp` or `overlapping` |
| `description` | Human-readable label |

### Behavior

- Static functions are wrapped and called once per snapshot.
- Dynamic functions consume the full `TemporalGraph`.
- If `function` points to a class, the loader handles class instantiation or direct wrapping depending on the class type.

### Current defaults

At the moment, the default `target_algorithms` list is:

```yaml
target_algorithms:
  - coach
  - core_expansion
  - graph_entropy
  - df_louvain
```

Change that list to choose what `main.py` runs.

## `config/dataset_config.yaml`

This file is consumed primarily by `scripts/benchmark.sh`.

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
| `source_idx` | Source column index |
| `target_idx` | Target column index |
| `delimiter` | Field separator |
| `max_steps` | Maximum number of temporal updates |
| `initial_fraction` | Fraction of rows used to form the base graph |
| `batch_range` | Fraction of total edges assigned to each temporal batch |

### Notes

- `scripts/benchmark.sh` parses this file with shell text processing, so keep the formatting simple.
- Values in `common` are reused through YAML anchors.
- Paths under `data/` refer to local files; the repository does not track the data directory.

## `config/visualization.yaml`

This file controls the Comet fetch/merge/plot pipeline.

### Top-level keys used in code

| Key | Used for |
| --- | --- |
| `raw_dir` | Raw Comet export location |
| `merge_dir` | Merged metric location |
| `output_dir` | Plot output location |
| `workspace` | Default Comet workspace |
| `hyperparameters` | Hyperparameter keys to inspect during merging |
| `metric_keys` | Metrics fetched from Comet |
| `projects` | Comet project names to fetch |
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
python -c "import yaml; yaml.safe_load(open('config/algorithms.yaml')); yaml.safe_load(open('config/dataset_config.yaml')); yaml.safe_load(open('config/visualization.yaml')); print('ok')"
```

## Common Changes

### Run fewer algorithms

Edit `target_algorithms` in [`config/algorithms.yaml`](../config/algorithms.yaml).

### Run different datasets with the shell wrapper

Edit `target_datasets` in [`config/dataset_config.yaml`](../config/dataset_config.yaml).

### Plot different algorithms or projects

Edit `projects`, `selected_algorithms`, `methods_name`, and `colors` in [`config/visualization.yaml`](../config/visualization.yaml).
