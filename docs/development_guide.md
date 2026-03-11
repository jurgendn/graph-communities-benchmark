# Development Guide

This project is organized around one shared benchmark pipeline:

1. load a graph as a `TemporalGraph`
2. load configured algorithms from `config/algorithms.yaml`
3. run each algorithm through the wrapper layer
4. evaluate each snapshot
5. log results to Comet ML

The main orchestration code lives in [`main.py`](../main.py), [`main_static.py`](../main_static.py), [`src/algorithms/factory.py`](../src/algorithms/factory.py), and [`src/pipeline_utils.py`](../src/pipeline_utils.py).

## Entry Points

- [`main.py`](../main.py): temporal edge-list and LFR `.gml` benchmarks
- [`main_static.py`](../main_static.py): static graph benchmarks

`main.py` can run both snapshot algorithms and temporal algorithms.
`main_static.py` runs snapshot algorithms only.

Static benchmarking reuses the same pipeline by loading graphs as `TemporalGraph(base_graph=G, steps=[])`.

## Model Layout

Algorithm implementations live under [`src/models/`](../src/models):

- `src/models/static/` — snapshot-based algorithms
- `src/models/dynamic/` — temporal algorithms
- `src/models/common/` — shared helpers and mixins

Within `static/` and `dynamic/`, use:

- `crisp/` for non-overlapping methods
- `overlap/` for overlapping methods

## Add A New Algorithm

Templates are available in:

- [`templates/static_algorithm_template.py`](../templates/static_algorithm_template.py)
- [`templates/dynamic_algorithm_template.py`](../templates/dynamic_algorithm_template.py)

### Add a new snapshot algorithm

Place the implementation in one of:

- `src/models/static/overlap/`
- `src/models/static/crisp/`

Implement the common interface from [`src/algorithms/base.py`](../src/algorithms/base.py):

```python
from typing import List

from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.factory import TemporalGraph


class MyAlgorithm(CommunityDetectionAlgorithm):
    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        ...
```

If your method already works on a single NetworkX graph, you can often register the function directly and let `StaticMethodWrapper` iterate over snapshots for you.

### Add a new temporal algorithm

Place the implementation in one of:

- `src/models/dynamic/overlap/`
- `src/models/dynamic/crisp/`

Temporal methods consume the full `TemporalGraph`. They can return either:

- `List[NodeClustering]`
- `MethodDynamicResults`

Returning `MethodDynamicResults` is useful when you want to preserve custom runtimes, affected-node traces, or other precomputed values.

### Register the algorithm

Add a snapshot method like this:

```yaml
target_snapshot_algorithms:
  - my_algorithm

snapshot_algorithms:
  my_algorithm:
    module: "src.models.static.overlap.my_algorithm"
    function: "MyAlgorithm"
    params: {}
    clustering_type: "overlapping"
    metadata: {}
    description: "My custom method"
```

Add a temporal method like this:

```yaml
target_temporal_algorithms:
  - my_temporal_algorithm

temporal_algorithms:
  my_temporal_algorithm:
    module: "src.models.dynamic.overlap.my_temporal_algorithm"
    function: "MyTemporalAlgorithm"
    params: {}
    clustering_type: "overlapping"
    metadata: {}
    description: "My temporal method"
```

## Test New Algorithms

Use a small temporal run and a small static run:

```bash
python main.py --dataset-path ./data/CollegeMsg.txt --dataset CollegeMsg --max-steps 5
python main_static.py --builtin karate --num-runs 1
```

If you changed only a module path or config entry, also run:

```bash
python -c "from src.algorithms.factory import load_algorithms; print(sorted(load_algorithms('config/algorithms.yaml').keys()))"
```

## Add A New Dataset

### Temporal dataset

1. Place the local file under `data/`.
2. Add an entry under `datasets` in [`config/dataset_config.yaml`](../config/dataset_config.yaml).
3. Set `type` to `edge_list` or `lfr`.
4. Optionally add the dataset key to `target_datasets`.

Example:

```yaml
datasets:
  my_dataset:
    path: ./data/my_dataset.txt
    dataset_name: MyDataset
    type: edge_list
    source_idx: 0
    target_idx: 1
    delimiter: " "
    max_steps: 9
    initial_fraction: 0.4
    batch_range: 0.00001
```

### Static dataset

1. Place the local file under `data/`.
2. Add an entry under `static_graphs` in [`config/dataset_config.yaml`](../config/dataset_config.yaml).
3. Optionally add the dataset key to `target_static_datasets`.

Example:

```yaml
static_graphs:
  my_static_graph:
    path: ./data/my_static_graph.txt
    dataset_name: MyStaticGraph
    source_idx: 0
    target_idx: 1
    delimiter: " "
```

## Work With LFR Ground Truth

`main.py` supports folders of `.gml` snapshots.

Expected naming:

```text
snapshot_t0.gml
snapshot_t1.gml
snapshot_t2.gml
```

Run them with:

```bash
python main.py --lfr-folder ./data/synthetic_n_5000_1 --ground-truth-attr communities
```

The loader is implemented in [`src/dataloader/data_reader.py`](../src/dataloader/data_reader.py). Ground truth is precomputed at load time and stored on `TemporalGraph._ground_truth_clusterings`.

## Add A New Metric

1. Implement the metric in `src/evaluations/`.
2. Compute it during evaluation in [`src/pipeline_utils.py`](../src/pipeline_utils.py).
3. Extend [`src/factory/communities.py`](../src/factory/communities.py) if new traces or averages are needed.
4. Add the metric key to [`config/visualization.yaml`](../config/visualization.yaml) if it should appear in fetched and plotted outputs.

## Important Modules

| Path | Role |
| --- | --- |
| [`main.py`](../main.py) | temporal and LFR CLI entry point |
| [`main_static.py`](../main_static.py) | static CLI entry point |
| [`src/utils/arg_parser.py`](../src/utils/arg_parser.py) | temporal CLI arguments |
| [`src/algorithms/factory.py`](../src/algorithms/factory.py) | loads configured algorithms |
| [`src/algorithms/wrappers.py`](../src/algorithms/wrappers.py) | adapts snapshot and temporal call styles |
| [`src/dataloader/data_reader.py`](../src/dataloader/data_reader.py) | temporal and LFR dataset loaders |
| [`src/dataloader/static_loader.py`](../src/dataloader/static_loader.py) | static loader returning one-snapshot `TemporalGraph` |
| [`src/pipeline_utils.py`](../src/pipeline_utils.py) | run, evaluate, and log pipeline |
| [`src/evaluations/metrics.py`](../src/evaluations/metrics.py) | modularity and NMI dispatch |
| [`src/evaluations/onmi_fast.py`](../src/evaluations/onmi_fast.py) | fast overlapping NMI implementation |
| [`src/models/common/`](../src/models/common) | shared model helpers and mixins |
| [`src/visualization/`](../src/visualization) | Comet fetch/merge/plot pipeline |

## Development Tips

- Prefer updating the shared pipeline in [`src/pipeline_utils.py`](../src/pipeline_utils.py) over duplicating evaluation logic.
- Keep snapshot algorithms stateless when possible; the wrapper will handle per-snapshot iteration.
- Keep temporal algorithms explicit about which state they carry from one snapshot to the next.
- Keep algorithm config entries simple and readable; most users will interact with YAML before they touch Python.
- When changing plot behavior, verify both [`config/visualization.yaml`](../config/visualization.yaml) and the helpers under [`src/visualization/`](../src/visualization/).
- Because `data/`, `experiments/`, and `assets/` are ignored, document any required local setup when adding new workflows.
