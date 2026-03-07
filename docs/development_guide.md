# Development Guide

This project is organized around a unified benchmarking pipeline:

1. load a temporal graph
2. load configured algorithms
3. run algorithms through wrappers
4. evaluate each snapshot
5. log results to Comet ML

Core orchestration lives in [`main.py`](../main.py), [`src/algorithms/factory.py`](../src/algorithms/factory.py), and [`src/pipeline_utils.py`](../src/pipeline_utils.py).

## Add A New Algorithm

Templates are available in:

- [`templates/static_algorithm_template.py`](../templates/static_algorithm_template.py)
- [`templates/dynamic_algorithm_template.py`](../templates/dynamic_algorithm_template.py)

### Static algorithm

Place the implementation in one of:

- `src/models/static_methods/overlap/`
- `src/models/static_methods/crisp/`

Implement the unified interface from [`src/algorithms/base.py`](../src/algorithms/base.py):

```python
from typing import List

from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.factory import TemporalGraph


class MyAlgorithm(CommunityDetectionAlgorithm):
    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        ...
```

If your algorithm is already a snapshot-level function, you can often register it directly and let `StaticMethodWrapper` handle iteration.

### Dynamic algorithm

Place the implementation in one of:

- `src/models/dynamic_methods/overlap/`
- `src/models/dynamic_methods/crisp/`

Dynamic methods consume the full `TemporalGraph`. They can return:

- `List[NodeClustering]`, or
- `MethodDynamicResults`

Returning `MethodDynamicResults` is useful when you want to preserve custom runtimes or precomputed traces.

### Register the algorithm

Add it to [`config/algorithms.yaml`](../config/algorithms.yaml):

```yaml
target_algorithms:
  - my_algorithm

algorithms:
  my_algorithm:
    module: "src.models.static_methods.overlap.my_algorithm"
    function: "MyAlgorithm"
    params: {}
    type: "static"
    clustering_type: "overlapping"
    metadata: {}
    description: "My custom method"
```

### Test it

```bash
python main.py --dataset-path ./data/CollegeMsg.txt --dataset CollegeMsg --max-steps 5
```

## Add A New Dataset

1. Place the local file under `data/`.
2. Add an entry to [`config/dataset_config.yaml`](../config/dataset_config.yaml).
3. Optionally add the dataset key to `target_datasets`.

Example:

```yaml
datasets:
  my_dataset:
    path: ./data/my_dataset.txt
    dataset_name: MyDataset
    source_idx: 0
    target_idx: 1
    delimiter: " "
    max_steps: 9
    initial_fraction: 0.4
    batch_range: 0.00001
```

## Work With LFR Ground Truth

`main.py` also supports folders of GraphML snapshots.

Expected naming:

```text
snapshot_t0.graphml
snapshot_t1.graphml
snapshot_t2.graphml
```

Run them with:

```bash
python main.py --lfr-folder ./data/lfr_benchmark/my_lfr --ground-truth-attr label
```

The loader is implemented in [`src/dataloader/data_reader.py`](../src/dataloader/data_reader.py).

## Add A New Metric

1. Implement the metric in `src/evaluations/`.
2. Compute it during evaluation in [`src/pipeline_utils.py`](../src/pipeline_utils.py).
3. Extend [`src/factory/communities.py`](../src/factory/communities.py) if new traces or averages are needed.
4. Add the metric key to [`config/visualization.yaml`](../config/visualization.yaml) if it should appear in fetched and plotted outputs.

## Important Modules

| Path | Role |
| --- | --- |
| [`main.py`](../main.py) | unified CLI entry point |
| [`src/utils/arg_parser.py`](../src/utils/arg_parser.py) | CLI arguments |
| [`src/algorithms/factory.py`](../src/algorithms/factory.py) | loads configured algorithms |
| [`src/algorithms/wrappers.py`](../src/algorithms/wrappers.py) | adapts static and dynamic call styles |
| [`src/pipeline_utils.py`](../src/pipeline_utils.py) | run, evaluate, and log pipeline |
| [`src/evaluations/metrics.py`](../src/evaluations/metrics.py) | modularity and NMI dispatch |
| [`src/evaluations/onmi_fast.py`](../src/evaluations/onmi_fast.py) | fast overlapping NMI implementation |
| [`src/visualization/`](../src/visualization) | Comet fetch/merge/plot pipeline |

## Development Tips

- Prefer updating `main.py` workflows over extending the legacy entry points.
- Keep algorithm config entries explicit and simple.
- When changing plot behavior, verify both `config/visualization.yaml` and the helpers under `src/visualization/`.
- Because `data/`, `experiments/`, and `assets/` are ignored, document any required local setup when adding new workflows.
