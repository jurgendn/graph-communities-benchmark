# Development Guide

This project is organized around one shared benchmark pipeline:

1. load a graph as a `TemporalGraph`
2. load target algorithms from `config/algorithms.yaml` (validated against the `@register`-based algorithm registry)
3. run each algorithm through the wrapper layer
4. evaluate each snapshot
5. log results to Comet ML

The main orchestration code lives in [`main.py`](../../main.py), [`main_static.py`](../../main_static.py), [`src/algorithms/factory.py`](../../src/algorithms/factory.py), and [`src/core/pipeline.py`](../../src/core/pipeline.py).

## Entry Points

- [`main.py`](../../main.py): temporal edge-list and LFR `.gml` benchmarks
- [`main_static.py`](../../main_static.py): static graph benchmarks

`main.py` can run both snapshot algorithms and temporal algorithms.
`main_static.py` runs snapshot algorithms only.

Static benchmarking reuses the same pipeline by loading graphs as `TemporalGraph(base_graph=G, steps=[])`.

## Model Layout

Algorithm implementations live under [`src/models/`](../../src/models):

- `src/models/static/` — snapshot-based algorithms
- `src/models/dynamic/` — temporal algorithms
- `src/models/common/` — shared helpers and mixins

Within `static/` and `dynamic/`, use:

- `crisp/` for non-overlapping methods
- `overlap/` for overlapping methods

## Add A New Algorithm

Templates are available in:

- [`templates/static_algorithm_template.py`](../../templates/static_algorithm_template.py)
- [`templates/dynamic_algorithm_template.py`](../../templates/dynamic_algorithm_template.py)

Adding an algorithm requires **three touches**:

1. **Implement the algorithm file** with a `@register(...)` decorator.
2. **Add the module path** to `_REGISTRATION_MODULES` in [`src/algorithms/factory.py`](../../src/algorithms/factory.py).
3. **Add the algorithm name** to the appropriate target list in [`config/algorithms.yaml`](../../config/algorithms.yaml).

### Add a new snapshot algorithm

Place the implementation in one of:

- `src/models/static/overlap/`
- `src/models/static/crisp/`

Implement the common interface from [`src/algorithms/base.py`](../../src/algorithms/base.py) and decorate with `@register`:

```python
from typing import List

from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.registry import register
from src.core.temporal_graph import TemporalGraph


@register(
    name="my_algorithm",
    algo_type="static",
    clustering_type="overlapping",   # or "crisp"
    default_params={"threshold": 0.5},
    description="My overlapping community detection method",
)
class MyAlgorithm(CommunityDetectionAlgorithm):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        ...
```

If your method already works on a single NetworkX graph, you can register the function directly and let `StaticMethodWrapper` iterate over snapshots for you.

### Add a new temporal algorithm

Place the implementation in one of:

- `src/models/dynamic/overlap/`
- `src/models/dynamic/crisp/`

Temporal methods consume the full `TemporalGraph`. They can return either:

- `List[NodeClustering]`
- `MethodDynamicResults`

Returning `MethodDynamicResults` is useful when you want to preserve custom runtimes, affected-node traces, or other precomputed values.

### Register the algorithm

**Step 1.** The `@register(...)` decorator in your source file handles identity. No YAML definitions needed — just the decorator.

**Step 2.** Add the module path to `_REGISTRATION_MODULES` in [`src/algorithms/factory.py`](../../src/algorithms/factory.py):

```python
_REGISTRATION_MODULES = [
    ...
    "src.models.static.overlap.my_algorithm",
]
```

**Step 3.** Add the algorithm name to `config/algorithms.yaml`:

For a snapshot method:

```yaml
target_snapshot_algorithms:
  - my_algorithm
```

For a temporal method:

```yaml
target_temporal_algorithms:
  - my_temporal_algorithm
```

Optionally override default parameters:

```yaml
algorithm_params:
  my_algorithm:
    threshold: 0.7
```

## Test New Algorithms

Use a small temporal run and a small static run:

```bash
python main.py --dataset-path ./data/CollegeMsg.txt --dataset CollegeMsg --max-steps 5
python main_static.py --builtin karate --num-runs 1
```

Verify your algorithm appears in the registry:

```bash
python -c "from src.algorithms.factory import _ensure_registrations; _ensure_registrations(); from src.algorithms.registry import ALGORITHM_REGISTRY; print(sorted(ALGORITHM_REGISTRY.keys()))"
```

If you changed only config entries, also verify loading works:

```bash
python -c "from src.algorithms.factory import load_algorithms; print(sorted(load_algorithms('config/algorithms.yaml').keys()))"
```

## Add A New Dataset

Before adding config entries, make sure the local files match the expected loader format in the [Dataset Preparation Guide](../dataset_preparation.md).

### Temporal dataset

1. Place the local file under `data/`.
2. Add an entry under `datasets` in [`config/dynamic_dataset_config.yaml`](../../config/dynamic_dataset_config.yaml).
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
2. Add an entry under `datasets` in [`config/static_dataset_config.yaml`](../../config/static_dataset_config.yaml).
3. Optionally add the dataset key to `target_datasets`.

Example:

```yaml
common: &common_settings
  preload_fraction: 1.0

datasets:
  my_static_graph:
    <<: *common_settings
    path: ./data/my_static_graph.txt
    dataset_name: MyStaticGraph
    source_idx: 0
    target_idx: 1
    delimiter: " "
```

## Work With LFR Ground Truth

`main.py` supports folders of `.gml` snapshots.

`main_static.py` also supports configured LFR folders in static mode and loads one snapshot as a single static graph.

Expected naming:

```text
snapshot_t0.gml
snapshot_t1.gml
snapshot_t2.gml
```

Run them with:

```bash
python main.py --lfr-folder ./data/synthetic_n_5000_1 --ground-truth-attr communities
./scripts/benchmark_static.sh synthetic-n-5000-1 1
```

The temporal loader is implemented in [`src/data/data_reader.py`](../../src/data/data_reader.py). The static single-snapshot loader is implemented in [`src/data/static_loader.py`](../../src/data/static_loader.py). Ground truth is precomputed at load time and stored on `TemporalGraph._ground_truth_clusterings`.

## Document Vast-PMO

If your branch includes a `VAST-PMO.md` design note, keep it in sync when you change:

- the Louvain backbone initialization
- the PMO gain function
- overlap constraints such as `r`
- parameter names or semantics in [`config/algorithms.yaml`](../../config/algorithms.yaml)

## Add A New Metric

1. Implement the metric in `src/evaluation/`.
2. Compute it during evaluation in [`src/core/pipeline.py`](../../src/core/pipeline.py).
3. Extend [`src/core/results.py`](../../src/core/results.py) if new traces or averages are needed.
4. Add the metric key to [`config/visualization_dynamic.yaml`](../../config/visualization_dynamic.yaml) and/or [`config/visualization_static.yaml`](../../config/visualization_static.yaml) if it should appear in fetched and plotted outputs.

## Add A New Post-Hoc Analyzer

Post-hoc analyzers operate on downloaded Comet artifacts rather than during the benchmark run.

1. Implement the analyzer function in `src/analyzer/` with the signature `fn(payload, clusterings=None, **kwargs) -> dict`.
2. Register it in the `ANALYZERS` dict in [`src/analyzer/runner.py`](../../src/analyzer/runner.py).
3. The CLI (`tools/analyze.py`) picks it up automatically via the `--analyzer` flag.

The `payload` is a `ClusteringArtifactPayload` (JSON metadata); `clusterings` is `Optional[List[NodeClustering]]` loaded from pickle. Both are provided by the runner.

## Important Modules

| Path | Role |
| --- | --- |
| [`main.py`](../../main.py) | temporal and LFR CLI entry point |
| [`main_static.py`](../../main_static.py) | static CLI entry point |
| [`src/utils/arg_parser.py`](../../src/utils/arg_parser.py) | temporal CLI arguments |
| [`src/algorithms/registry.py`](../../src/algorithms/registry.py) | `@register` decorator and `ALGORITHM_REGISTRY` |
| [`src/algorithms/cdlib_adapters.py`](../../src/algorithms/cdlib_adapters.py) | CDlib algorithm registrations |
| [`src/algorithms/factory.py`](../../src/algorithms/factory.py) | loads and instantiates registered algorithms |
| [`src/algorithms/wrappers.py`](../../src/algorithms/wrappers.py) | adapts snapshot and temporal call styles |
| [`src/data/data_reader.py`](../../src/data/data_reader.py) | temporal and LFR dataset loaders |
| [`src/data/static_loader.py`](../../src/data/static_loader.py) | static loader returning one-snapshot `TemporalGraph` |
| [`src/core/pipeline.py`](../../src/core/pipeline.py) | run, evaluate, and log pipeline |
| [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py) | modularity and NMI dispatch |
| [`src/evaluation/onmi_fast.py`](../../src/evaluation/onmi_fast.py) | fast overlapping NMI implementation |
| [`src/analyzer/artifacts.py`](../../src/analyzer/artifacts.py) | artifact enrichment, serialization, upload/download |
| [`src/analyzer/runner.py`](../../src/analyzer/runner.py) | post-hoc analysis orchestration |
| [`src/analyzer/overlap_quality.py`](../../src/analyzer/overlap_quality.py) | overlap quality structural analysis |
| [`tools/analyze.py`](../../tools/analyze.py) | CLI for post-hoc artifact analysis |
| [`src/models/common/`](../../src/models/common) | shared model helpers and mixins |
| [`src/visualization/`](../../src/visualization) | Comet fetch/merge/plot pipeline |

## Development Tips

- Prefer updating the shared pipeline in [`src/core/pipeline.py`](../../src/core/pipeline.py) over duplicating evaluation logic.
- Keep snapshot algorithms stateless when possible; the wrapper will handle per-snapshot iteration.
- Keep temporal algorithms explicit about which state they carry from one snapshot to the next.
- Keep algorithm config entries simple and readable; the YAML run config should only list names and parameter overrides.
- When changing plot behavior, verify the relevant visualization config file and the helpers under [`src/visualization/`](../../src/visualization/).
- Because `data/`, `experiments/`, and `assets/` are ignored, document any required local setup when adding new workflows.
