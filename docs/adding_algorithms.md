# Adding Algorithms

This guide walks you through integrating a new community detection algorithm into the benchmark. Whether you have a snapshot-level method or a temporal (streaming) method, the process follows the same three-touch pattern.

## Overview

Every algorithm in this project is registered via a `@register` decorator that declares its identity and metadata in code. The YAML config (`config/algorithms.yaml`) is a minimal run configuration — it only selects which registered algorithms to execute and optionally overrides default parameters.

### Three touches to add an algorithm

1. **Implement the algorithm file** with a `@register(...)` decorator.
2. **Add the module path** to `_REGISTRATION_MODULES` in `src/algorithms/factory.py`.
3. **Add the algorithm name** to the appropriate target list in `config/algorithms.yaml`.

That's it. No module paths, function names, or descriptions in YAML.

## Two integration paths

| Path | Best for | How it works |
| --- | --- | --- |
| **Class-based** (recommended) | Full control, custom iteration, temporal state | Subclass `CommunityDetectionAlgorithm`, implement `__call__` |
| **Plain function** | Simple per-snapshot CDlib-style methods | Decorate a function that takes `(graph, **params)` and returns `NodeClustering` |

Both paths use `@register`. The factory detects which path you used and wraps accordingly.

---

## Path 1: Class-based (recommended)

### Static (snapshot) algorithm

Copy the template:

```bash
cp templates/static_algorithm_template.py src/models/static/overlap/my_algorithm.py
```

Edit the file:

```python
from typing import List

import networkx as nx
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.registry import register
from src.core.temporal_graph import TemporalGraph


@register(
    name="my_algorithm",
    algo_type="static",
    clustering_type="overlapping",   # or "crisp"
    default_params={"threshold": 0.5, "iterations": 10},
    description="My community detection method (Author et al., 2024)",
)
class MyAlgorithm(CommunityDetectionAlgorithm):
    """
    Short description of the algorithm.

    Args:
        threshold: Minimum similarity to merge communities.
        iterations: Number of refinement passes.
    """

    def __init__(self, threshold: float = 0.5, iterations: int = 10):
        self.threshold = threshold
        self.iterations = iterations

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        results = []
        for snapshot in tg.iter_snapshots():
            communities = self._detect(snapshot)
            nc = NodeClustering(
                communities=communities,
                graph=snapshot,
                method_name="my_algorithm",
            )
            results.append(nc)
        return results

    def _detect(self, graph: nx.Graph) -> List[List]:
        # Your algorithm here.
        # Return a list of communities, each a list of node IDs.
        ...
```

Key points:

- `name` in `@register` must be unique across the entire project.
- `default_params` keys must match the `__init__` parameter names.
- `algo_type="static"` tells the factory this is a per-snapshot algorithm.
- Return one `NodeClustering` per snapshot, in iteration order.

### Dynamic (temporal) algorithm

Copy the template:

```bash
cp templates/dynamic_algorithm_template.py src/models/dynamic/overlap/my_temporal.py
```

Edit the file:

```python
from typing import List

import networkx as nx
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.algorithms.registry import register
from src.core.temporal_graph import TemporalGraph


@register(
    name="my_temporal",
    algo_type="dynamic",
    clustering_type="overlapping",
    default_params={"window": 3},
    description="My temporal community detection method",
)
class MyTemporal(CommunityDetectionAlgorithm):
    def __init__(self, window: int = 3):
        self.window = window

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        results = []
        for i, snapshot in enumerate(tg.iter_snapshots()):
            if i > 0:
                changes = tg.steps[i - 1]
                # changes.insertions — list of (u, v) tuples added
                # changes.deletions  — list of (u, v) tuples removed
            else:
                changes = None

            communities = self._detect(snapshot, changes)
            nc = NodeClustering(
                communities=communities,
                graph=snapshot,
                method_name="my_temporal",
            )
            results.append(nc)
        return results

    def _detect(self, graph: nx.Graph, changes) -> List[List]:
        # Your incremental algorithm here.
        ...
```

Key points:

- `algo_type="dynamic"` tells the factory this consumes the full `TemporalGraph`.
- Access temporal changes via `tg.steps[i-1]` (`.insertions` and `.deletions`).
- You can also return `MethodDynamicResults` instead of `List[NodeClustering]` to report custom per-step runtimes and traces.

---

## Path 2: Plain function

For simple methods that operate on a single graph and already return a `NodeClustering`, you can skip the class entirely:

```python
from src.algorithms.registry import register


@register(
    name="my_simple_method",
    algo_type="static",
    clustering_type="overlapping",
    default_params={"alpha": 0.8},
    description="Simple overlapping method",
)
def my_simple_method(graph, alpha=0.8):
    # graph is a networkx.Graph
    # Must return a cdlib.NodeClustering
    from cdlib import NodeClustering
    communities = [[n for n in graph.nodes()]]
    return NodeClustering(communities=communities, graph=graph, method_name="my_simple_method")
```

The factory wraps plain functions in `StaticMethodWrapper`, which handles per-snapshot iteration automatically. This is the same mechanism used for CDlib algorithms in `src/algorithms/cdlib_adapters.py`.

---

## Registering with the factory

After implementing your algorithm, add its module to `_REGISTRATION_MODULES` in [`src/algorithms/factory.py`](../src/algorithms/factory.py):

```python
_REGISTRATION_MODULES = [
    # Custom algorithms
    "src.models.static.overlap.big_clam",
    "src.models.static.overlap.copra",
    ...
    "src.models.static.overlap.my_algorithm",    # <-- add this
    # CDlib adapters
    "src.algorithms.cdlib_adapters",
]
```

This ensures your `@register` decorator executes when the factory loads.

## Activating in YAML

Add the algorithm name to the appropriate target list in [`config/algorithms.yaml`](../config/algorithms.yaml):

```yaml
target_snapshot_algorithms:
  - my_algorithm

target_temporal_algorithms:
  - my_temporal
```

Optionally override default parameters:

```yaml
algorithm_params:
  my_algorithm:
    threshold: 0.7
    iterations: 20
```

Parameters under `algorithm_params` are merged on top of the `default_params` from your `@register` decorator. Only list values you want to change.

---

## Verification

### 1. Check that your algorithm is in the registry

```bash
python -c "
from src.algorithms.factory import _ensure_registrations
_ensure_registrations()
from src.algorithms.registry import ALGORITHM_REGISTRY
print(sorted(ALGORITHM_REGISTRY.keys()))
"
```

Your algorithm name should appear in the list.

### 2. Check that YAML loading works

```bash
python -c "
from src.algorithms.factory import load_algorithms
algos = load_algorithms('config/algorithms.yaml')
print(sorted(algos.keys()))
"
```

### 3. Run a small benchmark

```bash
# Temporal benchmark
python main.py --dataset-path ./data/CollegeMsg.txt --dataset CollegeMsg --max-steps 5

# Static benchmark
python main_static.py --builtin karate --num-runs 1
```

### 4. Fail-fast behavior

If you add a name to the YAML target list that doesn't match any `@register` call, loading will raise a `ValueError` listing all available algorithm names:

```text
ValueError: Unknown algorithm 'typo_name' in YAML config. Not found in registry.
Available: angel, big_clam, coach, copra, ...
```

---

## The `@register` decorator reference

```python
@register(
    name="...",            # Unique name. Must match YAML references.
    algo_type="...",       # "static" or "dynamic"
    clustering_type="...", # "crisp" or "overlapping"
    default_params={...},  # Default kwargs passed to __init__ or the function
    description="...",     # Short human-readable label
)
```

| Parameter | Required | Values |
| --- | --- | --- |
| `name` | yes | Any unique string |
| `algo_type` | yes | `"static"` or `"dynamic"` |
| `clustering_type` | yes | `"crisp"` or `"overlapping"` |
| `default_params` | no | Dict of default keyword arguments (default: `{}`) |
| `description` | no | Human-readable description (default: `""`) |

The decorator stores an `AlgorithmSpec` in the global `ALGORITHM_REGISTRY` dict. The factory reads this registry at load time.

---

## Temporal data model reference

When implementing dynamic algorithms, these are the key objects:

| Object | Description |
| --- | --- |
| `tg.base_graph` | NetworkX graph at t=0 |
| `tg.steps` | List of `TemporalChanges` objects |
| `tg.steps[i].insertions` | List of `(u, v)` edge tuples added at step i |
| `tg.steps[i].deletions` | List of `(u, v)` edge tuples removed at step i |
| `tg.iter_snapshots()` | Yields snapshot graphs in order (t=0 ... t=T) |
| `tg[i]` | Snapshot graph at time i |
| `len(tg)` | Total number of snapshots (`len(steps) + 1`) |

---

## Quick checklist

- [ ] Algorithm file created in `src/models/static/` or `src/models/dynamic/`
- [ ] `@register(...)` decorator with correct `name`, `algo_type`, `clustering_type`
- [ ] `default_params` keys match constructor/function parameters
- [ ] Module path added to `_REGISTRATION_MODULES` in `src/algorithms/factory.py`
- [ ] Algorithm name added to target list in `config/algorithms.yaml`
- [ ] Registry verification passes
- [ ] Small benchmark runs without errors

## Further reading

- [Architecture](architecture.md) — system overview and component descriptions
- [Configuration Guide](configuration.md) — YAML config reference
- [Development Guide](development_guide.md) — datasets, metrics, and general development
- [Metrics Documentation](metrics.md) — what gets evaluated and logged
