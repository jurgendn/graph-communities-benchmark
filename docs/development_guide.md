# Development Guide

This guide provides detailed instructions for extending and customizing the Graph Communities Benchmark framework.

## Table of Contents

- [Adding New Algorithms](#adding-new-algorithms)
  - [Adding a Custom Static Algorithm](#adding-a-custom-static-algorithm)
  - [Adding a Custom Dynamic Algorithm](#adding-a-custom-dynamic-algorithm)
  - [Using Existing CDlib Algorithms](#using-existing-cdlib-algorithms)
- [Adding New Datasets](#adding-new-datasets)
- [Adding New Evaluation Metrics](#adding-new-evaluation-metrics)
- [Configuration System](#configuration-system)

---

## Adding New Algorithms

This section guides you through adding new community detection algorithms to the benchmark framework.

### Unified API

All algorithms must implement the `CommunityDetectionAlgorithm` interface:

```python
from abc import ABC, abstractmethod
from typing import List
from cdlib import NodeClustering
from src.factory.factory import TemporalGraph

class CommunityDetectionAlgorithm(ABC):
    @abstractmethod
    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """Run community detection on a temporal graph."""
        pass
```

### Adding a Custom Static Algorithm

A ready-to-use template is provided at [`templates/static_algorithm_template.py`](../templates/static_algorithm_template.py). Copy it and follow the steps below.

#### Step 1: Create the Algorithm Class

Create a new Python file in [`src/models/static_methods/overlap/`](../src/models/static_methods/overlap/) or [`src/models/static_methods/crisp/`](../src/models/static_methods/crisp/) depending on whether it's overlapping or crisp. The class should inherit from `CommunityDetectionAlgorithm`:

```python
"""Your Algorithm: Description of your algorithm."""

from typing import List
import networkx as nx
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.factory import TemporalGraph


class YourAlgorithm(CommunityDetectionAlgorithm):
    """Detect overlapping communities using your method."""

    def __init__(
        self,
        param1: float = 0.5,
        param2: int = 100,
    ):
        """Initialize YourAlgorithm parameters."""
        self.param1 = param1
        self.param2 = param2

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """
        Run the algorithm on each snapshot of the temporal graph.

        Args:
            tg: TemporalGraph with snapshots

        Returns:
            List[NodeClustering], one per snapshot
        """
        results = []
        for snapshot in tg.iter_snapshots():
            communities = self._detect_communities(snapshot)
            results.append(NodeClustering(communities, snapshot, "YourAlgorithm"))
        return results

    def _detect_communities(self, graph: nx.Graph) -> List[List]:
        """Internal method to detect communities."""
        # Your community detection logic here
        pass
```

#### Step 2: Register the Algorithm in Config

Add your algorithm to [`config/algorithms.yaml`](config/algorithms.yaml):

```yaml
algorithms:
  # ... existing algorithms ...

  your_algorithm:
    module: "src.models.static_methods.overlap.your_algorithm"
    function: "YourAlgorithm"
    params:
      param1: 0.5
      param2: 100
    type: "static"
    clustering_type: "overlapping"  # or "crisp"
    metadata: {}
    description: "Your Algorithm Description"
```

#### Step 3: Add to Target Algorithms

Add your algorithm to the `target_algorithms` list to enable it for benchmarks:

```yaml
target_algorithms:
  - coach
  - percomvc
  - your_algorithm  # Add your algorithm here
```

### Adding a Custom Dynamic Algorithm

### Adding a Custom Dynamic Algorithm

A ready-to-use template is provided at [`templates/dynamic_algorithm_template.py`](../templates/dynamic_algorithm_template.py). Copy it and follow the steps below.

#### Step 1: Create the Algorithm Class

Create a new Python file in [`src/models/dynamic_methods/overlap/`](../src/models/dynamic_methods/overlap/) or [`src/models/dynamic_methods/crisp/`](../src/models/dynamic_methods/crisp/) depending on whether it's overlapping or crisp. The class should inherit from `CommunityDetectionAlgorithm`:

```python
"""Your Dynamic Algorithm: Description of your algorithm."""

from typing import List
import networkx as nx
from cdlib import NodeClustering

from src.algorithms.base import CommunityDetectionAlgorithm
from src.factory.factory import TemporalGraph


class YourDynamicAlgorithm(CommunityDetectionAlgorithm):
    """Detect communities in dynamic/temporal graphs."""

    def __init__(self, obs: int = 1):
        """Initialize YourDynamicAlgorithm parameters."""
        self.obs = obs

    def __call__(self, tg: TemporalGraph) -> List[NodeClustering]:
        """
        Run the algorithm on the temporal graph.

        Args:
            tg: TemporalGraph object with snapshots

        Returns:
            List[NodeClustering], one per snapshot
        """
        results = []

        # Your implementation here
        for snapshot in tg.iter_snapshots():
            # Detect communities for this snapshot
            communities = self._detect_communities(snapshot)
            results.append(NodeClustering(communities, snapshot, "YourDynamicAlgorithm"))

        return results

    def _detect_communities(self, graph: nx.Graph) -> List[List]:
        """Internal method to detect communities."""
        # Your community detection logic here
        pass
```

#### Step 2: Register in Config

Add your algorithm to [`config/algorithms.yaml`](config/algorithms.yaml):

```yaml
algorithms:
  your_dynamic_algorithm:
    module: "src.models.dynamic_methods.overlap.your_dynamic_algorithm"
    function: "YourDynamicAlgorithm"
    params:
      obs: 1
    type: "dynamic"
    clustering_type: "overlapping"  # or "crisp"
    metadata: {}
    description: "Your Dynamic Algorithm Description"
```

#### Step 3: Add to Target Algorithms

Add your algorithm to the `target_algorithms` list:

```yaml
target_algorithms:
  - tiles
  - your_dynamic_algorithm
```

### Using Existing CDlib Algorithms

If you want to use an existing algorithm from the [CDlib library](https://cdlib.readthedocs.io/), simply add it to [`config/algorithms.yaml`](config/algorithms.yaml):

```yaml
algorithms:
  # Example: Adding a CDlib algorithm
  angel:
    module: "cdlib.algorithms"
    function: "angel"
    params:
      threshold: 0.25
    type: "static"
    clustering_type: "crisp"
    metadata: {}
    description: "ANGEL: A New Graph-based Entity Linking algorithm"
```

### Testing Your Algorithm

After adding your algorithm, test it by running a benchmark:

```bash
# Using main.py (unified entry point)
python main.py --dataset CollegeMsg --max-steps 10

# Or directly with Python
PYTHONPATH=. python main.py \
    --dataset-path ./data/CollegeMsg.txt \
    --dataset CollegeMsg \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 10
```

---

---

### Adding LFR Benchmark Datasets

The framework supports LFR (Lancichinetti-Fortunato-Radicchi) benchmark graphs with ground truth communities.

#### LFR Folder Structure

Place your LFR snapshot files in a folder:

```
data/lfr_benchmark/
├── snapshot_t0.graphml   # nx.Graph saved as GraphML with 'label' attribute
├── snapshot_t1.graphml
└── ...
```

Each GraphML file contains an `nx.Graph` object with ground truth in node attributes.

#### File Naming

The loader expects files named `snapshot_t{index}.graphml`. The index starts from 0.

#### Generating LFR Snapshots

Use the built-in LFR generator:

```python
from src.utils.lfr_generator.generator import (
    unweighted_undirected_lfr_graph,
    save_temporal_networkx_snapshots,
)

edges, memberships = unweighted_undirected_lfr_graph(
    num_nodes=500,
    average_k=10,
    mu=0.1,
    tau1=3.0,
    tau2=1.0,
    com_size_min=10,
    com_size_max=50,
    overlapping_nodes=0,
    overlap_membership=0,
    seed=42,
)

save_temporal_networkx_snapshots(
    snapshots=memberships,
    output_prefix="data/lfr_benchmark/my_lfr",
    format="graphml",
)
```

This creates `data/lfr_benchmark/my_lfr/snapshot_t0.graphml`, `snapshot_t1.graphml`, etc.

#### Running the Benchmark on LFR Data

```bash
python main.py \
    --lfr-folder ./data/lfr_benchmark/my_lfr \
    --ground-truth-attr label \
    --max-steps 10
```

#### Ground Truth Attribute

- **Crisp**: Node attribute as integer, e.g., `graph.nodes[0]['label'] = 1`
- **Overlapping**: Node attribute as comma-separated string, e.g., `graph.nodes[0]['label'] = '1,23,4,5'`

---

## Adding New Datasets

To add a new dataset to the benchmark:

### Step 1: Prepare the Dataset File

Place your dataset file in the `data/` directory. The file should be an edge-list text file with the following format:

```
<source_node> <target_node> [optional_attributes...]
```

Lines starting with `//` are treated as comments and skipped.

### Step 2: Add to Dataset Config

Add your dataset to [`config/dataset_config.yaml`](config/dataset_config.yaml):

```yaml
datasets:
  your_dataset:
    path: ./data/your_dataset.txt
    dataset_name: YourDataset
    source_idx: 0
    target_idx: 1
    delimiter: " "  # Use "," for CSV, "\t" for TSV, etc.
    max_steps: 9
    initial_fraction: 0.4
    batch_range: 0.00001
```

### Step 3: Add to Target Datasets

Add your dataset to the `target_datasets` list:

```yaml
target_datasets:
  - college-msg
  - your_dataset  # Add your dataset here
```

---

## Adding New Evaluation Metrics

To add a new evaluation metric:

### Step 1: Implement the Metric

Create a new function in [`src/evaluations/`](src/evaluations/) or add to an existing file. The function should accept a graph and communities and return a numeric score.

Example (see [`src/evaluations/target_modularity.py`](src/evaluations/target_modularity.py)):

```python
def your_metric(graph: nx.Graph, communities: NodeClustering) -> float:
    """
    Calculate your custom metric.
    
    Args:
        graph: NetworkX graph
        communities: NodeClustering object
        
    Returns:
        Metric score
    """
    # Your implementation here
    return score
```

### Step 2: Register in the Pipeline

Add your metric to the `evaluate()` function in [`src/pipeline_utils.py`](../src/pipeline_utils.py).

For each snapshot, compute your metric after the existing modularity call and
store the trace on `results`:

```python
# src/pipeline_utils.py — inside evaluate(), in the per-snapshot loop
your_metric_score = your_metric(snapshot, communities)

# Append to a trace list on results (add this attribute to MethodDynamicResults too)
results.your_metric_trace.append(your_metric_score)
```

Also add the corresponding field and summary property to
`MethodDynamicResults` in [`src/factory/communities.py`](../src/factory/communities.py):

```python
class MethodDynamicResults(BaseModel):
    # ... existing fields ...
    your_metric_trace: List[float] = []

    @property
    def avg_your_metric(self) -> float:
        if not self.your_metric_trace:
            return 0.0
        return sum(self.your_metric_trace) / len(self.your_metric_trace)
```

Finally, log it in `log_results()` (also in `src/pipeline_utils.py`):

```python
# Summary metric
summary_metrics["avg_your_metric"] = results.avg_your_metric

# Per-step metric
step_metrics["your_metric"] = results.your_metric_trace[step]
```

### Step 3: Add to Visualization Config

Add your metric to [`config/visualization.yaml`](config/visualization.yaml):

```yaml
metric_keys:
  - "cdlib_modularity"
  - "customize_q0_modularity"
  - "num_communities"
  - "runtime"
  - "your_metric"  # Add your metric here
```

---

## Configuration System

The benchmark uses YAML configuration files for easy customization:

### Algorithm Configuration ([`config/algorithms.yaml`](config/algorithms.yaml))

- `target_algorithms`: List of algorithms to run in benchmarks
- `algorithms`: All available algorithms with their configurations
- Each algorithm entry includes:
  - `module`: Python module path
  - `function`: Function or class name
  - `params`: Algorithm parameters
  - `type`: Algorithm type ("static" or "dynamic")
  - `clustering_type`: Clustering type ("crisp" or "overlapping")
  - `metadata`: Additional metadata
  - `description`: Algorithm description

### Dataset Configuration ([`config/dataset_config.yaml`](config/dataset_config.yaml))

- `target_datasets`: List of datasets to benchmark
- `datasets`: All available datasets with their configurations
- `common`: Common settings shared across datasets
- Each dataset entry includes:
  - `path`: Path to dataset file
  - `dataset_name`: Dataset name for tagging
  - `source_idx`: Column index for source node
  - `target_idx`: Column index for target node
  - `delimiter`: Field delimiter
  - `max_steps`: Maximum number of snapshots
  - `initial_fraction`: Fraction of edges for base graph
  - `batch_range`: Fraction of edges per batch

### Visualization Configuration ([`config/visualization.yaml`](config/visualization.yaml))

- `raw_dir`: Directory for raw experiment data
- `merge_dir`: Directory for merged experiment data
- `output_dir`: Directory for generated plots
- `workspace`: Comet ML workspace name
- `projects`: List of Comet ML projects to fetch
- `metric_keys`: Metrics to track and visualize
- `plotter`: Visualization settings including colors, labels, and plot types
