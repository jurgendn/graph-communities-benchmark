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

### Adding a Custom Static Algorithm

To add a custom static overlapping community detection algorithm:

#### Step 1: Create the Algorithm Class

Create a new Python file in [`src/static_methods/`](src/static_methods/) with your algorithm implementation. The class should:

1. Accept a [`networkx.Graph`](https://networkx.org/documentation/stable/reference/classes/graph.html) as input
2. Return a [`cdlib.NodeClustering`](https://cdlib.readthedocs.io/en/latest/reference/classes.html#nodeclustering) object

Example implementation (see [`src/static_methods/big_clam.py`](src/static_methods/big_clam.py)):

```python
"""Your Algorithm: Description of your algorithm."""

import networkx as nx
from cdlib import NodeClustering

class YourAlgorithm:
    """Detect overlapping communities using your method."""
    
    def __init__(
        self,
        param1: float = 0.5,
        param2: int = 100,
        seed: int = 123
    ):
        """
        Initialize YourAlgorithm parameters.
        
        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2
            seed: Random seed for reproducibility
        """
        self.param1 = param1
        self.param2 = param2
        self.seed = seed
    
    def __call__(self, graph: nx.Graph) -> NodeClustering:
        """
        Run the algorithm on the input graph.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            NodeClustering object containing detected communities
        """
        # Your implementation here
        # 1. Detect communities
        communities = self._detect_communities(graph)
        
        # 2. Return as NodeClustering object
        return NodeClustering(communities, graph, "YourAlgorithm")
    
    def _detect_communities(self, graph: nx.Graph) -> list[set]:
        """Internal method to detect communities."""
        # Your community detection logic here
        # Return a list of sets, where each set contains node IDs
        pass
```

#### Step 2: Register the Algorithm in Config

Add your algorithm to [`config/algorithms.yaml`](config/algorithms.yaml):

```yaml
overlapping_algorithms:
  # ... existing algorithms ...
  
  your_algorithm:
    module: "src.static_methods.your_algorithm"  # Path to your module
    function: "YourAlgorithm"                     # Class name
    params:
      param1: 0.5
      param2: 100
      seed: 123
    metadata: {}
    description: "Your Algorithm: Brief description"
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

To add a custom dynamic community detection algorithm:

#### Step 1: Create the Algorithm Class

Create a new Python file in [`src/dynamic_methods/`](src/dynamic_methods/) with your algorithm implementation. The class should:

1. Accept a [`TemporalGraph`](src/factory/factory.py) object as input
2. Return a [`MethodDynamicResults`](src/factory/communities.py) object

Example implementation (see [`src/dynamic_methods/tiles.py`](src/dynamic_methods/tiles.py)):

```python
"""Your Dynamic Algorithm: Description of your algorithm."""

from src.factory.communities import MethodDynamicResults
from src.factory.factory import TemporalGraph

class YourDynamicAlgorithm:
    """Detect communities in dynamic/temporal graphs."""
    
    def __init__(self, obs: int = 1):
        """
        Initialize YourDynamicAlgorithm parameters.
        
        Args:
            obs: Observation window size
        """
        self.obs = obs
    
    def __call__(self, tg: TemporalGraph) -> MethodDynamicResults:
        """
        Run the algorithm on the temporal graph.
        
        Args:
            tg: TemporalGraph object with snapshots
            
        Returns:
            MethodDynamicResults object containing detected communities
        """
        results = MethodDynamicResults()
        
        # Your implementation here
        for snapshot in tg.iter_snapshots():
            # Detect communities for this snapshot
            communities = self._detect_communities(snapshot)
            
            # Store results
            results.update_intermediate_results(
                runtime=0.0,  # Actual runtime
                cdlib_modularity_overlap=0.0,  # Actual modularity
                customize_q0_overlap=0.0,  # Actual modularity
                affected_nodes=snapshot.number_of_nodes(),
                num_communities=len(communities)
            )
        
        return results
    
    def _detect_communities(self, graph):
        """Internal method to detect communities."""
        # Your community detection logic here
        pass
```

#### Step 2: Register in Benchmark

Register your algorithm in the `run_dynamic_benchmark()` function in [`src/benchmark.py`](src/benchmark.py):

```python
def run_dynamic_benchmark(
    temporal_graph: TemporalGraph,
    algorithm_names: list[str] | None = None,
) -> dict[str, MethodDynamicResults]:
    ALGORITHMS_OVERLAPPING = {
        "tiles": {
            "func": dynamic_methods.Tiles(obs=1),
            "params": {},
        },
        "your_algorithm": {  # Add your algorithm here
            "func": dynamic_methods.YourDynamicAlgorithm(obs=1),
            "params": {},
        }
    }
    # ... rest of the function
```

### Using Existing CDlib Algorithms

If you want to use an existing algorithm from the [CDlib library](https://cdlib.readthedocs.io/), simply add it to [`config/algorithms.yaml`](config/algorithms.yaml):

```yaml
overlapping_algorithms:
  # Example: Adding a CDlib algorithm
  angel:
    module: "cdlib.algorithms"
    function: "angel"
    params:
      threshold: 0.25
    metadata: {}
    description: "ANGEL: A New Graph-based Entity Linking algorithm"
```

### Testing Your Algorithm

After adding your algorithm, test it by running a benchmark:

```bash
# Using the generic benchmark script
./scripts/benchmark.sh college-msg main_static.py

# Or directly with Python
PYTHONPATH=. python main_static.py \
    --dataset-path ./data/CollegeMsg.txt \
    --dataset CollegeMsg \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 10
```

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

### Step 2: Register in Benchmark

Add your metric to the benchmark runner in [`src/benchmark.py`](src/benchmark.py):

```python
# In run_static_benchmark()
for step_idx, snapshot in enumerate(steps_bar):
    # ... existing code ...
    
    your_metric_score = your_metric(snapshot, communities)
    
    intermediate = IntermediateResults(
        runtime=elapsed,
        cdlib_modularity_overlap=cdlib_modularity,
        customize_q0_overlap=q0_modularity,
        affected_nodes=snapshot.number_of_nodes(),
        num_communities=len(communities.communities),
        your_metric=your_metric_score,  # Add your metric
    )
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
- `overlapping_algorithms`: All available algorithms with their configurations
- Each algorithm entry includes:
  - `module`: Python module path
  - `function`: Function or class name
  - `params`: Algorithm parameters
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
