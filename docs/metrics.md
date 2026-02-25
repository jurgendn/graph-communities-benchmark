# Metrics Documentation

This guide describes the evaluation metrics used in the Graph Communities Benchmark framework.

## Overview

The benchmark tracks multiple metrics for each algorithm to evaluate their performance on temporal graph snapshots. All metrics are logged to Comet ML for visualization and comparison.

## Supported Metrics

| Metric | Description | Type |
|--------|-------------|------|
| **Runtime** | Execution time per snapshot (seconds) | Performance |
| **CDlib Modularity** | Overlapping modularity score from CDlib | Quality |
| **Custom Q0 Modularity** | Optimized overlapping modularity with Numba | Quality |
| **Number of Communities** | Communities detected per snapshot | Structure |
| **Affected Nodes** | Number of nodes in each snapshot | Scale |

## Detailed Metrics

### Runtime

**Description**: Execution time per snapshot in seconds.

**Purpose**: Measures the computational efficiency of each algorithm.

**Calculation**: Time difference between algorithm start and end for each snapshot.

**Usage**: Compare algorithm performance and scalability across different graph sizes.

### CDlib Modularity

**Description**: Overlapping modularity score calculated using CDlib's implementation.

**Purpose**: Measures the quality of detected communities based on edge density.

**Formula**: Standard Newman-Girvan modularity adapted for overlapping communities.

**Range**: Typically between -1 and 1, with higher values indicating better community structure.

**Reference**: See [CDlib modularity documentation](https://cdlib.readthedocs.io/en/latest/reference/cdlib.classes.html#cdlib.classes.NodeClustering)

### Custom Q0 Modularity

**Description**: Optimized overlapping modularity implementation with Numba acceleration.

**Purpose**: Provides a fast, accurate modularity calculation for overlapping communities.

**Formula**:
```
Q = (1/2m) * sum_{C_j} sum_{u,v in C_j} (A_uv - d_u*d_v/2m) * f(alpha_uC, alpha_vC)
```

where `f(alpha_u, alpha_v) = (alpha_u + alpha_v) / 2`

**Key Properties**:
- Bounded in [-1, 1]
- Ensures `sum_{C_j} alpha_{u,C_j} = 1` for all nodes u
- Numba-accelerated for parallel computation on large graphs

**Implementation**: See [`src/evaluations/target_modularity.py`](../src/evaluations/target_modularity.py)

**Performance**:
- Uses sparse matrix operations for memory efficiency
- Parallel computation with Numba's `prange`
- Optional Numba acceleration (falls back to pure Python if not available)

### Number of Communities

**Description**: Count of detected communities per snapshot.

**Purpose**: Measures the granularity of community detection.

**Usage**: Compare how algorithms partition the graph into communities.

**Interpretation**:
- Fewer communities: More coarse-grained partitioning
- More communities: More fine-grained partitioning

### Affected Nodes

**Description**: Number of nodes present in each snapshot.

**Purpose**: Tracks the scale of each temporal snapshot.

**Usage**: Monitor graph evolution and ensure consistent processing across snapshots.

## Modularity Stability

**Description**: Range of modularity values across all snapshots.

**Calculation**: `max(modularity) - min(modularity)`

**Purpose**: Measures the consistency of algorithm performance across temporal evolution.

**Interpretation**:
- Lower values: More consistent performance
- Higher values: Performance varies significantly across snapshots

## Metric Logging

All metrics are logged to Comet ML with the following structure:

### Summary Metrics

- Average runtime across all snapshots
- Total runtime
- Average modularity (both CDlib and Custom Q0)
- Modularity stability

### Step Metrics

Per-snapshot metrics:
- Runtime
- CDlib modularity
- Custom Q0 modularity
- Number of communities
- Affected nodes

## Adding Custom Metrics

To add a new evaluation metric:

### Step 1: Implement the Metric

Create a function in [`src/evaluations/`](../src/evaluations/):

```python
import networkx as nx
from cdlib import NodeClustering

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

Add your metric to [`src/benchmark.py`](../src/benchmark.py):

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

Add your metric to [`config/visualization.yaml`](../config/visualization.yaml):

```yaml
metric_keys:
  - "cdlib_modularity"
  - "customize_q0_modularity"
  - "num_communities"
  - "runtime"
  - "your_metric"  # Add your metric here
```

## Numba Acceleration

The Custom Q0 Modularity implementation uses Numba for performance optimization:

### Requirements

```bash
pip install numba
```

### Benefits

- 10-100x speedup for large graphs
- Parallel computation across CPU cores
- Minimal code changes required

### Disabling Numba

If Numba causes issues, you can disable it by uninstalling:

```bash
pip uninstall numba
```

The implementation will fall back to pure Python automatically.

## References

1. Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure in networks. *Physical review E*, 69(2), 026113.
2. Lancichinetti, A., & Fortunato, S. (2009). Community detection algorithms: a comparative analysis. *Physical review E*, 80(5), 056117.
3. CDlib Documentation: https://cdlib.readthedocs.io/
