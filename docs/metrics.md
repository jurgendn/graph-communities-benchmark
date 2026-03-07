# Metrics Documentation

This guide describes the evaluation metrics used in the Graph Communities Benchmark framework.

## Overview

The benchmark tracks multiple metrics for each algorithm to evaluate their performance on temporal graph snapshots. All metrics are logged to Comet ML for visualization and comparison.

## Supported Metrics

| Metric | Description | Applicable To |
|--------|-------------|---------------|
| **Runtime** | Execution time per snapshot (seconds) | All |
| **CDlib Modularity** | Girvan-Newman (crisp) or overlapping modularity | All |
| **Custom Q0 Modularity** | Custom Q0 (overlapping) or 0 (crisp) | All |
| **Number of Communities** | Communities detected per snapshot | All |
| **Affected Nodes** | Number of nodes in each snapshot | All |

## Modularity Calculation by Clustering Type

The framework automatically selects the appropriate modularity metric based on the algorithm's `clustering_type`:

### Crisp Algorithms

For **crisp** community detection (each node belongs to exactly one community):
- **CDlib Modularity**: Girvan-Newman modularity (NetworkX `community.modularity`)
- **Custom Q0**: Set to `0` (not applicable for crisp partitions)

### Overlapping Algorithms

For **overlapping** community detection (nodes can belong to multiple communities):
- **CDlib Modularity**: CDlib's `modularity_overlap` score
- **Custom Q0**: Optimized Q0 implementation with Numba acceleration

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

## Ground Truth Evaluation (NMI)

For benchmark graphs with known ground truth communities (e.g., LFR graphs), 
the framework computes Normalized Mutual Information (NMI) to measure how well detected communities match the ground truth.

### Overview

When ground truth is available (via node attribute), 
the framework calculates:

| Metric | Description | Clustering Type |
|--------|-------------|-----------------|
| **NMI** | Standard Normalized Mutual Information | Crisp |
| **ONMI-MGH** | Overlapping NMI (McDaid-Greene-Hurley) | Overlapping |

### Ground Truth Format

The ground truth is stored in a node attribute (configurable, default: `label`):

| Type | Attribute Value | Example |
|------|-----------|-------|---------|
| **Crisp** | Single int | `graph.nodes[0]['label'] = 1` |
| **Overlapping** | Comma-separated string | `graph.nodes[0]['label'] = '1,23,4,5'` |

### Usage

```bash
python main.py \
    --lfr-folder ./data/lfr_benchmark \
    --ground-truth-attr label \
    --max-steps 10
```

### NMI Metrics

- **NMI** (0-1): 1 = perfect match, 0 = no mutual information
- **ONMI-MGH** (0-1): Extended NMI for overlapping communities

Reference: McDaid, A. F., Greene, D., & Hurley, N. (2011). Normalized mutual information to evaluate overlapping community detection.
```

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
