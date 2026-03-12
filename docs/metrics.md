# Metrics Documentation

The benchmark evaluates each algorithm per snapshot and logs both summary and step-level metrics to Comet ML.

## Core Metrics

| Metric | Meaning |
| --- | --- |
| `runtime` | Execution time for a snapshot or temporal step |
| `cdlib_modularity` | CDlib overlap modularity for overlapping methods, or Newman-Girvan modularity for crisp methods |
| `customize_q0_modularity` | Custom Q0 modularity for overlapping methods, `0.0` for crisp methods |
| `num_communities` | Number of detected communities |
| `nmi` | Ground-truth agreement when LFR labels are available |

Internally, the traces are stored on [`src/factory/communities.py`](../src/factory/communities.py).

## Crisp vs Overlapping Evaluation

The modularity dispatcher in [`src/evaluations/metrics.py`](../src/evaluations/metrics.py) uses `clustering_type` from the algorithm config.

### Crisp

- `cdlib_modularity`: Newman-Girvan modularity via NetworkX
- `customize_q0_modularity`: `0.0`

### Overlapping

- `cdlib_modularity`: `cdlib.evaluation.modularity_overlap(...).score`
- `customize_q0_modularity`: custom Q0 implementation from [`src/evaluations/target_modularity.py`](../src/evaluations/target_modularity.py)

## Runtime

Runtime is recorded once per snapshot result.

- Static wrappers measure total algorithm time and distribute it evenly across the returned snapshots.
- Dynamic implementations can return their own per-step runtimes through `MethodDynamicResults`.

See [`src/pipeline_utils.py`](../src/pipeline_utils.py).

## Ground-Truth Evaluation

When the loader provides `_ground_truth_clusterings`, evaluation adds NMI.

### Supported ground-truth formats

- Crisp labels: integer-like value per node, for example `1`
- Overlapping labels: comma-separated memberships, for example `1,3,8`

### Command example

```bash
python main.py \
  --lfr-folder ./data/synthetic_n_5000_1 \
  --ground-truth-attr communities \
  --max-steps 10

python main_static.py --builtin karate --num-runs 1
```

### NMI behavior

- Crisp ground truth uses standard normalized mutual information.
- Overlapping ground truth uses a local high-performance implementation of the MGH overlapping NMI formula, matched against CDlib for score compatibility.

## What Gets Logged To Comet

Summary metrics include:

- `avg_runtime`
- `total_runtime`
- `avg_cdlib_modularity_overlap`
- `avg_customize_q0_overlap`
- stability ranges for both modularity traces
- `avg_nmi` when available

Per-step metrics include:

- `runtime`
- `cdlib_modularity`
- `customize_q0_modularity`
- `num_communities`
- `nmi` when available

## Fast ONMI Path

Overlapping ground-truth evaluation is routed through [`src/evaluations/onmi_fast.py`](../src/evaluations/onmi_fast.py) and called from [`src/evaluations/metrics.py`](../src/evaluations/metrics.py).

- It preserves CDlib-compatible MGH scoring behavior.
- It uses vectorized entropy computations and sparse membership matrices for speed.
- [`src/evaluations/onmi.py`](../src/evaluations/onmi.py) now remains only as a compatibility wrapper.

## Adding A New Metric

1. Implement the metric in `src/evaluations/`.
2. Extend evaluation in [`src/pipeline_utils.py`](../src/pipeline_utils.py).
3. Add storage fields to [`src/factory/communities.py`](../src/factory/communities.py) if needed.
4. Add the metric name to [`config/visualization_dynamic.yaml`](../config/visualization_dynamic.yaml) and/or [`config/visualization_static.yaml`](../config/visualization_static.yaml) if you want it fetched and plotted.

## References

- Newman and Girvan modularity
- CDlib modularity and NMI implementations
- Custom Q0 implementation in [`src/evaluations/target_modularity.py`](../src/evaluations/target_modularity.py)
