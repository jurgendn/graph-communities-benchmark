# Visualization Guide

The repository has two separate visualization paths:

1. local graph/community visualization for a single snapshot
2. experiment-metric visualization from Comet ML runs

## 1. Community Visualization

The graph visualizer lives in [`src/utils/visualize.py`](../src/utils/visualize.py).

It provides utilities to:

- generate a sample graph
- run a simple overlapping detector for demos
- compute a custom circular layout
- render overlapping nodes between community centers

The central class is `Visualizer`.

Example usage:

```python
import networkx as nx
from cdlib import algorithms

from src.utils.visualize import Visualizer

graph = nx.karate_club_graph()
clustering = algorithms.angel(graph, threshold=0.25)

viz = Visualizer(seed=42)
viz.visualize(graph, clustering)
```

Refer to the methods in [`src/utils/visualize.py`](../src/utils/visualize.py) for the available plotting helpers and parameters.

## 2. Experiment Metrics Visualization

The experiment pipeline reads Comet runs, stores raw JSON locally, merges runs by algorithm, and builds grouped plots.

### Recommended command

```bash
./scripts/plot.sh
```

That script runs:

```bash
PYTHONPATH=. python tools/fetch_and_merge.py
PYTHONPATH=. python tools/plots.py
```

### Fetch and merge only

```bash
PYTHONPATH=. python tools/fetch_and_merge.py
PYTHONPATH=. python tools/fetch_and_merge.py --force
PYTHONPATH=. python tools/fetch_and_merge.py --skip-merge
```

### Generate plots only

```bash
PYTHONPATH=. python tools/plots.py
PYTHONPATH=. python tools/plots.py --metric runtime
```

## Pipeline Components

### `Fetcher`

Implemented in [`src/visualization/data.py`](../src/visualization/data.py).

- connects to the Comet API
- fetches configured metrics for each configured project
- caches fetched experiment ids via `.fetched_experiments.json`
- writes raw JSON files to `experiments/raw/<project>/`

### `Merger`

Also in [`src/visualization/data.py`](../src/visualization/data.py).

- reads raw experiment JSON
- groups runs by algorithm name
- categorizes runs by batch range
- writes merged per-metric JSON to `experiments/merged/<project>/<batch-range>/`

### `GroupedPlotter`

Implemented in [`src/visualization/core.py`](../src/visualization/core.py).

- creates grouped dataset figures
- supports modularity box/bar/line plots
- supports runtime bar plots
- supports time-series plots for metrics such as `num_communities`

## Configuration

Visualization behavior is controlled by [`config/visualization.yaml`](../config/visualization.yaml).

Important sections:

- `workspace`
- `metric_keys`
- `projects`
- `plotter.common_plotter_settings`
- `plotter.grouped_plotter`

## Output Locations

- raw exports: `experiments/raw/`
- merged metric files: `experiments/merged/`
- generated figures: `assets/grouped/`

## Common Issues

### `No module named 'src'`

Run the tools with `PYTHONPATH=.` or use `./scripts/plot.sh`.

### No experiments found

- verify `COMET_API_KEY`
- verify `COMET_WORKSPACE`
- verify project names in [`config/visualization.yaml`](../config/visualization.yaml)

### Missing plots for a dataset

- confirm merged files exist under `experiments/merged/<project>/`
- confirm the dataset is included in the configured plot groups
- confirm the metric name exists in `metric_keys`
