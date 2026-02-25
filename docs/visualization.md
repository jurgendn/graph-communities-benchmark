# Visualization Guide

This guide describes the visualization capabilities of the Graph Communities Benchmark framework.

## Overview

The benchmark includes a comprehensive visualization system with two components:

1. **Community Network Visualization**: Visualize overlapping communities detected by algorithms
2. **Experiment Metrics Visualization**: Generate comparative plots for benchmark metrics

## 1. Community Network Visualization

Visualize overlapping communities detected by algorithms in the network.

### Usage

```python
from src.utils.visualize import visualize_communities

# After running community detection:
result = algorithms.angel(graph, threshold=0.25)
visualize_communities(graph, result.communities)
```

### Features

- **Circular Layout**: Communities arranged around cluster centers with nodes positioned near their respective clusters
- **Overlap Visualization**: Overlapping nodes highlighted with red edges and labeled with their membership count
- **Color Coding**: Up to 20 colors (tab20 colormap) for easy community distinction
- **Interactive Legend**: Shows community indices and overlap indicators
- **Edge Visualization**: Network edges displayed with transparency to reveal structure

### Example Output

The visualization generates a matplotlib figure showing:
- Nodes colored by their primary community
- Red edges connecting overlapping nodes
- Labels showing membership count for nodes in multiple communities
- A legend with community colors

### Customization

You can customize the visualization by modifying [`src/utils/visualize.py`](../src/utils/visualize.py):

```python
visualize_communities(
    graph, 
    communities,
    figsize=(12, 12),      # Figure size
    node_size=100,         # Node size
    edge_alpha=0.1,        # Edge transparency
    show_legend=True,       # Show legend
    save_path=None          # Path to save figure
)
```

## 2. Experiment Metrics Visualization

Generate comparative plots for benchmark metrics across experiments using data from Comet ML.

### Workflow

#### Step 1: Fetch and Merge Data

Retrieve all experiments from Comet ML workspace and merge metrics:

```bash
# Using the plot script (recommended)
./scripts/plot.sh

# Or using Python directly
PYTHONPATH=. python3 tools/fetch_and_merge.py
```

**Options**:
```bash
python tools/fetch_and_merge.py              # Fetch only new experiments
python tools/fetch_and_merge.py --force      # Re-fetch all experiments
python tools/fetch_and_merge.py --skip-merge # Only fetch, skip merging
```

#### Step 2: Generate Plots

Create grouped comparison plots:

```bash
PYTHONPATH=. python3 tools/plots.py
```

### Data Pipeline

The visualization system uses a three-stage pipeline:

#### Fetcher (`src.visualization.data.Fetcher`)

- Connects to Comet ML API using credentials from `.env`
- Downloads all experiments for configured projects
- Extracts parameters and metrics
- Saves raw data to `experiments/raw/`

#### Merger (`src.visualization.data.Merger`)

- Reads raw experiment files
- Groups metrics by algorithm name
- Normalizes metric values and step indices
- Aggregates multiple runs per algorithm
- Outputs merged structure to `experiments/merged/`

#### GroupedPlotter (`src.visualization.core.GroupedPlotter`)

- Loads merged metrics
- Generates comparison plots (line, box, bar charts)
- Supports per-dataset and cross-dataset analysis
- Saves results to `assets/grouped/`

### Output Structure

```
assets/grouped/
├── per_datasets/
│   └── {project_name}/
│       ├── {metric_name}_line.png        # Line plot with multiple runs
│       ├── {metric_name}_box.png         # Box plot comparing algorithms
│       └── {metric_name}_bar.png         # Bar chart of aggregated values
└── merged_all/
    └── {metric_name}_{type}.png         # Cross-dataset comparison plots
```

### Supported Metrics

- `num_communities` - Number of detected communities per snapshot
- `cdlib_modularity` - Overlapping modularity score
- `customize_q0_modularity` - Custom modularity variant
- `runtime` - Execution time per snapshot

### Configuration

Visualization settings are configured in [`config/visualization.yaml`](../config/visualization.yaml):

```yaml
raw_dir: "experiments/raw"
merge_dir: "experiments/merged"
output_dir: "assets"

workspace: "your_workspace"

metric_keys:
  - "cdlib_modularity"
  - "customize_q0_modularity"
  - "num_communities"
  - "runtime"

projects:
  - project-name-1
  - project-name-2

plotter:
  selected_algorithms:
    - coach
    - tiles
    - core_expansion
  
  methods_name:
    coach: CoAcH
    tiles: Tiles
    core_expansion: Core Expansion
  
  colors:
    coach: "#1B4F72"
    tiles: "#17A2B8"
    core_expansion: "#E09F3E"
```

### Plot Types

#### Line Plots

Shows metric values over time (snapshot steps) for each algorithm.

**Use case**: Track algorithm performance across temporal evolution.

#### Box Plots

Shows distribution of metric values for each algorithm across all runs.

**Use case**: Compare algorithm performance and identify outliers.

#### Bar Charts

Shows aggregated metric values (mean, median, etc.) for each algorithm.

**Use case**: Quick comparison of overall algorithm performance.

### Customizing Plots

#### Adding New Algorithms

Add to [`config/visualization.yaml`](../config/visualization.yaml):

```yaml
plotter:
  selected_algorithms:
    - coach
    - your_algorithm  # Add your algorithm
  
  methods_name:
    your_algorithm: "Your Algorithm Name"
  
  colors:
    your_algorithm: "#FF5733"
```

#### Changing Plot Styles

Modify [`src/visualization/styles.py`](../src/visualization/styles.py) to customize:

- Line styles (solid, dashed, dotted)
- Marker styles (circle, square, triangle)
- Font sizes
- Color palettes

#### Adding New Metrics

See [Metrics Documentation](metrics.md) for adding new evaluation metrics.

## Troubleshooting

### No Experiments Found

If you get "No experiments found" error:

1. Verify your Comet ML credentials in `.env`
2. Check that the project name in [`config/visualization.yaml`](../config/visualization.yaml) is correct
3. Ensure you've run benchmarks and experiments are logged to Comet ML

### Plot Generation Fails

If plot generation fails:

1. Check that merged data exists in `experiments/merged/`
2. Verify the metric names in [`config/visualization.yaml`](../config/visualization.yaml) match logged metrics
3. Check for missing or corrupted data files

### Colors Not Displaying Correctly

If colors don't display correctly:

1. Verify color codes in [`config/visualization.yaml`](../config/visualization.yaml) are valid hex codes
2. Check that algorithm names match between `selected_algorithms` and `methods_name`

## Advanced Usage

### Custom Plot Types

To create custom plot types, extend the `Plot` class in [`src/visualization/core.py`](../src/visualization/core.py):

```python
class CustomPlot(Plot):
    def generate(self, data, output_path):
        # Your custom plot logic
        fig, ax = plt.subplots()
        # ... plot code ...
        fig.savefig(output_path)
```

### Filtering Data

To filter experiments by parameters:

```python
from src.visualization.data import Fetcher

fetcher = Fetcher()
fetcher.fetch()
# Filter by batch_range parameter
filtered = [exp for exp in fetcher.experiments 
             if exp.params.get('batch_range') == 0.0001]
```

## References

- Matplotlib Documentation: https://matplotlib.org/
- Comet ML Documentation: https://www.comet.com/docs/
