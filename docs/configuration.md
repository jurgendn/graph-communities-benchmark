# Configuration Guide

This guide describes the configuration system used in Graph Communities Benchmark framework.

## Overview

The benchmark uses YAML configuration files for easy customization of algorithms, datasets, and visualization settings. All configuration files are located in the [`config/`](../config/) directory.

## Configuration Files

| File | Purpose |
|-------|---------|
| [`config/algorithms.yaml`](../config/algorithms.yaml) | Algorithm definitions and parameters |
| [`config/dataset_config.yaml`](../config/dataset_config.yaml) | Dataset configurations and parameters |
| [`config/visualization.yaml`](../config/visualization.yaml) | Visualization and plotting settings |

## Algorithm Configuration ([`config/algorithms.yaml`](../config/algorithms.yaml))

### Structure

```yaml
# Specify which algorithms to run (in order)
target_algorithms:
  - coach
  - percomvc
  - core_expansion

# All available algorithms with their configurations
overlapping_algorithms:
  algorithm_name:
    module: "module.path"
    function: "function_name"
    params:
      param1: value1
      param2: value2
    metadata: {}
    description: "Algorithm description"
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `target_algorithms` | list | List of algorithm names to run in benchmarks |
| `overlapping_algorithms` | dict | All available algorithms with their configurations |
| `module` | str | Python module path (e.g., `cdlib.algorithms` or `src.static_methods.big_clam`) |
| `function` | str | Function or class name to import |
| `params` | dict | Algorithm parameters |
| `metadata` | dict | Additional metadata for the algorithm |
| `description` | str | Human-readable description |

### Example: Adding a CDlib Algorithm

```yaml
overlapping_algorithms:
  angel:
    module: "cdlib.algorithms"
    function: "angel"
    params:
      threshold: 0.25
    metadata: {}
    description: "ANGEL: A New Graph-based Entity Linking algorithm"
```

### Example: Adding a Custom Algorithm

```yaml
overlapping_algorithms:
  my_algorithm:
    module: "src.static_methods.my_algorithm"
    function: "MyAlgorithm"
    params:
      param1: 0.5
      param2: 100
    metadata:
      author: "Your Name"
      year: 2025
    description: "My Custom Algorithm for community detection"
```

### Available Algorithms

The following algorithms are currently configured:

- **ANGEL**: A fast, local-first overlapping community detection algorithm
- **DEMON**: Democratic Estimate of the Modular Organization of a Network
- **COACH**: Core-Attachment based clustering
- **NDOCD**: Network Decomposition-based Overlapping Community Detection
- **BigClam**: Overlapping community detection using gradient ascent
- **CosineOverlap**: Overlapping community detection using cosine similarity
- **SLPA**: Speaker-Listener Label Propagation Algorithm
- **Percomvc**: Permanence based Overlapping Community Detection
- **Core Expansion**: Core Expansion algorithm
- **Graph Entropy**: Graph Entropy based community detection
- **UMSTMO**: Universal Multi-Scale Community Detection
- **DPCLUS**: Density-Periphery based Clustering
- **IPCA**: Iterative Principal Component Analysis
- **LAIS2**: Label Propagation with Improved Seed Selection
- **Walkscan**: Walk-based SCAN algorithm
- **DCS**: Distributed Community Search
- **LFM**: Lancichinetti-Fortunato-Radicchi benchmark
- **EBGC**: Entropy-based Graph Clustering

## Dataset Configuration ([`config/dataset_config.yaml`](../config/dataset_config.yaml))

### Structure

```yaml
# Datasets to run benchmarks on
target_datasets:
  - college-msg
  - bio-wormnet-v3

# Common settings shared across datasets
common: &common_settings
  max_steps: 9
  initial_fraction: 0.4
  batch_range: 0.00001

# All available datasets with their configurations
datasets:
  dataset_name:
    path: ./data/dataset.txt
    dataset_name: DatasetName
    source_idx: 0
    target_idx: 1
    delimiter: " "
    <<: *common_settings
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `target_datasets` | list | List of dataset names to benchmark |
| `datasets` | dict | All available datasets with their configurations |
| `path` | str | Path to dataset file |
| `dataset_name` | str | Dataset name for tagging |
| `source_idx` | int | Column index for source node |
| `target_idx` | int | Column index for target node |
| `delimiter` | str | Field delimiter (e.g., `" "`, `","`, `"\t"`) |
| `max_steps` | int | Maximum number of temporal snapshots |
| `initial_fraction` | float | Fraction of edges for base graph |
| `batch_range` | float | Fraction of edges per batch |

### Common Settings

The `common` section defines settings that are shared across multiple datasets using YAML anchors:

```yaml
common: &common_settings
  max_steps: 9
  initial_fraction: 0.4
  batch_range: 0.00001

datasets:
  dataset1:
    <<: *common_settings  # Inherits all common settings
    path: ./data/dataset1.txt
    # ... other dataset-specific settings ...
```

### Example: Adding a New Dataset

```yaml
datasets:
  my_dataset:
    path: ./data/my_dataset.txt
    dataset_name: MyDataset
    source_idx: 0
    target_idx: 1
    delimiter: ","
    max_steps: 10
    initial_fraction: 0.3
    batch_range: 0.0001
```

### Dataset Categories

Batch ranges can be categorized by size:

| Category | Batch Range | Description |
|----------|-------------|-------------|
| Large | 0.001 | Large batches, fewer snapshots |
| Medium | 0.0001 | Medium batches |
| Small | 0.00001 | Small batches, more snapshots |

## Visualization Configuration ([`config/visualization.yaml`](../config/visualization.yaml))

### Structure

```yaml
# Directory paths
raw_dir: "experiments/raw"
merge_dir: "experiments/merged"
output_dir: "assets"

# Comet ML workspace
workspace: "your_workspace"

# Hyperparameters to track
hyperparameters:
  - batch-range

# Metrics to visualize
metric_keys:
  - "cdlib_modularity"
  - "customize_q0_modularity"
  - "num_communities"
  - "runtime"

# Comet ML projects
projects:
  - project-name-1
  - project-name-2

# Plotter settings
plotter:
  # Batch range categories
  batch_range_categories:
    large: 0.001
    medium: 0.0001
    small: 0.00001
  
  # Dataset categories
  synthetic_datasets:
    - project-lfr-large
    - project-lfr-small
  
  real_world_datasets:
    - project-collegemsg
    - project-bio-wormnet
  
  # Project name mapping
  projects_name_mapping:
    project-collegemsg: "CollegeMsg"
    project-bio-wormnet: "Bio-WormNet"
  
  # Selected algorithms
  selected_algorithms:
    - coach
    - tiles
    - core_expansion
  
  # Algorithm display names
  methods_name:
    coach: CoAcH
    tiles: Tiles
    core_expansion: Core Expansion
  
  # Algorithm colors
  colors:
    coach: "#1B4F72"
    tiles: "#17A2B8"
    core_expansion: "#E09F3E"
  
  # Plot order
  orders:
    - tiles
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `raw_dir` | str | Directory for raw experiment data |
| `merge_dir` | str | Directory for merged experiment data |
| `output_dir` | str | Directory for generated plots |
| `workspace` | str | Comet ML workspace name |
| `hyperparameters` | list | Hyperparameters to track |
| `metric_keys` | list | Metrics to visualize |
| `projects` | list | Comet ML projects to fetch |
| `plotter` | dict | Visualization settings |

### Plotter Settings

| Sub-field | Type | Description |
|-----------|------|-------------|
| `batch_range_categories` | dict | Batch range size categories |
| `synthetic_datasets` | list | Synthetic dataset project names |
| `real_world_datasets` | list | Real-world dataset project names |
| `projects_name_mapping` | dict | Project name to display name mapping |
| `selected_algorithms` | list | Algorithms to include in plots |
| `methods_name` | dict | Algorithm name to display name mapping |
| `colors` | dict | Algorithm colors (hex codes) |
| `orders` | list | Plot order for algorithms |

### Color Palette

The following colors are used for algorithms:

| Algorithm | Color | Hex |
|------------|--------|-----|
| CoAcH | Steel Blue | `#1B4F72` |
| Core Expansion | Muted Gold | `#E09F3E` |
| Graph Entropy | Sea Green | `#2E8B57` |
| SLPA | Purple | `#8E44AD` |
| UMSTMO | Brick Red | `#C0392B` |
| RWGP-DFL-Overlap | Vibrant Red | `#E74C3C` |
| Tiles | Vibrant Cyan | `#17A2B8` |

## Best Practices

### 1. Version Control

Keep configuration files under version control to ensure reproducibility:

```bash
git add config/
git commit -m "Update algorithm configuration"
```

### 2. Environment-Specific Settings

Use environment variables for sensitive information:

```bash
# .env
COMET_API_KEY=your_api_key
COMET_WORKSPACE=your_workspace
```

### 3. Documentation

Add comments to configuration files for clarity:

```yaml
# This algorithm performs best on small graphs
small_graph_algorithm:
  module: "cdlib.algorithms"
  function: "small_graph_algo"
  params:
    threshold: 0.25  # Lower threshold for small graphs
```

### 4. Validation

Validate configuration files before running benchmarks:

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config/algorithms.yaml'))"
```

## Troubleshooting

### YAML Syntax Errors

If you encounter YAML syntax errors:

1. Check indentation (YAML uses spaces, not tabs)
2. Verify quote matching
3. Check for trailing commas (not allowed in YAML)

### Algorithm Not Found

If you get "Algorithm not found" error:

1. Verify algorithm is in `overlapping_algorithms` dict
2. Check that `module` and `function` paths are correct
3. Ensure algorithm is in `target_algorithms` list

### Dataset Not Found

If you get "Dataset not found" error:

1. Verify dataset is in `datasets` dict
2. Check that `path` points to existing file
3. Ensure dataset is in `target_datasets` list

## References

- YAML Specification: https://yaml.org/spec/
- PyYAML Documentation: https://pyyaml.org/wiki/PyYAMLDocumentation
