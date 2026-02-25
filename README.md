# Graph Communities Benchmark

A comprehensive benchmarking framework for evaluating **overlapping community detection algorithms** on temporal (dynamic) graphs. This project supports both **static methods** applied to graph snapshots and **dynamic methods** that operate on evolving networks.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Temporal Graph Support**: Load edge-list data and simulate dynamic graph evolution with configurable insertion/deletion batches
- **Static Benchmark Mode**: Run community detection algorithms independently on each snapshot
- **Dynamic Benchmark Mode**: Use algorithms designed for evolving networks (e.g., TILES)
- **Community Visualization**: Visualize overlapping communities with circular layout showing community overlaps and membership counts
- **Experiment Tracking**: Integrated [Comet ML](https://www.comet.com/) logging for metrics, parameters, and reproducibility
- **Extensible Architecture**: Easily add new algorithms, datasets, and evaluation metrics

---

## Implemented Algorithms

### Static (Overlapping) Methods

| Algorithm | Description |
|-----------|-------------|
| **ANGEL** | A fast, local-first overlapping community detection algorithm |
| **DEMON** | Democratic Estimate of the Modular Organization of a Network |
| **COACH** | Core-Attachment based clustering |
| **NDOCD** | Network Decomposition-based Overlapping Community Detection (Ding et al., 2016) |

### Dynamic Methods

| Algorithm | Description |
|-----------|-------------|
| **TILES** | Temporal Incremental Local Expansion for streaming community detection |

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/jurgendn/graph-communities-benchmark.git
cd graph-communities-benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Copy the example environment file and configure your Comet ML credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
COMET_API_KEY=your_api_key_here
COMET_PROJECT_NAME=graph-communities-benchmark
COMET_WORKSPACE=your_workspace_here
```

---

## Quick Start

### Running Static Benchmarks

Run overlapping community detection algorithms on each temporal snapshot:

```bash
PYTHONPATH=. python main_static.py \
    --dataset-path ./data/CollegeMsg.txt \
    --dataset CollegeMsg \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 10
```

### Running Dynamic Benchmarks

Run streaming/dynamic community detection algorithms:

```bash
PYTHONPATH=. python main_dynamic.py \
    --dataset-path ./data/CollegeMsg.txt \
    --dataset CollegeMsg \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 10
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-path` | str | `data/CollegeMsg.txt` | Path to the dataset file |
| `--dataset` | str | `CollegeMsg` | Name of the dataset (used for tagging) |
| `--source-idx` | int | `0` | Column index for source node |
| `--target-idx` | int | `1` | Column index for target node |
| `--batch-range` | float | `1e-4` | Fraction of edges per temporal batch |
| `--initial-fraction` | float | `0.4` | Fraction of edges used for base graph |
| `--max-steps` | int | `10` | Maximum number of temporal snapshots |
| `--load-full-nodes` | flag | `False` | Pre-load all nodes into the base graph |
| `--delimiter` | str | `" "` | Delimiter used in the dataset file |

---

## Datasets

The benchmark includes several real-world temporal network datasets:

| Dataset | Description | Size |
|---------|-------------|------|
| `CollegeMsg.txt` | Private messages on an online social network | ~60K edges |
| `email-Eu-core-temporal.txt` | Email network (EU research institution) | ~300K edges |
| `primaryschool.csv` | Face-to-face contact network | ~125K edges |
| `sx-askubuntu.txt` | Stack Exchange Ask Ubuntu interactions | ~900K edges |
| `sx-mathoverflow.txt` | Stack Exchange Math Overflow interactions | ~500K edges |
| `soc-sign-bitcoinalpha.csv` | Bitcoin Alpha trust network | ~24K edges |
| `soc-sign-bitcoinotc.csv` | Bitcoin OTC trust network | ~36K edges |
| `soc-redditHyperlinks-body.tsv` | Reddit hyperlink network | ~300M edges |

### Dataset Format

Datasets should be edge-list text files with the following format:

```
<source_node> <target_node> [optional_attributes...]
```

Lines starting with `//` are treated as comments and skipped.

---

## Project Structure

```
graph-communities-benchmark/
├── main_static.py          # Entry point for static benchmarks
├── main_dynamic.py         # Entry point for dynamic benchmarks
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── data/                   # Dataset files
│   ├── CollegeMsg.txt
│   ├── email-Eu-core-temporal.txt
│   └── ...
│
├── scripts/                # Benchmark execution scripts
│   ├── benchmark_college_msg.sh
│   └── benchmark_soc-sign-bitcoinalpha.sh
│
├── tools/                  # Visualization & analysis scripts
│   ├── fetch.py           # Fetch experiments from Comet ML
│   ├── merge.py           # Merge metrics by algorithm
│   └── plots.py           # Generate comparison plots
│
├── assets/                # Generated visualization outputs
│   └── grouped/           # Grouped comparison plots
│
└── src/
    ├── benchmark.py        # Core benchmark runner functions
    │
    ├── dataloader/
    │   └── data_reader.py  # Temporal graph loading utilities
    │
    ├── factory/
    │   ├── factory.py      # TemporalGraph and TemporalChanges classes
    │   └── communities.py  # Result containers (IntermediateResults, MethodDynamicResults)
    │
    ├── static_methods/
    │   ├── base.py         # Available static method definitions
    │   └── ndocd.py        # NDOCD algorithm implementation
    │
    ├── dynamic_methods/
    │   ├── __init__.py
    │   └── tiles.py        # TILES algorithm wrapper
    │
    ├── evaluations/
    │   └── base.py         # Evaluation metrics (modularity, NMI)
    │
    ├── utils/
    │   ├── arg_parser.py   # Command-line argument parsing
    │   └── visualize.py    # Community network visualization
    │
    └── visualization/      # Metrics visualization pipeline
        ├── config.py       # Configuration management
        ├── core.py         # GroupedPlotter & Plot classes
        ├── data.py         # Fetcher & Merger for Comet ML data
        ├── common.py       # Common utilities
        ├── styles.py       # Plot styling
        └── utils.py        # Data loading & processing
```

---

## Metrics

The benchmark tracks the following metrics for each algorithm:

| Metric | Description |
|--------|-------------|
| **Runtime** | Execution time per snapshot (seconds) |
| **Modularity** | Overlapping modularity score (Newman-Girvan based) |
| **Number of Communities** | Communities detected per snapshot |
| **Modularity Stability** | Range of modularity across snapshots |
| **Affected Nodes** | Number of nodes in each snapshot |

All metrics are logged to Comet ML for visualization and comparison.

---

## Visualization

The benchmark includes a comprehensive visualization system with two components:

### 1. Community Network Visualization

Visualize overlapping communities detected by algorithms in the network:

```python
from src.utils.visualize import visualize_communities

# After running community detection:
result = algorithms.angel(graph, threshold=0.25)
visualize_communities(graph, result.communities)
```

**Features**:
- **Circular Layout**: Communities arranged around cluster centers with nodes positioned near their respective clusters
- **Overlap Visualization**: Overlapping nodes highlighted with red edges and labeled with their membership count
- **Color Coding**: Up to 20 colors (tab20 colormap) for easy community distinction
- **Interactive Legend**: Shows community indices and overlap indicators
- **Edge Visualization**: Network edges displayed with transparency to reveal structure

### 2. Experiment Metrics Visualization & Analysis

Generate comparative plots for benchmark metrics across experiments using data from Comet ML.

**Workflow**:

1. **Fetch Experiments** - Retrieve all experiments from Comet ML workspace:
```bash
python tools/fetch.py
```

2. **Merge Metrics** - Aggregate metrics by algorithm across multiple runs:
```bash
python tools/merge.py
```

3. **Generate Plots** - Create grouped comparison plots:
```bash
python tools/plots.py                    # All metrics
python tools/plots.py --metric modularity  # Specific metric
python tools/plots.py --out custom/path    # Custom output directory
```

**Data Pipeline**:

- **Fetcher** (`src.visualization.data.Fetcher`): Connects to Comet ML API using credentials from `.env`, downloads all experiments for a project, extracts parameters and metrics, saves raw data to `data/raw/`
- **Merger** (`src.visualization.data.Merger`): Reads raw experiment files, groups metrics by algorithm name, normalizes metric values and step indices, aggregates multiple runs per algorithm, outputs merged structure to `data/merge/`
- **GroupedPlotter** (`src.visualization.core.GroupedPlotter`): Loads merged metrics, generates comparison plots (line, box, bar charts), supports per-dataset and cross-dataset analysis, saves results to `assets/grouped/`

**Output Structure**:
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

**Supported Metrics**:
- `num_communities` - Number of detected communities per snapshot
- `cdlib_modularity` - Overlapping modularity score
- `customize_q0_modularity` - Custom modularity variant
- `runtime` - Execution time per snapshot

---

## Extending the Benchmark

### Adding a New Static Algorithm

1. Add the algorithm to `ALGORITHMS_OVERLAPPING` in `src/benchmark.py`:

```python
ALGORITHMS_OVERLAPPING = {
    # Existing algorithms...
    "your_algorithm": {
        "func": algorithms.your_algorithm,  # cdlib function
        "params": {"param1": value1}
    },
}
```

### Adding a New Dynamic Algorithm

1. Create a new class in `src/dynamic_methods/`:

```python
class YourAlgorithm:
    def __call__(self, tg: TemporalGraph) -> MethodDynamicResults:
        # Implementation
        pass
```

2. Register it in the `ALGORITHMS_OVERLAPPING` dict in `run_dynamic_benchmark()`.

---

## Comet ML Integration

Experiment results are automatically logged to Comet ML with:

- **Tags**: Algorithm name, dataset name
- **Parameters**: Algorithm, dataset, number of snapshots, initial graph size
- **Summary Metrics**: Average runtime, total runtime, average modularity, modularity stability
- **Step Metrics**: Per-snapshot runtime, modularity, and community count

View your experiments at: [comet.com](https://www.comet.com/)

---

## Example Output

```
Loaded temporal graph: 11 snapshots
Base graph: 1893 nodes, 7048 edges

angel:
  steps: 11
  avg_runtime: 0.2341s
  avg_modularity: 0.4215

demon:
  steps: 11
  avg_runtime: 0.1823s
  avg_modularity: 0.3987

ndocd:
  steps: 11
  avg_runtime: 0.5612s
  avg_modularity: 0.4102
```

---

## Dependencies

- [cdlib](https://cdlib.readthedocs.io/) - Community Detection Library
- [networkx](https://networkx.org/) - Graph analysis library
- [matplotlib](https://matplotlib.org/) - Visualization library
- [pydantic](https://docs.pydantic.dev/) - Data validation
- [comet_ml](https://www.comet.com/) - Experiment tracking
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{graph_communities_benchmark,
  author = {Dung Nguyen Tien},
  title = {Graph Communities Benchmark},
  year = {2025},
  url = {https://github.com/jurgendn/graph-communities-benchmark}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
