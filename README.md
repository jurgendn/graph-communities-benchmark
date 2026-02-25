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

## Documentation

| Document | Description |
|-----------|-------------|
| [Installation Guide](docs/installation.md) | Setup and installation instructions |
| [Quick Start](docs/quick_start.md) | Get started with running benchmarks |
| [Development Guide](docs/development_guide.md) | Add new algorithms, datasets, and metrics |
| [Metrics Documentation](docs/metrics.md) | Evaluation metrics and their calculations |
| [Visualization Guide](docs/visualization.md) | Generate plots and visualize results |
| [Configuration Guide](docs/configuration.md) | Configure algorithms, datasets, and visualization |

## Implemented Algorithms

### Static (Overlapping) Methods

The benchmark supports a wide range of overlapping community detection algorithms configured in [`config/algorithms.yaml`](config/algorithms.yaml):

| Algorithm | Description |
|-----------|-------------|
| **ANGEL** | A fast, local-first overlapping community detection algorithm |
| **DEMON** | Democratic Estimate of the Modular Organization of a Network |
| **COACH** | Core-Attachment based clustering |
| **NDOCD** | Network Decomposition-based Overlapping Community Detection (Ding et al., 2016) |
| **BigClam** | Overlapping community detection using gradient ascent on affiliation matrix (custom implementation) |
| **CosineOverlap** | Overlapping community detection using cosine similarity in embedding space (custom implementation) |
| **SLPA** | Speaker-Listener Label Propagation Algorithm |
| **Percomvc** | Permanence based Overlapping Community Detection |
| **Core Expansion** | Core Expansion algorithm for overlapping community detection |
| **Graph Entropy** | Graph Entropy based community detection |
| **UMSTMO** | Universal Multi-Scale Community Detection |
| **DPCLUS** | Density-Periphery based Clustering |
| **IPCA** | Iterative Principal Component Analysis |
| **LAIS2** | Label Propagation with Improved Seed Selection |
| **Walkscan** | Walk-based SCAN algorithm for overlapping communities |
| **DCS** | Distributed Community Search |
| **LFM** | Lancichinetti-Fortunato-Radicchi benchmark for overlapping communities |
| **EBGC** | Entropy-based Graph Clustering |

### Dynamic Methods

| Algorithm | Description |
|-----------|-------------|
| **TILES** | Temporal Incremental Local Expansion for streaming community detection |

## Quick Start

### Using Configuration-Based Scripts

```bash
# List available datasets
./scripts/benchmark.sh --list

# Run benchmarks for all target datasets
./scripts/benchmark.sh --all main_static.py

# Run benchmarks for a specific dataset
./scripts/benchmark.sh college-msg main_static.py
```

### Direct Command-Line Execution

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

See [Quick Start Guide](docs/quick_start.md) for more details.

## Project Structure

```
graph-communities-benchmark/
├── main_static.py          # Entry point for static benchmarks
├── main_dynamic.py         # Entry point for dynamic benchmarks
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── config/                 # Configuration files
│   ├── algorithms.yaml     # Algorithm configurations
│   ├── dataset_config.yaml # Dataset configurations
│   └── visualization.yaml  # Visualization settings
│
├── data/                   # Dataset files
│   ├── CollegeMsg.txt
│   ├── email-Eu-core-temporal.txt
│   └── ...
│
├── docs/                   # Documentation
│   ├── development_guide.md # Guide for adding algorithms, datasets, metrics
│   ├── installation.md     # Installation guide
│   ├── quick_start.md      # Quick start guide
│   ├── metrics.md          # Metrics documentation
│   ├── visualization.md    # Visualization guide
│   └── configuration.md   # Configuration guide
│
├── scripts/                # Benchmark execution scripts
│   ├── benchmark.sh        # Generic benchmark runner
│   └── plot.sh            # Plot generation script
│
├── tools/                  # Visualization & analysis scripts
│   ├── fetch_and_merge.py # Fetch and merge experiments from Comet ML
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
    │   ├── base.py         # Base class for static methods
    │   ├── big_clam.py     # BigClam algorithm implementation
    │   ├── cosine_overlap.py # CosineOverlap algorithm implementation
    │   ├── ndocd.py        # NDOCD algorithm implementation
    │   └── ndocd_torch.py  # NDOCD PyTorch implementation
    │
    ├── dynamic_methods/
    │   ├── __init__.py
    │   └── tiles.py        # TILES algorithm wrapper
    │
    ├── evaluations/
    │   ├── base.py         # Base evaluation metrics
    │   └── target_modularity.py # Optimized overlapping modularity with Numba
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

## Metrics

The benchmark tracks the following metrics for each algorithm:

| Metric | Description |
|--------|-------------|
| **Runtime** | Execution time per snapshot (seconds) |
| **CDlib Modularity** | Overlapping modularity score from CDlib (Newman-Girvan based) |
| **Custom Q0 Modularity** | Optimized overlapping modularity with Numba acceleration |
| **Number of Communities** | Communities detected per snapshot |
| **Modularity Stability** | Range of modularity across snapshots |
| **Affected Nodes** | Number of nodes in each snapshot |

All metrics are logged to Comet ML for visualization and comparison.

See [Metrics Documentation](docs/metrics.md) for detailed information.

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

## Comet ML Integration

Experiment results are automatically logged to Comet ML with:

- **Tags**: Algorithm name, dataset name
- **Parameters**: Algorithm, dataset, number of snapshots, initial graph size
- **Summary Metrics**: Average runtime, total runtime, average modularity, modularity stability
- **Step Metrics**: Per-snapshot runtime, modularity, and community count

View your experiments at: [comet.com](https://www.comet.com/)

## Dependencies

- [cdlib](https://cdlib.readthedocs.io/) - Community Detection Library
- [networkx](https://networkx.org/) - Graph analysis library
- [matplotlib](https://matplotlib.org/) - Visualization library
- [pydantic](https://docs.pydantic.dev/) - Data validation
- [comet_ml](https://www.comet.com/) - Experiment tracking
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management
- [numba](https://numba.pydata.org/) - JIT compilation for performance

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see the [Development Guide](docs/development_guide.md) for information on how to add new algorithms, datasets, and metrics.

For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
