# Installation Guide

This guide provides detailed instructions for installing and configuring the Graph Communities Benchmark framework.

## Prerequisites

- Python 3.10+
- pip or conda
- Git (for cloning the repository)

## Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/jurgendn/graph-communities-benchmark.git
cd graph-communities-benchmark
```

### Step 2: Create Virtual Environment

Creating a virtual environment is recommended to isolate project dependencies:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n graph-communities-benchmark python=3.10
conda activate graph-communities-benchmark
```

### Step 3: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

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

> **Note**: Comet ML is used for experiment tracking. You can sign up for free at [comet.com](https://www.comet.com/).

### Step 5: Verify Installation

Run a quick test to verify the installation:

```bash
PYTHONPATH=. python -c "import networkx; import cdlib; print('Installation successful!')"
```

## Dependencies

The benchmark requires the following Python packages:

| Package | Version | Description |
|---------|---------|-------------|
| [cdlib](https://cdlib.readthedocs.io/) | Latest | Community Detection Library |
| [networkx](https://networkx.org/) | Latest | Graph analysis library |
| [matplotlib](https://matplotlib.org/) | Latest | Visualization library |
| [pydantic](https://docs.pydantic.dev/) | Latest | Data validation |
| [comet_ml](https://www.comet.com/) | Latest | Experiment tracking |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Latest | Environment variable management |
| [numba](https://numba.pydata.org/) | Latest | JIT compilation for performance |
| [numpy](https://numpy.org/) | Latest | Numerical computing |
| [scipy](https://scipy.org/) | Latest | Scientific computing |
| [tqdm](https://tqdm.github.io/) | Latest | Progress bars |
| [pyyaml](https://pyyaml.org/) | Latest | YAML configuration parsing |

## Optional Dependencies

For additional functionality, you may want to install:

- [jupyter](https://jupyter.org/): For running the provided `sandbox.ipynb` notebook
- [notebook](https://jupyter-notebook.readthedocs.io/): Jupyter notebook interface

```bash
pip install jupyter notebook
```

## Troubleshooting

### Python Version Issues

If you encounter Python version issues, ensure you're using Python 3.10 or later:

```bash
python --version
```

### Comet ML Authentication

If you encounter authentication errors with Comet ML, verify your credentials in `.env`:

```bash
cat .env
```

### Import Errors

If you encounter import errors, ensure you've set the `PYTHONPATH` correctly:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Numba Installation

If Numba fails to install, you may need to install LLVM first:

```bash
# On Ubuntu/Debian
sudo apt-get install llvm

# On macOS
brew install llvm

# Then install Numba
pip install numba
```

## Next Steps

After successful installation, proceed to the [Quick Start Guide](quick_start.md) to run your first benchmark.
