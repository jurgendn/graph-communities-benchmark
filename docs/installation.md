# Installation Guide

This project targets Python 3.10+ and uses Comet ML for experiment logging.

## Prerequisites

- Python 3.10 or newer
- `pip`
- Git
- `jq` for `scripts/benchmark.sh`

## Setup

### Clone the repository

```bash
git clone https://github.com/jurgendn/graph-communities-benchmark.git
cd graph-communities-benchmark
```

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

If you prefer conda:

```bash
conda create -n graph-communities-benchmark python=3.10
conda activate graph-communities-benchmark
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

Current pinned dependencies include:

- `cdlib`
- `networkx`
- `pydantic`
- `comet_ml`
- `python-dotenv`
- `numpy`
- `tqdm`
- `dynetx`
- `scipy`
- `matplotlib`
- `pyyaml`

### Configure environment variables

Copy the template and set your Comet credentials:

```bash
cp .env.example .env
```

Recommended minimum values:

```ini
COMET_API_KEY=your_api_key_here
COMET_WORKSPACE=your_workspace_here
```

`COMET_PROJECT_NAME` is present in the template, but benchmark runs currently build the Comet project name from the dataset automatically.

## Verify the environment

```bash
python main.py --help
python main_static.py --help
```

You can also verify imports directly:

```bash
python -c "import cdlib, networkx, comet_ml, yaml; print('ok')"
```

## Optional Notes

- Some CDlib algorithms may require extra system or Python packages that are not part of `requirements.txt`.
- Datasets are expected under `data/`, but that directory is gitignored, so you may need to add data locally.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

For direct tool execution, run from the repository root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python tools/fetch_and_merge.py --help
PYTHONPATH=. python tools/plots.py --help
```

The provided `./scripts/plot.sh` script already does this for you.

### Comet authentication errors

Check that `.env` contains valid `COMET_API_KEY` and `COMET_WORKSPACE` values.

### Missing datasets

Make sure the files referenced by [`config/dataset_config.yaml`](../config/dataset_config.yaml) exist locally under `data/`.

## Next Step

Continue with the [Quick Start Guide](quick_start.md).
