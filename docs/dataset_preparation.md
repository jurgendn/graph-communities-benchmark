# Dataset Preparation Guide

This guide explains how to prepare local datasets for the benchmark, how each loader interprets the files, and how to register datasets in `config/`.

## Supported Input Types

The repository supports three dataset layouts:

- Temporal edge lists for `main.py`
- Static edge lists or `.gml` files for `main_static.py`
- LFR snapshot folders for dynamic or static evaluation

By convention, local datasets live under `data/`. That directory is gitignored, so each machine needs its own local copy.

## Prepare A Temporal Edge List

Use a plain text file where each non-empty line contains at least two fields.

Requirements:

- The loader reads the file in order, so row order defines the temporal sequence.
- Blank lines are ignored.
- Lines starting with `//` are ignored as comments.
- The selected `source_idx` and `target_idx` columns are used as the edge endpoints.
- Extra columns are ignored.
- The graph is loaded as undirected.
- Repeated edges increase the stored edge weight.

Minimal space-delimited example:

```text
1 2
2 3
3 4
4 2
```

Comma-delimited example:

```text
user_a,user_b,1680300000
user_b,user_c,1680300300
user_c,user_a,1680300600
```

For the CSV example above, use `source_idx: 0`, `target_idx: 1`, and `delimiter: ","`.

Temporal batching behavior:

- `initial_fraction` decides how much of the ordered file becomes the base graph.
- `batch_range` decides how many rows are grouped into each temporal update.
- `max_steps` caps the number of generated updates.
- `delete_insert_ratio` controls how many existing edges are randomly removed relative to each insertion batch.

## Prepare A Static Edge List Or GML File

Static edge lists use the same edge parser as temporal runs, but all selected edges are loaded into a single snapshot.

Requirements:

- Use the same edge-list rules described for temporal data.
- `preload_fraction` optionally truncates the file to its first fraction of rows.
- `main_static.py --dataset-path` also accepts `.gml` files.

If your static graph already includes node labels for ground truth, pass the attribute name with `--ground-truth-attr` or set `ground_truth_attr` in `config/static_dataset_config.yaml`.

## Prepare An LFR Snapshot Folder

Use a directory of `.gml` snapshots named with the `snapshot_t<index>.gml` pattern.

Expected layout:

```text
data/my_lfr_dataset/
  snapshot_t0.gml
  snapshot_t1.gml
  snapshot_t2.gml
```

Requirements:

- Dynamic LFR loading expects at least two snapshots.
- Static LFR loading uses `snapshot_t0.gml` when present, otherwise the earliest `snapshot_t*.gml` file.
- Ground truth is read from a node attribute such as `communities`.
- Snapshot indices must be numeric so the files sort correctly.

The dynamic loader diffs consecutive snapshots to build insertions and deletions automatically.

## Register A Dataset In Config

### Dynamic edge-list entry

Add the dataset under `datasets` in `config/dynamic_dataset_config.yaml`:

```yaml
datasets:
  my-temporal-dataset:
    path: ./data/my_temporal_dataset.txt
    dataset_name: MyTemporalDataset
    type: edge_list
    source_idx: 0
    target_idx: 1
    delimiter: " "
    initial_fraction: 0.4
    batch_range: 0.0001
    max_steps: 9
```

### Dynamic LFR entry

```yaml
datasets:
  my-lfr-dataset:
    path: ./data/my_lfr_dataset/
    dataset_name: my_lfr_dataset
    type: lfr
    ground_truth_attr: communities
    max_steps: 9
```

### Static edge-list entry

Add the dataset under `datasets` in `config/static_dataset_config.yaml`:

```yaml
datasets:
  my-static-dataset:
    path: ./data/my_static_dataset.txt
    dataset_name: MyStaticDataset
    type: edge_list
    source_idx: 0
    target_idx: 1
    delimiter: " "
    preload_fraction: 1.0
```

### Static LFR entry

```yaml
datasets:
  my-lfr-dataset:
    path: ./data/my_lfr_dataset/
    dataset_name: my_lfr_dataset
    type: lfr
    ground_truth_attr: communities
```

If you want the shell wrappers to include the dataset in `--list` output and `--all` runs, also add the dataset key to `target_datasets`.

## Public Datasources

When selecting real-world datasets for benchmarking, two comprehensive repositories are commonly referenced:

### SNAP (Stanford Network Analysis Project)

- URL: https://snap.stanford.edu/data
- Categories: Social networks, communication networks, citation networks, collaboration networks, temporal networks, and more
- Many datasets in this repository's config files originate from SNAP (e.g., CollegeMsg, ca-*, soc-sign-bitcoin*, sx-*).
- Datasets are typically provided as edge lists in plain text or CSV format.
- Refer to individual dataset pages for detailed statistics and citation information.

### Network Repository

- URL: https://networkrepository.com
- Categories: Social, biological, technological, web graphs, labeled networks, dynamic networks, and many others
- Offers interactive visualization and exploration before download.
- Datasets are downloadable in standard formats consistent with edge-list or GML requirements.
- Suitable for discovering benchmarks across diverse domains and network types.

## Validate The Dataset Locally

Start with a small run before launching a full benchmark.

Temporal edge list:

```bash
python main.py \
  --dataset-path ./data/my_temporal_dataset.txt \
  --dataset MyTemporalDataset \
  --source-idx 0 \
  --target-idx 1 \
  --batch-range 0.001 \
  --initial-fraction 0.4 \
  --max-steps 2
```

Dynamic LFR folder:

```bash
python main.py \
  --lfr-folder ./data/my_lfr_dataset \
  --ground-truth-attr communities \
  --max-steps 2
```

Static dataset from config:

```bash
python main_static.py --config my-static-dataset --num-runs 1
```

Checks worth confirming:

- The reported node and edge counts look reasonable.
- The dataset name in logs matches your config entry.
- Ground truth is detected when you expect it.
- The run finishes without delimiter or column-index errors.

## Common Pitfalls

- Header rows are not skipped automatically unless they begin with `//`.
- Temporal edge-list order matters because batching uses file order, not timestamps.
- Wrong `delimiter`, `source_idx`, or `target_idx` values usually produce empty or malformed graphs.
- Dynamic LFR benchmarks need at least two snapshots.
- `data/` is not versioned, so teammates must recreate the same local paths themselves.
