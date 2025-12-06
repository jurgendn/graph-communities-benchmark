from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Static Community Detection Benchmarking")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/CollegeMsg.txt",
        help="Path to the dataset file (default: data/CollegeMsg.txt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CollegeMsg",
        help="Name of the dataset to load (default: CollegeMsg)",
    )
    parser.add_argument("--source-idx", type=int, default=0, help="Source node column index (default: 0)")
    parser.add_argument("--target-idx", type=int, default=1, help="Target node column index (default: 1)")
    parser.add_argument(
        "--batch-range",
        type=float,
        default=1e-4,
        help="Batch range for temporal graph snapshots (default: 1e-4)",
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.4,
        help="Initial fraction of edges to include in the base graph (default: 0.4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of snapshots to create (default: 10)",
    )
    parser.add_argument(
        "--load-full-nodes",
        action="store_true",
        help="Whether to load all nodes in the base graph (default: False)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=" ",
        help="Delimiter used in the dataset file (default: space)",
    )
    return parser.parse_args()