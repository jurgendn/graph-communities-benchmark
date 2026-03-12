#!/usr/bin/env python3
"""Parse dynamic_dataset_config.yaml and output as JSON for shell consumption"""

import sys
import json
import yaml


CONFIG_PATH = "config/dynamic_dataset_config.yaml"


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: parse_config.py <dataset_name>"}))
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract dataset info
    dataset = config.get("datasets", {}).get(dataset_name, {})
    
    if not dataset:
        print(json.dumps({"error": f"Dataset '{dataset_name}' not found"}))
        sys.exit(1)
    
    # Build output dict
    output = {
        "path": dataset.get("path", ""),
        "dataset_name": dataset.get("dataset_name", ""),
        "type": dataset.get("type", "edge_list"),
        "source_idx": dataset.get("source_idx", 0),
        "target_idx": dataset.get("target_idx", 1),
        "batch_range": dataset.get("batch_range", 0.001),
        "initial_fraction": dataset.get("initial_fraction", 0.4),
        "max_steps": dataset.get("max_steps", 9),
        "delimiter": dataset.get("delimiter", " "),
        "ground_truth_attr": dataset.get("ground_truth_attr", "communities"),
    }
    
    print(json.dumps(output))


if __name__ == "__main__":
    main()
