#!/bin/bash

# Generic benchmark runner that reads dataset configuration from dataset_config.yaml
# Usage: ./scripts/benchmark.sh <dataset_name> [main_script]
# Example: ./scripts/benchmark.sh bio-wormnet-v3 main_static.py

set -euo pipefail

# Check if dataset name is provided
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset_name> [main_script]"
    echo "Example: $0 bio-wormnet-v3 main_static.py"
    echo ""
    echo "Available datasets:"
    grep "^  [a-z-]*:" config/dataset_config.yaml | sed 's/://g' | sed 's/^  /  - /'
    exit 1
fi

DATASET_NAME=$1
MAIN_SCRIPT=${2:-main_static.py}
CONFIG_FILE="config/dataset_config.yaml"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Parse YAML and extract dataset configuration
# Using a simple grep/awk approach for bash compatibility
extract_config() {
    local dataset=$1
    local key=$2
    grep -A 20 "^  $dataset:" "$CONFIG_FILE" | grep "^    $key:" | head -1 | sed "s/.*: //" | sed "s/['\"]//g"
}

# Extract configuration values
PATH_VAL=$(extract_config "$DATASET_NAME" "path")
DATASET_ID=$(extract_config "$DATASET_NAME" "dataset_name")
SOURCE_IDX=$(extract_config "$DATASET_NAME" "source_idx")
TARGET_IDX=$(extract_config "$DATASET_NAME" "target_idx")
BATCH_RANGE=$(extract_config "$DATASET_NAME" "batch_range")
INITIAL_FRACTION=$(extract_config "$DATASET_NAME" "initial_fraction")
MAX_STEPS=$(extract_config "$DATASET_NAME" "max_steps")
DELIMITER=$(extract_config "$DATASET_NAME" "delimiter")

# Validate that we found the dataset
if [[ -z "$DATASET_ID" ]]; then
    echo "Error: Dataset '$DATASET_NAME' not found in $CONFIG_FILE"
    echo ""
    echo "Available datasets:"
    grep "^  [a-z-]*:" config/dataset_config.yaml | sed 's/://g' | sed 's/^  /  - /'
    exit 1
fi

# Build command arguments as an array for proper quoting
args=(
    "python3" "$MAIN_SCRIPT"
    "--dataset-path" "$PATH_VAL"
    "--dataset" "$DATASET_ID"
    "--source-idx" "$SOURCE_IDX"
    "--target-idx" "$TARGET_IDX"
    "--batch-range" "$BATCH_RANGE"
    "--initial-fraction" "$INITIAL_FRACTION"
    "--max-steps" "$MAX_STEPS"
)

# Add delimiter if specified
if [[ -n "$DELIMITER" ]]; then
    args+=("--delimiter" "$DELIMITER")
fi

echo "Running benchmark for: $DATASET_NAME"
echo "Using main script: $MAIN_SCRIPT"
echo "Dataset path: $PATH_VAL"
echo ""

# Execute the command with PYTHONPATH set
PYTHONPATH=. "${args[@]}"
