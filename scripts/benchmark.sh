#!/bin/bash

# Minimal benchmark runner for researchers
# Prioritizes: correctness, reproducibility, transparency

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Print Functions (Simplified)
# ============================================================================

print_help() {
	echo "Usage: ./scripts/benchmark.sh <dataset_name> [script] [num_runs]"
	echo ""
	echo "Examples:"
	echo "  ./scripts/benchmark.sh college-msg main.py 1"
	echo "  ./scripts/benchmark.sh --all main.py 1"
	echo ""
	echo "Datasets:"
	python3 -c "import yaml; print('\n'.join(yaml.safe_load(open('config/dynamic_dataset_config.yaml'))['target_datasets']))"
}

list_datasets() {
	python3 -c "import yaml; print('\n'.join(yaml.safe_load(open('config/dynamic_dataset_config.yaml'))['target_datasets']))"
}

# ============================================================================
# Benchmark Runner
# ============================================================================

run_dataset_benchmark() {
	local dataset=$1
	local script=${2:-main.py}
	local num_runs=${3:-1}

	# Smart detection: if second arg is a number, treat as num_runs and use default script
	if [[ "$2" =~ ^[0-9]+$ ]]; then
		num_runs=$2
		script="main.py"
	fi

	# Extract config as JSON
	local config_json=$(python3 scripts/parse_config.py "$dataset")

	# Check for errors
	if echo "$config_json" | jq -e '.error' >/dev/null 2>&1; then
		local error=$(echo "$config_json" | jq -r '.error')
		echo "Error: $error"
		return 1
	fi

	# Parse fields using jq
	local path=$(echo "$config_json" | jq -r '.path')
	local dataset_name=$(echo "$config_json" | jq -r '.dataset_name')
	local type=$(echo "$config_json" | jq -r '.type')
	local source_idx=$(echo "$config_json" | jq -r '.source_idx')
	local target_idx=$(echo "$config_json" | jq -r '.target_idx')
	local batch_range=$(echo "$config_json" | jq -r '.batch_range')
	local initial_fraction=$(echo "$config_json" | jq -r '.initial_fraction')
	local max_steps=$(echo "$config_json" | jq -r '.max_steps')
	local delimiter=$(echo "$config_json" | jq -r '.delimiter')
	local ground_truth_attr=$(echo "$config_json" | jq -r '.ground_truth_attr')

	# Explicit parameter display (for reproducibility)
	echo "=== Starting: $dataset_name ==="
	echo "[$(date +%H:%M:%S)]"
	echo "Configuration:"
	echo "  Path: $path"
	echo "  Type: $type"
	echo "  Max steps: $max_steps"
	echo "  Script: $script"
	echo "  Runs: $num_runs"
	if [[ "$type" == "edge_list" ]]; then
		echo "  Batch range: $batch_range"
		echo "  Initial fraction: $initial_fraction"
		echo "  Source/Target idx: $source_idx/$target_idx"
	else
		echo "  Ground truth attr: $ground_truth_attr"
	fi
	echo ""

	# Build command based on explicit type
	local cmd="python3 $script --dataset $dataset_name --max-steps $max_steps --num-runs $num_runs"

	if [[ "$type" == "edge_list" ]]; then
		cmd="$cmd --dataset-path $path --source-idx $source_idx --target-idx $target_idx --batch-range $batch_range --initial-fraction $initial_fraction"
	else
		cmd="$cmd --lfr-folder $path --ground-truth-attr $ground_truth_attr"
	fi

	# Show exact command (for transparency)
	echo "Running: $cmd"
	echo ""

	# Execute
	eval "$cmd"

	local exit_code=$?
	if [[ $exit_code -eq 0 ]]; then
		echo "[$(date +%H:%M:%S)] Completed: $dataset_name"
	else
		echo "[$(date +%H:%M:%S)] FAILED: $dataset_name (exit code: $exit_code)"
		return 1
	fi
	echo ""
}

run_all_benchmarks() {
	local script=${1:-main.py}
	local num_runs=${2:-1}

	echo "=== Running all benchmarks ==="
	echo "[$(date +%H:%M:%S)]"
	echo ""

	local datasets=$(list_datasets)

	for dataset in $datasets; do
		run_dataset_benchmark "$dataset" "$script" "$num_runs" || true
	done

	echo "[$(date +%H:%M:%S)] All benchmarks completed"
	echo ""
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
	case "$1" in
	--help | -h)
		print_help
		exit 0
		;;
	--list | -l)
		list_datasets
		exit 0
		;;
	--all)
		shift
		run_all_benchmarks "$@"
		;;
	"")
		print_help
		exit 1
		;;
	*)
		run_dataset_benchmark "$@"
		;;
	esac
}

main "$@"
