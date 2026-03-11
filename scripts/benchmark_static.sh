#!/bin/bash

# Static graph benchmark runner
# Loads datasets as single-snapshot graphs and runs community detection algorithms
#
# Usage:
#   ./scripts/benchmark_static.sh <dataset_key> [num_runs]
#   ./scripts/benchmark_static.sh --all [num_runs]
#   ./scripts/benchmark_static.sh --list
#   ./scripts/benchmark_static.sh --builtin karate [num_runs]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Print Functions
# ============================================================================

print_help() {
	echo "Usage: ./scripts/benchmark_static.sh <dataset_key> [num_runs]"
	echo ""
	echo "Examples:"
	echo "  ./scripts/benchmark_static.sh karate 3"
	echo "  ./scripts/benchmark_static.sh email-enron-large 1"
	echo "  ./scripts/benchmark_static.sh --builtin karate 3"
	echo "  ./scripts/benchmark_static.sh --all 1"
	echo "  ./scripts/benchmark_static.sh --list"
	echo ""
	echo "Options:"
	echo "  --all          Run all target_static_datasets from config"
	echo "  --list         List available static datasets"
	echo "  --builtin NAME Load a built-in NetworkX graph"
	echo "  --help         Show this help"
}

list_datasets() {
	python3 main_static.py --list-datasets
}

# ============================================================================
# Benchmark Runner
# ============================================================================

run_static_benchmark() {
	local dataset=$1
	local num_runs=${2:-1}

	echo "=== Static Benchmark: $dataset ==="
	echo "[$(date +%H:%M:%S)]"
	echo "  Runs: $num_runs"
	echo ""

	local cmd="python3 main_static.py --config $dataset --num-runs $num_runs"

	echo "Running: $cmd"
	echo ""

	eval "$cmd"

	local exit_code=$?
	if [[ $exit_code -eq 0 ]]; then
		echo "[$(date +%H:%M:%S)] Completed: $dataset"
	else
		echo "[$(date +%H:%M:%S)] FAILED: $dataset (exit code: $exit_code)"
		return 1
	fi
	echo ""
}

run_builtin_benchmark() {
	local name=$1
	local num_runs=${2:-1}

	echo "=== Static Benchmark (built-in): $name ==="
	echo "[$(date +%H:%M:%S)]"
	echo ""

	local cmd="python3 main_static.py --builtin $name --num-runs $num_runs"

	echo "Running: $cmd"
	echo ""

	eval "$cmd"
}

run_all_benchmarks() {
	local num_runs=${1:-1}

	echo "=== Running all static benchmarks ==="
	echo "[$(date +%H:%M:%S)]"
	echo ""

	# Extract target_static_datasets from config
	local datasets=$(python3 -c "
import yaml
with open('config/dataset_config.yaml') as f:
    cfg = yaml.safe_load(f)
targets = cfg.get('target_static_datasets', list(cfg.get('static_graphs', {}).keys()))
print('\n'.join(targets))
")

	for dataset in $datasets; do
		run_static_benchmark "$dataset" "$num_runs" || true
	done

	echo "[$(date +%H:%M:%S)] All static benchmarks completed"
	echo ""
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
	if [[ $# -eq 0 ]]; then
		print_help
		exit 1
	fi

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
	--builtin)
		shift
		if [[ $# -eq 0 ]]; then
			echo "Error: --builtin requires a graph name"
			python3 main_static.py --list-builtins
			exit 1
		fi
		run_builtin_benchmark "$@"
		;;
	*)
		run_static_benchmark "$@"
		;;
	esac
}

main "$@"
