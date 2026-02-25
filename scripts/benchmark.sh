#!/bin/bash

# benchmark.sh - Generic benchmark runner for graph datasets
# Reads configuration from dataset_config.yaml
# Supports both static (main_static.py) and dynamic (main_dynamic.py) benchmarks
# 
# Dynamic benchmark functionality merged from run_dynamic.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/dataset_config.yaml"

# ============================================================================
# Print Functions
# ============================================================================

print_header() {
    local title="$1"
    local width=70
    echo ""
    echo "╔$(printf '═%.0s' $(seq 1 $((width-2))))╗"
    printf "║ %-$((width-4))s ║\n" "$title"
    echo "╚$(printf '═%.0s' $(seq 1 $((width-2))))╝"
    echo ""
}

print_section() {
    local title="$1"
    echo ""
    echo "▸ $title"
    echo "$(printf '─%.0s' $(seq 1 60))"
}

print_help() {
    print_header "Graph Dataset Benchmark Runner"
    
    echo "USAGE"
    echo "  ./scripts/benchmark.sh <dataset_name> [main_script] [num_runs] [delete_insert_ratio]"
    echo "  ./scripts/benchmark.sh --all [main_script] [num_runs] [delete_insert_ratio]"
    echo "  ./scripts/benchmark.sh --list"
    echo "  ./scripts/benchmark.sh --help"
    echo ""
    
    echo "EXAMPLES"
    echo "  ./scripts/benchmark.sh bio-wormnet-v3 main_static.py"
    echo "  ./scripts/benchmark.sh ca-condmat main_static.py 5"
    echo "  ./scripts/benchmark.sh --all main_static.py"
    echo "  ./scripts/benchmark.sh --all main_static.py 3"
    echo "  ./scripts/benchmark.sh bio-wormnet-v3 main_dynamic.py 5 0.8"
    echo "  ./scripts/benchmark.sh --all main_dynamic.py 5 0.8"
    echo ""
    
    echo "PARAMETERS"
    echo "  dataset_name           Name of the dataset from config"
    echo "  main_script            Python script to run (default: main_static.py)"
    echo "  num_runs               Number of runs (optional)"
    echo "  delete_insert_ratio    Delete/insert ratio for dynamic benchmarks (default: 0.8)"
    echo ""
    
    print_available_datasets
}

print_available_datasets() {
    echo "AVAILABLE DATASETS (from target_datasets)"
    local count=1
    while IFS= read -r dataset; do
        printf "  %2d. %-30s\n" "$count" "$dataset"
        count=$((count + 1))
    done < <(get_all_datasets)
    echo ""
}

print_summary() {
    local total="$1"
    local successful="$2"
    local failed="$3"
    shift 3
    local failed_datasets=("$@")
    
    echo ""
    echo "╔$(printf '═%.0s' $(seq 1 68))╗"
    printf "║ %-66s ║\n" "BENCHMARK SUMMARY"
    echo "╠$(printf '═%.0s' $(seq 1 68))╣"
    printf "║ %-30s %6d %-29s ║\n" "Total datasets:" "$total" ""
    printf "║ %-30s %6d %-29s ║\n" "Successful:" "$successful" ""
    
    if [[ $failed -gt 0 ]]; then
        printf "║ %-30s %6d %-29s ║\n" "Failed:" "$failed" ""
    fi
    
    if [[ $failed -gt 0 ]]; then
        echo "╠$(printf '═%.0s' $(seq 1 68))╣"
        printf "║ %-66s ║\n" "Failed datasets:"
        for ds in "${failed_datasets[@]}"; do
            printf "║   ✗ %-61s ║\n" "$ds"
        done
    fi
    
    echo "╚$(printf '═%.0s' $(seq 1 68))╝"
    echo ""
}

print_dataset_banner() {
    local current="$1"
    local total="$2"
    local dataset="$3"
    
    echo ""
    echo "┌$(printf '─%.0s' $(seq 1 68))┐"
    printf "│ [%2d/%2d] %-58s │\n" "$current" "$total" "$dataset"
    echo "└$(printf '─%.0s' $(seq 1 68))┘"
}

# ============================================================================
# Configuration Functions
# ============================================================================

extract_config() {
    local dataset=$1
    local key=$2
    
    # First try to find the key directly under the dataset
    local value=$(grep -A 20 "^  $dataset:" "$CONFIG_FILE" | grep "^    $key:" | head -1 | sed "s/.*: //" | sed "s/['\"]//g" | sed 's/#.*//' | xargs)
    
    # If not found, try to extract from common settings (for inherited values via YAML anchors)
    if [[ -z "$value" ]]; then
        value=$(grep -A 20 "^common:" "$CONFIG_FILE" | grep "^  $key:" | head -1 | sed "s/.*: //" | sed "s/['\"]//g" | sed 's/#.*//' | xargs)
    fi
    
    echo "$value"
}

get_all_datasets() {
    # Extract datasets from target_datasets list in YAML
    awk '
        /^target_datasets:/ { in_target=1; next }
        in_target && /^  - / { gsub(/^  - /, ""); print; next }
        in_target && /^[^ ]/ { exit }
    ' "$CONFIG_FILE"
}

validate_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
}

# ============================================================================
# Benchmark Runner
# ============================================================================

run_dataset_benchmark() {
    local dataset=$1
    local script=$2
    local num_runs=${3:-}
    local delete_insert_ratio=${4:-}
    
    # Extract configuration values
    local path_val=$(extract_config "$dataset" "path")
    local dataset_id=$(extract_config "$dataset" "dataset_name")
    local source_idx=$(extract_config "$dataset" "source_idx")
    local target_idx=$(extract_config "$dataset" "target_idx")
    local batch_range=$(extract_config "$dataset" "batch_range")
    local initial_fraction=$(extract_config "$dataset" "initial_fraction")
    local max_steps=$(extract_config "$dataset" "max_steps")
    local delimiter=$(extract_config "$dataset" "delimiter")
    
    # Validate dataset exists
    if [[ -z "$dataset_id" ]]; then
        echo "✗ Error: Dataset '$dataset' not found in $CONFIG_FILE"
        return 1
    fi
    
    # Apply defaults from common_settings if values are empty
    # batch_range=${batch_range:-0.00001}
    # batch_range=${batch_range:-0.0001}
    batch_range=${batch_range:-0.001}
    initial_fraction=${initial_fraction:-0.4}
    max_steps=${max_steps:-9}
    delete_insert_ratio=${delete_insert_ratio:-0.8}
    
    # Build command arguments
    local args=(
        "python3" "$script"
        "--dataset-path" "$path_val"
        "--dataset" "$dataset_id"
        "--source-idx" "$source_idx"
        "--target-idx" "$target_idx"
        "--batch-range" "$batch_range"
        "--initial-fraction" "$initial_fraction"
        "--max-steps" "$max_steps"
    )
    
    # Add optional arguments
    if [[ -n "$delimiter" ]]; then
        args+=("--delimiter" "$delimiter")
    fi

    if [[ -n "$num_runs" ]]; then
        args+=("--num-runs" "$num_runs")
    fi
    
    # Add delete-insert-ratio for dynamic benchmarks
    if [[ "$script" == *"dynamic"* ]]; then
        args+=("--delete-insert-ratio" "$delete_insert_ratio")
    fi
    
    # Print configuration
    echo "  Dataset:       $dataset_id"
    echo "  Script:        $script"
    echo "  Path:          $path_val"
    echo "  Steps:         $max_steps"
    echo "  Initial:       $initial_fraction"
    if [[ -n "$num_runs" ]]; then
        echo "  Runs:          $num_runs"
    fi
    if [[ "$script" == *"dynamic"* ]]; then
        echo "  Del/Ins Ratio: $delete_insert_ratio"
    fi
    echo ""
    
    # Execute benchmark
    PYTHONPATH=. "${args[@]}"
}

run_all_benchmarks() {
    local script=$1
    local num_runs=${2:-}
    local delete_insert_ratio=${3:-}
    
    print_header "Running All Dataset Benchmarks"
    echo "Script: $script"
    if [[ -n "$num_runs" ]]; then
        echo "Runs per dataset: $num_runs"
    fi
    if [[ "$script" == *"dynamic"* && -n "$delete_insert_ratio" ]]; then
        echo "Delete/Insert Ratio: $delete_insert_ratio"
    fi
    
    local all_datasets=($(get_all_datasets))
    local total=${#all_datasets[@]}
    local current=0
    local successful=0
    local failed=0
    local failed_datasets=()
    
    for dataset in "${all_datasets[@]}"; do
        current=$((current + 1))
        print_dataset_banner "$current" "$total" "$dataset"
        
        if run_dataset_benchmark "$dataset" "$script" "$num_runs" "$delete_insert_ratio"; then
            echo "✓ Completed: $dataset"
            successful=$((successful + 1))
        else
            echo "✗ Failed: $dataset"
            failed=$((failed + 1))
            failed_datasets+=("$dataset")
        fi
    done
    
    print_summary "$total" "$successful" "$failed" "${failed_datasets[@]}"
    
    if [[ $failed -gt 0 ]]; then
        exit 1
    fi
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    validate_config
    
    # Handle flags
    if [[ $# -eq 0 ]]; then
        print_help
        exit 1
    fi
    
    case "$1" in
        --help|-h)
            print_help
            exit 0
            ;;
        --list|-l)
            print_available_datasets
            exit 0
            ;;
        --all)
            shift
            local script=${1:-main_static.py}
            local num_runs=${2:-}
            local delete_insert_ratio=${3:-}
            run_all_benchmarks "$script" "$num_runs" "$delete_insert_ratio"
            ;;
        *)
            # Single dataset mode
            local dataset=$1
            local script=${2:-main_static.py}
            local num_runs=${3:-}
            local delete_insert_ratio=${4:-}
            
            print_header "Running Benchmark: $dataset"
            if run_dataset_benchmark "$dataset" "$script" "$num_runs" "$delete_insert_ratio"; then
                echo ""
                echo "✓ Benchmark completed successfully"
                exit 0
            else
                echo ""
                echo "✗ Benchmark failed"
                exit 1
            fi
            ;;
    esac
}

# Run main
main "$@"