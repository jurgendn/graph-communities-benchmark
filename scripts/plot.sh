#!/bin/bash

# Plot generation script
# Fetches and merges benchmark results, then generates visualization charts

set -euo pipefail

BENCHMARK_TYPE=${1:-all}

echo "Step 1: Fetching and merging benchmark results..."
PYTHONPATH=. python3 tools/fetch_and_merge.py --benchmark-type "$BENCHMARK_TYPE"

if [[ $? -ne 0 ]]; then
	echo "Error: fetch_and_merge.py failed"
	exit 1
fi

echo ""
echo "Step 2: Generating plots..."
PYTHONPATH=. python3 tools/plots.py --benchmark-type "$BENCHMARK_TYPE"

if [[ $? -ne 0 ]]; then
	echo "Error: plots.py failed"
	exit 1
fi

echo ""
echo "✓ Plots generated successfully!"
