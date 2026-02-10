#!/bin/bash
PYTHONPATH=. python main_static.py \
    --dataset-path data/lfr_large_community_size.txt \
    --dataset lfr_large_community_size \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 9 \
    --delimiter " "
    # --load-full-nodes
