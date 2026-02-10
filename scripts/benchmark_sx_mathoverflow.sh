#!/bin/bash
PYTHONPATH=. python main_static.py \
    --dataset-path data/sx-mathoverflow.txt \
    --dataset sx_mathoverflow \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 9 \
    --delimiter " "
    # --load-full-nodes
