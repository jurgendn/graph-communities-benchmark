#!/bin/bash
PYTHONPATH=. python main_static.py \
    --dataset-path data/sx-askubuntu.txt \
    --dataset sx_askubuntu \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 9 \
    --delimiter " "
    # --load-full-nodes

# PYTHONPATH=. python main_dynamic.py \
#     --dataset-path data/sx-askubuntu.txt \
#     --dataset sx_askubuntu \
#     --source-idx 0 \
#     --target-idx 1 \
#     --batch-range 2e-3 \
#     --initial-fraction 0.3 \
#     --max-steps 10 \
#     --delimiter " "
    # --load-full-nodes