#!/bin/bash
PYTHONPATH=. python main_static.py \
    --dataset-path ./data/edit-enwikibooks.txt \
    --dataset edit-enwikibooks \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-4 \
    --initial-fraction 0.4 \
    --max-steps 9 \
    --delimiter " "
    # --delete-insert-ratio 0.5 \
    # --load-full-nodes

# PYTHONPATH=. python main_dynamic.py \
#     --dataset-path ./data/CollegeMsg.txt \
#     --dataset CollegeMsg \
#     --source-idx 0 \
#     --target-idx 1 \
#     --batch-range 1e-5 \
#     --initial-fraction 0.6 \
#     --max-steps 10 \
    # --delete-insert-ratio 0.5 \
    # --load-full-nodes