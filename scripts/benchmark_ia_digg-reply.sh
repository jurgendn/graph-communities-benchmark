#!/usr/bin/env bash
PYTHONPATH=. python main_static.py \
    --dataset-path ./data/ia-digg-reply.txt \
    --dataset ia-digg-reply \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-5 \
    --initial-fraction 0.4 \
    --max-steps 10 \
    --delimiter " "
    # --load-full-nodes