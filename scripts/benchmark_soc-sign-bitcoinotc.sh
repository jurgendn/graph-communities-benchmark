#!/bin/bash
PYTHONPATH=. python main_static.py \
    --dataset-path data/soc-sign-bitcoinotc.csv \
    --dataset soc_sign_bitcoin_otc \
    --source-idx 0 \
    --target-idx 1 \
    --batch-range 1e-3 \
    --initial-fraction 0.4 \
    --max-steps 10 \
    --delimiter ","
    # --load-full-nodes

# PYTHONPATH=. python main_dynamic.py \
#     --dataset-path data/soc-sign-bitcoinotc.csv \
#     --dataset soc_sign_bitcoin_otc \
#     --source-idx 0 \
#     --target-idx 1 \
#     --batch-range 1e-5 \
#     --initial-fraction 0.6 \
#     --max-steps 10 \
#     --delimiter ","
#     # --load-full-nodes