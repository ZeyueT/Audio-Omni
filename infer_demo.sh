#!/bin/bash


# Example:
# Example:
# CUDA_VISIBLE_DEVICES=0 python3 ./run_gradio.py --model-config 'model/Audio-Omni.json' --ckpt-path 'model/model.ckpt' --share


CUDA_VISIBLE_DEVICES=0 \
python3 ./run_gradio.py \
    --server-port 7777 \
    --model-config 'model/Audio-Omni.json' \
    --ckpt-path 'model/model.ckpt' \
    --share

