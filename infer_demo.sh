#!/bin/bash


# CUDA_VISIBLE_DEVICES=7 \
# python3 ./run_gradio.py \
#     --model-config 'stable-audio-open-1.0/model_config_10s_unified_clap_t5_pretransform.json' \
#     --ckpt-path 'model/a6f086e4/epoch=17-step=45000.ckpt' \
#     --share    

export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export SKIP_QWEN_WEIGHTS=1

CUDA_VISIBLE_DEVICES=2 \
python3 ./run_gradio.py \
    --server-port 8088 \
    --model-config 'demo_config/model_config.json' 

