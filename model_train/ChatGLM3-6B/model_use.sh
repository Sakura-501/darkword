#! /bin/bash
# 似乎依赖不行，换个conda环境跑一下
# conda activate chatglm3-use
echo "y" | CUDA_VISIBLE_DEVICES=1 python3 finetune_demo/inference_hf.py darkword-ChatGLM3-6B/checkpoint-3000/ --prompt $1
# CUDA_VISIBLE_DEVICES=1 python3 base_model_use.py
