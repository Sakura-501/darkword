#! /bin/bash
# 似乎依赖不行，换个conda环境跑一下
# conda activate chatglm3-use
run_file="/home/w1nd/darkword/1darkword/model_train/ChatGLM3-6B/finetune_demo/inference_hf.py"
# model_name="THUDM/chatglm3-6b"
# model_name="/home/w1nd/.cache/huggingface/hub/models--THUDM--chatglm3-6b/"
model_name="/home/w1nd/darkword/1darkword/model_train/ChatGLM3-6B/darkword-ChatGLM3-6B/checkpoint-3000/"
echo "y" | CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 $run_file $model_name --prompt $1
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 $run_file $model_name --prompt $1
# CUDA_VISIBLE_DEVICES=0 python3 /home/w1nd/darkword/1darkword/model_train/ChatGLM3-6B/base_model_use.py
