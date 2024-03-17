#!/bin/bash
# OMP_NUM_THREADS=2 torchrun --standalone --nnodes=1 --nproc_per_node=2  --master_port=56789 finetune_hf.py  /home/w1nd/darkword/1darkword/data_crawl/darkword_data_chatglm3  THUDM/chatglm3-6b  configs/lora.yaml configs/ds_zero_2.json
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=12346 finetune_hf.py  /home/w1nd/darkword/1darkword/data_crawl/darkword_data_chatglm3/  THUDM/chatglm3-6b  configs/lora.yaml configs/ds_zero_2.json 
# CUDA_VISIBLE_DEVICES=0,1 python3 finetune_hf.py  /home/w1nd/darkword/1darkword/data_crawl/darkword_data_chatglm3  THUDM/chatglm3-6b  configs/lora.yaml configs/ds_zero_2.json