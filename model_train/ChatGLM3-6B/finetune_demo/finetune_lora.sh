#!/bin/bash
# OMP_NUM_THREADS=2 torchrun --standalone --nnodes=1 --nproc_per_node=2  --master_port=56789 finetune_hf.py  /home/w1nd/darkword/1darkword/data_crawl/darkword_data_chatglm3  THUDM/chatglm3-6b  configs/lora.yaml configs/ds_zero_2.json
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=12346 finetune_hf.py  /home/w1nd/darkword/1darkword/data_crawl/darkword_data_chatglm3/  THUDM/chatglm3-6b configs/lora.yaml
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=12346 finetune_hf.py  /home/w1nd/darkword/1darkword/data_crawl/darkword_data_chatglm3/  THUDM/chatglm3-6b configs/lora.yaml
# hostfile=""
# include="localhost:0,1"
# data_path="/home/w1nd/darkword/1darkword/data_crawl/darkword_data_chatglm3/"
# model_name="THUDM/chatglm3-6b"
# dp_path="configs/ds_zero_2.json"
# deepspeed --hostfile=$hostfile --include=$include finetune_hf.py $data_path $model_name --deepspeed $dp_path
# CUDA_VISIBLE_DEVICES=0,1 python3 finetune_hf.py  /home/w1nd/darkword/1darkword/data_crawl/darkword_data_chatglm3  THUDM/chatglm3-6b  configs/lora.yaml configs/ds_zero_2.json