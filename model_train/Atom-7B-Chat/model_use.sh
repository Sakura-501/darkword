#! /bin/bash
echo $1 | CUDA_VISIBLE_DEVICES=0 python3 model_use.py