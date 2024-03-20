#! /bin/bash
echo $1 | CUDA_VISIBLE_DEVICES=1 python model_use.py