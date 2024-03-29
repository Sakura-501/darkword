# output_model=../darkword-Atom-7B-Chat
# output_model=../darkword-threefold-Atom-7B-Chat
output_model=/home/w1nd/darkword/1darkword/model_train/Atom-7B-Chat/darkword-Atom-7B-Chat-1e4-1-16-16
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./finetune.sh ${output_model}

model_name_or_path="FlagAlpha/Atom-7B-Chat"
# train_files="../../../1data_crawl/atom-sample/train_sft.csv"
train_files="/home/w1nd/darkword/1darkword/data_crawl/darkword_data_atom/darkword_train_data.csv"
# validation_files="../../../1data_crawl/atom-sample/dev_sft.csv ../../../1data_crawl/atom-sample/dev_sft_sharegpt.csv"
validation_files="/home/w1nd/darkword/1darkword/data_crawl/darkword_data_atom/darkword_eval_data.csv"
deepspeed --include localhost:0 --master_port=12349 finetune_clm_lora.py \
    --model_name_or_path $model_name_or_path \
    --train_files $train_files \
    --do_train \
    --validation_files  $validation_files \
    --do_eval \
    --eval_steps 10 \
    --max_eval_samples 1024 \
    --evaluation_strategy  steps \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 16 \
    --warmup_steps 400 \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 50 \
    --save_total_limit 4000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to wandb \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 true \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
    

    # --load_in_bits 4 \
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
