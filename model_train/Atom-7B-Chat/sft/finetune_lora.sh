output_model=../darkword-Atom-7B-Chat
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./finetune.sh ${output_model}

model_name_or_path="FlagAlpha/Atom-7B-Chat"
# train_files="../../../1data_crawl/atom-sample/train_sft.csv"
train_files="../../../data_crawl/darkword_data_atom/darkword_train_data.csv"
# validation_files="../../../1data_crawl/atom-sample/dev_sft.csv ../../../1data_crawl/atom-sample/dev_sft_sharegpt.csv"
validation_files="../../../data_crawl/darkword_data_atom/darkword_validate_data.csv"
deepspeed --include localhost:0 finetune_clm_lora.py \
    --model_name_or_path $model_name_or_path \
    --train_files $train_files \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 10 \
    --warmup_steps 400 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 20 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    --validation_files  $validation_files \
    --do_eval \
    --eval_steps 20 \
    --max_eval_samples 800 \
    --evaluation_strategy  steps \
    | tee -a ${output_model}/train.log
    


    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
