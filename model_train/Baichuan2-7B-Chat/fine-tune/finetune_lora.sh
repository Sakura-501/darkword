hostfile=""
include="localhost:0"
data_path="/home/w1nd/darkword/1darkword/data_crawl/darkword_data_baichuan2/train_data.json"
model_name="baichuan-inc/Baichuan2-7B-Chat"
output_dir="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat"
CUDA_VISIBLE_DEVICES=0 deepspeed --hostfile=$hostfile --include=$include --master_port=12347 fine-tune.py  \
    --report_to "none" \
    --data_path $data_path \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --model_max_length 1024 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --use_lora True \
    > ../lora.log 2>&1
    # --bf16 True \
    # --tf32 True \