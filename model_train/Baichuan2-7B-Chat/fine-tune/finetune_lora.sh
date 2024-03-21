hostfile=""
include="localhost:0"
data_path="/home/w1nd/darkword/1darkword/data_crawl/darkword_data_baichuan2/train_data.json"
# data_path="/home/w1nd/darkword/1darkword/data_crawl/darkword_data_baichuan2/train_data_ten.json"
model_name="baichuan-inc/Baichuan2-7B-Chat"
# output_dir="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat"
output_dir="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-tenfold-Baichuan2-7B-Chat"
# log_file_path="../lora.log"
log_file_path="../lora_ten.log"
CUDA_VISIBLE_DEVICES=0 deepspeed --hostfile=$hostfile --include=$include --master_port=12347 fine-tune.py  \
    --report_to "none" \
    --data_path $data_path \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --model_max_length 512 \
    --num_train_epochs 30 \
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
    --bf16 True \
    > $log_file_path 2>&1
    # --tf32 True \