hostfile=""
include="localhost:0"
data_path="/home/w1nd/darkword/1darkword/data_crawl/darkword_data_baichuan2/train_data.json"
# data_path="/home/w1nd/darkword/1darkword/data_crawl/darkword_data_baichuan2/train_data_ten.json"
model_name="baichuan-inc/Baichuan2-7B-Chat"
output_dir="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat-1e4-2-8-16"
# output_dir="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat-2"
# log_file_path="../lora.log"

if [ ! -d ${output_dir} ];then  
    mkdir ${output_dir}
fi

CUDA_VISIBLE_DEVICES=0 deepspeed --hostfile=$hostfile --include=$include --master_port=12347 fine-tune.py  \
    --report_to "wandb" \
    --data_path $data_path \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --model_max_length 512 \
    --num_train_epochs 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy epoch \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.2 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --use_lora True \
    --bf16 True \
    | tee -a ${output_dir}/train.log
    # --tf32 True \