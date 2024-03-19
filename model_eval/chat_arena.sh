# model_path="/home/w1nd/darkword/1darkword/model_train/Atom-7B-Chat/darkword-Atom-7B-Chat"
model_path="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat"

# 命令行启动
CUDA_VISIBLE_DEVICES=0 PEFT_SHARE_BASE_WEIGHTS=true python3 -m fastchat.serve.cli --model $model_path