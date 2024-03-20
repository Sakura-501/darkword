# model_path="/home/w1nd/darkword/1darkword/model_train/Atom-7B-Chat/darkword-Atom-7B-Chat"
model_path_1="/home/w1nd/darkword/1darkword/model_train/Atom-7B-Chat/darkword-Atom-7B-Chat"
model_path_2="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat"

# 命令行启动
CUDA_VISIBLE_DEVICES=0 PEFT_SHARE_BASE_WEIGHTS=true python3 -m fastchat.serve.cli --model $model_path_1

# 单个模型启动web-ui
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path /home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat
# python3 -m fastchat.serve.model_worker --model-path $model_path_1
python3 -m fastchat.serve.gradio_web_server

# 多个模型启动web-ui
python3 -m fastchat.serve.controller
# worker 0
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
# worker 1
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
# 多选项卡
python3 -m fastchat.serve.gradio_web_server_multi