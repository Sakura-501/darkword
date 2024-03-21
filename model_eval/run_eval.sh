# echo "解释一下菠菜有几种意思" | CUDA_VISIBLE_DEVICES=1 python3 /home/w1nd/darkword/1darkword/model_eval/winrate/winrate_eval.py
# base_and_lora_atom大模型批量产生response
CUDA_VISIBLE_DEVICES=0 python3 /home/w1nd/darkword/1darkword/model_eval/winrate/atom_generate_response.py

# lora_baichuan2大模型批量产生response
# echo "y" | CUDA_VISIBLE_DEVICES="1" python3 /home/w1nd/darkword/1darkword/model_eval/winrate/baichuan2_generate_response.py

# lora_chatglm3大模型批量产生response
# 因为涉及到peft==0.7.0,transformers==4.37.2的问题，所以记得先conda activate chatglm3-use 
# pip freeze | grep peft
# CUDA_VISIBLE_DEVICES="1" python3 /home/w1nd/darkword/1darkword/model_eval/winrate/chatglm3_generate_response.py