# echo "解释一下菠菜有几种意思" | CUDA_VISIBLE_DEVICES=1 python3 /home/w1nd/darkword/1darkword/model_eval/winrate/winrate_eval.py
# base_and_lora_atom大模型批量产生response
# CUDA_VISIBLE_DEVICES=0 python3 /home/w1nd/darkword/1darkword/model_eval/winrate/atom_generate_response.py

# lora_baichuan2大模型批量产生response
CUDA_VISIBLE_DEVICES="1" python3 /home/w1nd/darkword/1darkword/model_eval/winrate/baichuan2_generate_response.py