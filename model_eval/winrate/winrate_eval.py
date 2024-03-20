import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
import re
import json

llama2_chinese_model_path="FlagAlpha/Atom-7B-Chat"
atom_lora_model_path="/home/w1nd/darkword/1darkword/model_train/Atom-7B-Chat/darkword-threefold-Atom-7B-Chat"
baichuan2_lora_model_path="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat"
chatglm3_lora_model_path="/home/w1nd/darkword/1darkword/model_train/ChatGLM3-6B/darkword-ChatGLM3-6B"
device_map = "cuda" if torch.cuda.is_available() else "auto"

# 加载基础llama2-chinese大模型atom
base_model_tokenizer = AutoTokenizer.from_pretrained(llama2_chinese_model_path,use_fast=False)
base_model_tokenizer.pad_token = base_model_tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(llama2_chinese_model_path,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
base_model = base_model.eval()

def base_model_generate_response(query):
    input_ids = base_model_tokenizer([f'<s>Human: {query}\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
    input_ids = input_ids.to(device_map)
    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":1024,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":0.3,
        "repetition_penalty":1.3,
        "eos_token_id":base_model_tokenizer.eos_token_id,
        "bos_token_id":base_model_tokenizer.bos_token_id,
        "pad_token_id":base_model_tokenizer.pad_token_id
    }
    generate_ids  = base_model.generate(**generate_input)
    text = base_model_tokenizer.decode(generate_ids[0])
    pattern = r"Assistant: (.*?)\n</s>"
    matches = re.findall(pattern,text,re.DOTALL)
    return matches[0]

# 加载lora微调后的atom大模型
atom_model_config=PeftConfig.from_pretrained(atom_lora_model_path)
atom_tokenizer = AutoTokenizer.from_pretrained(atom_model_config.base_model_name_or_path,use_fast=False)
atom_tokenizer.pad_token = atom_tokenizer.eos_token
atom_model = AutoModelForCausalLM.from_pretrained(atom_model_config.base_model_name_or_path,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
atom_model = PeftModel.from_pretrained(atom_model, atom_lora_model_path, device_map=device_map)
atom_model = atom_model.eval()

def atom_model_generate_response(query):
    input_ids = atom_tokenizer([f'<s>Human: {query}\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
    input_ids = input_ids.to(device_map)
    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":1024,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":0.3,
        "repetition_penalty":1.3,
        "eos_token_id":atom_tokenizer.eos_token_id,
        "bos_token_id":atom_tokenizer.bos_token_id,
        "pad_token_id":atom_tokenizer.pad_token_id
    }
    generate_ids  = atom_model.generate(**generate_input)
    text = atom_tokenizer.decode(generate_ids[0])
    pattern = r"Assistant: (.*?)\n</s>"
    matches = re.findall(pattern,text,re.DOTALL)
    return matches[0]

def load_eval_data():
    with open("/home/w1nd/darkword/1darkword/model_eval/data/eval_test.json","r",encoding="utf-8") as jsonfile:
        eval_data = json.load(jsonfile)
    jsonfile.close()
    print("评估数据的数量："+str(len(eval_data)))
    print(eval_data)
    return eval_data

if __name__ == "__main__":
    # query=input()
    eval_data=load_eval_data()
    for one_conversation in eval_data:
        
    # a_model_response=base_model_generate_response(query)
    # b_model_response=atom_model_generate_response(query)
    # print(a_model_response)
    # print(b_model_response)