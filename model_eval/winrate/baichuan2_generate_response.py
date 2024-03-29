from urllib import response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig,AutoPeftModelForCausalLM
import re
import json

llama2_chinese_model_path="FlagAlpha/Atom-7B-Chat"
atom_lora_model_path="/home/w1nd/darkword/1darkword/model_train/Atom-7B-Chat/darkword-threefold-Atom-7B-Chat"
baichuan2_lora_model_path="/home/w1nd/darkword/1darkword/model_train/Baichuan2-7B-Chat/darkword-Baichuan2-7B-Chat-1e4-2-8-16"
chatglm3_lora_model_path="/home/w1nd/darkword/1darkword/model_train/ChatGLM3-6B/darkword-ChatGLM3-6B"
device_map = "cuda" if torch.cuda.is_available() else "auto"


# 加载lora微调后的baichuan2大模型
# baichuan2_model_config=PeftConfig.from_pretrained(baichuan2_lora_model_path)
# baichuan2_tokenizer = AutoTokenizer.from_pretrained(baichuan2_model_config.base_model_name_or_path,use_fast=False,trust_remote_code=True,local_files_only=True)
# baichuan2_model = AutoModelForCausalLM.from_pretrained(baichuan2_model_config.base_model_name_or_path,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
# baichuan2_model = PeftModel.from_pretrained(baichuan2_model, baichuan2_lora_model_path, device_map=device_map)
baichuan2_tokenizer = AutoTokenizer.from_pretrained(baichuan2_lora_model_path, use_fast=False, trust_remote_code=True,local_files_only=True)
baichuan2_model = AutoPeftModelForCausalLM.from_pretrained(baichuan2_lora_model_path,device_map=device_map,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,)
baichuan2_model = baichuan2_model.eval()

def baichuan2_model_generate_response(query):
    messages=[]
    messages.append({"role":"user","content":query})
    try:
        response = baichuan2_model.chat(baichuan2_tokenizer,messages)
        return response
    except:
        return ""

def load_eval_data(eval_data_path):
    with open(eval_data_path,"r",encoding="utf-8") as jsonfile:
        eval_data = json.load(jsonfile)
    jsonfile.close()
    print("评估数据的数量："+str(len(eval_data)))
    print(eval_data)
    return eval_data

def generate_response(eval_data,response_path):
    responses=[]
    
    for each_conversation in eval_data:
        question=each_conversation["conversations"][0]["content"]
        standard_answer=each_conversation["conversations"][1]["content"]
        baichuan2_lora_response=baichuan2_model_generate_response(question)
        one_response={"question":question,"standard_answer":standard_answer,"baichuan2_lora_response":baichuan2_lora_response}
        print(one_response)
        responses.append(one_response)
        with open(response_path,"wt",encoding="utf-8") as jsonfile:
            json.dump(responses,jsonfile,ensure_ascii=False,indent=4)
        jsonfile.close()  
        
       
            

if __name__ == "__main__":
    eval_data_path="/home/w1nd/darkword/1darkword/model_eval/data/dev.json"
    # response_path="/home/w1nd/darkword/1darkword/model_eval/data/base_and_lora_baichuan2_response.json"
    response_path="/home/w1nd/darkword/1darkword/model_eval/data/responses/lora_baichuan2_response.json"
    eval_data = load_eval_data(eval_data_path)
    generate_response(eval_data,response_path)
    # query=input()
    # a_model_response=base_model_generate_response(query)
    # b_model_response=baichuan2_model_generate_response(query)
    # print(a_model_response)
    # print(b_model_response)