from urllib import response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json

llama2_chinese_model_path="FlagAlpha/Atom-7B-Chat"
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
        "max_new_tokens":512,
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
    try:
        matches = re.findall(pattern,text,re.DOTALL)
        return matches[0]
    except:
        return ""


def load_eval_data(eval_data_path):
    with open(eval_data_path,"r",encoding="utf-8") as jsonfile:
        eval_data = json.load(jsonfile)
    jsonfile.close()
    print("评估数据的数量："+str(len(eval_data)))
    print(eval_data)
    return eval_data

def generate_response(eval_data,base_atom_response_path):
    base_atom_responses=[]
    for each_conversation in eval_data:
        question=each_conversation["conversations"][0]["content"]
        standard_answer=each_conversation["conversations"][1]["content"]
        atom_base_response=base_model_generate_response(question)
        base_one_response={"question":question,"standard_answer":standard_answer,"atom_base_response":atom_base_response}
        print(base_one_response)
        base_atom_responses.append(base_one_response)
        # 1. base_atom的保存
        with open(base_atom_response_path,"wt",encoding="utf-8") as jsonfile:
            json.dump(base_atom_responses,jsonfile,ensure_ascii=False,indent=4)
        jsonfile.close()  
            

if __name__ == "__main__":
    eval_data_path="/home/w1nd/darkword/1darkword/model_eval/data/dev.json"
    # response_path="/home/w1nd/darkword/1darkword/model_eval/data/base_and_lora_atom_response.json"
    base_atom_response_path="/home/w1nd/darkword/1darkword/model_eval/data/responses/base_atom_response.json"
    eval_data = load_eval_data(eval_data_path)
    generate_response(eval_data,base_atom_response_path)
    # query=input()
    # a_model_response=base_model_generate_response(query)
    # b_model_response=atom_model_generate_response(query)
    # print(a_model_response)
    # print(b_model_response)