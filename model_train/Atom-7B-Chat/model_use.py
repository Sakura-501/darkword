import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
# 例如: finetune_model_path='FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'
finetune_model_path='/home/w1nd/darkword/1darkword/model_train/Atom-7B-Chat/darkword-Atom-7B-Chat'  
config = PeftConfig.from_pretrained(finetune_model_path)
# 例如: base_model_name_or_path='meta-llama/Llama-2-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
device_map = "cuda" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map=device_map)
model =model.eval()
query=input()
input_ids = tokenizer([f'<s>Human: {query}\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)

# 以前的调用方法，应该是错误的。
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import AutoPeftModelForCausalLM
# device_map = "cuda" if torch.cuda.is_available() else "auto"
# # model_name="FlagAlpha/Atom-7B-Chat"
# # model_name="THUDM/chatglm3-6b"
# model_name="darkword-atom-7B-Chat"
# # model = AutoModelForCausalLM.from_pretrained('FlagAlpha/Atom-7B-Chat',device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
# model = AutoPeftModelForCausalLM.from_pretrained(model_name,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,attn_implementation="flash_attention_2")
# # model = AutoModelForCausalLM.from_pretrained(model_name,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,)
# # model = AutoModelForCausalLM.from_pretrained(model_name,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,)
# model =model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False,trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# # input_ids = tokenizer(['<s>Human: 黑灰产中，关于赌博的黑话关键词有哪些</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
# input_ids = tokenizer(['<s>Human: 黑灰产领域的色情关键词\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
# if torch.cuda.is_available():
#   input_ids = input_ids.to(device_map)
# generate_input = {
#     "input_ids":input_ids,
#     "max_new_tokens":256,
#     "do_sample":True,
#     "top_k":50,
#     "top_p":0.95,
#     "temperature":0.3,
#     "repetition_penalty":1.3,
#     "eos_token_id":tokenizer.eos_token_id,
#     "bos_token_id":tokenizer.bos_token_id,
#     "pad_token_id":tokenizer.pad_token_id
# }
# generate_ids  = model.generate(**generate_input)
# text = tokenizer.decode(generate_ids[0])
# print(text)