import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device_map = "cuda" if torch.cuda.is_available() else "auto"
# model_name="FlagAlpha/Atom-7B-Chat"
# model_name="THUDM/chatglm3-6b"
model_name="darkword-atom-7B-Chat"
# model = AutoModelForCausalLM.from_pretrained('FlagAlpha/Atom-7B-Chat',device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
# model = AutoModelForCausalLM.from_pretrained(model_name,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,attn_implementation="flash_attention_2")
model = AutoModelForCausalLM.from_pretrained(model_name,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# input_ids = tokenizer(['<s>Human: 黑灰产中，关于赌博的黑话关键词有哪些</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
input_ids = tokenizer(['<s>Human: 黑灰产工具的不断进化的方法和手段分析\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to(device_map)
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":256,
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