import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import AutoPeftModelForCausalLM
model_name="darkword-Baichuan2-7B-Chat"
origin_model_name="baichuan-inc/Baichuan2-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(origin_model_name,trust_remote_code=True)
messages = []
# messages.append({"role": "user", "content": "给我几个在黑灰产领域关于赌博的关键词"})
messages.append({"role": "user", "content": "毒品的黑话种子关键词"})
response = model.chat(tokenizer, messages)
print(response)