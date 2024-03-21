from transformers import AutoTokenizer, AutoModel
from peft import AutoPeftModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
# lora_model_path="darkword-ChatGLM3-6B/checkpoint-3000"
# model = AutoPeftModelForCausalLM.from_pretrained(lora_model_path,trust_remote_code=True,device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(lora_model_path,trust_remote_code=True,device_map="auto")
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
print(history)