from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=True,
)

prompt = """
Return only valid JSON:
{"drone_1": [7, 7], "drone_2": [5, 8], "drone_3": [1, 1]}
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
