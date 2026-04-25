from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalQwenBackend:
    def __init__(self, model_name="Qwen/Qwen3-1.7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=True,
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.0,
            do_sample=False,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return text[len(prompt):].strip()
