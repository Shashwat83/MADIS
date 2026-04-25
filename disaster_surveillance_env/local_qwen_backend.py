from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalQwenBackend:
    """Local Qwen text-generation backend compatible with LLMCoordinator."""

    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B") -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None,
            token=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        try:
            return self._generate_on_device(prompt, self.device)
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda":
                raise
            torch.cuda.empty_cache()
            print("CUDA OOM during generation; retrying once on CPU.")
            self.device = "cpu"
            self.model.to("cpu")
            self.model.eval()
            return self._generate_on_device(prompt, self.device)

    def _generate_on_device(self, prompt: str, device: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._clean_generated_text(text)

    @staticmethod
    def _clean_generated_text(text: str) -> str:
        cleaned = text.strip()
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[1].strip()

        json_start = cleaned.find("{")
        if json_start != -1:
            cleaned = cleaned[json_start:].strip()

        return cleaned
