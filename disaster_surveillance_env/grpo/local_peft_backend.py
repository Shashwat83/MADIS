from __future__ import annotations

from typing import Optional

from ..hf_hub_auth import from_pretrained_token_kwargs


class LocalPeftCoordinatorBackend:
    def __init__(
        self,
        *,
        base_model_name: str,
        adapter_path: str,
        tokenizer_path: Optional[str] = None,
        max_new_tokens: int = 220,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Local adapter inference requires transformers, torch, and peft. "
                "Install them with `pip install -e '.[grpo]'` or `pip install -e '.[sft]'`."
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        tokenizer_source = tokenizer_path or base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=trust_remote_code,
            **from_pretrained_token_kwargs(),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
            **from_pretrained_token_kwargs(),
        }
        try:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, **model_kwargs)
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, **model_kwargs)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()


class LoadedPeftCoordinatorBackend:
    def __init__(self, *, model: object, tokenizer: object, torch_module: object, max_new_tokens: int = 220) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()
