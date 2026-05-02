import torch
import transformers
from .base import ModelHandler
from typing import Optional

class SmolLMHandler(ModelHandler):
    """Handler for HuggingFace's SmolLM family.
    
    Covers: SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B, SmolLM3-3B
    
    Architecture notes:
    - Standard decoder-only transformer (Llama-like architecture)
    - SmolLM2: compact models for on-device use
    - SmolLM3: multilingual (EN, FR, ES, DE, IT, PT), 128k context
    - No special loading requirements
    """

    def load(self, device_map: str = "auto"):
        print(f"Loading SmolLM: {self.model_path}")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            device_map=device_map,
            torch_dtype="auto",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def format_input(self, text: str, source_lang: str = "en", target_lang_code: str = "fr", tokenizer: Optional[transformers.PreTrainedTokenizer] = None) -> str:
        """SmolLM base models: raw text. Instruct: SmolLM chat template."""
        if tokenizer is None:
            return text
        
        if "instruct" in self.model_path.lower():
            messages = [{"role": "user", "content": text}]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return text
        
        return text

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        # SmolLM uses Llama-like architecture: model.model.norm
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        return None
