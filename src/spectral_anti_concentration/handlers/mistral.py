import torch
import transformers
from .base import ModelHandler
from typing import Optional

class MistralHandler(ModelHandler):
    """Handler for Mistral AI models.
    
    Covers: Mistral-7B-v0.1, Mistral-7B-v0.3, etc.
    
    Architecture notes:
    - Decoder-only transformer with Sliding Window Attention + RMSNorm
    - Base models use raw text
    - Instruct models use [INST] ... [/INST] format
    - v0.3 has extended vocabulary with v3 tokenizer
    """

    def load(self, device_map: str = "auto"):
        print(f"Loading Mistral: {self.model_path}")
        
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
        """Mistral base: raw text. Instruct: [INST] template."""
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
        # Mistral standard: model.model.norm (RMSNorm)
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        return None
