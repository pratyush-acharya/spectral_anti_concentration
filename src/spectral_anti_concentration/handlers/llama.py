import torch
import transformers
from .base import ModelHandler
from typing import Optional

class LlamaHandler(ModelHandler):
    """Handler for Meta's Llama family.
    
    Covers: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, etc.
    
    Architecture notes:
    - Standard decoder-only transformer with RMSNorm
    - Gated models — requires HF_TOKEN and license acceptance
    - Base models use raw text (no chat template)
    - Instruct models use Llama-style chat template
    """

    def load(self, device_map: str = "auto"):
        print(f"Loading Llama: {self.model_path}")
        
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
        """Llama base models: raw text. Instruct: Llama chat template."""
        if tokenizer is None:
            return text
        
        # Check if this is an instruct model
        if "instruct" in self.model_path.lower():
            messages = [{"role": "user", "content": text}]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return text
        
        # Base model: raw text
        return text

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        # Llama standard: model.model.norm (RMSNorm)
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        return None
