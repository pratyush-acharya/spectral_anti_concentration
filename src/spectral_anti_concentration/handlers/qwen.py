import torch
import transformers
from .base import ModelHandler
from typing import Optional

class QwenHandler(ModelHandler):
    """Specialized handler for Qwen2, Qwen2.5, Qwen3, and Qwen3.5.
    
    Works for both dense and MoE models (e.g. Qwen3-30B-A3B).
    """

    def load(self, device_map: str = "auto"):
        print(f"Loading Qwen: {self.model_path}")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            low_cpu_mem_usage=True, 
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def format_input(self, text: str, source_lang: str = "en", target_lang_code: str = "fr", tokenizer: Optional[transformers.PreTrainedTokenizer] = None) -> str:
        """Format input for Qwen models.
        
        Tries chat template (works for most Qwen models), falls back to raw text
        for base models that may not have a chat template configured.
        """
        if tokenizer is None:
            return text
        
        messages = [{"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False 
            )
        except TypeError:
            # Qwen2.5 and older don't support enable_thinking
            try:
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                # Base model with no chat template at all
                return text
        except Exception:
            # Any other failure: fall back to raw text
            return text

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        # Qwen 2/2.5/3 standard path
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        # Qwen 3.5 Hybrid architecture path
        elif hasattr(model, "model") and hasattr(model.model, "final_norm"):
            return model.model.final_norm
        return None
