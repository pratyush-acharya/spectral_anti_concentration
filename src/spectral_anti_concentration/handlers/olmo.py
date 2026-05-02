import torch
import transformers
from .base import ModelHandler
from typing import Optional

class OLMoHandler(ModelHandler):
    """Handler for Allen AI's OLMo family (dense + MoE).
    
    Covers:
    - OLMo-2-0425-1B (dense, 1B)
    - OLMoE-1B-7B-0125 (MoE, 7B total / 1B active, hidden_size=2048)
    
    Architecture notes:
    - Fully open models (weights + data + code)
    - Requires trust_remote_code=True for custom OLMo architecture
    - OLMoE uses 64 experts per layer with 8 active per token
    - Uses custom norm layer paths
    """

    def load(self, device_map: str = "auto"):
        is_moe = "olmoe" in self.model_path.lower()
        model_type = "OLMoE (MoE)" if is_moe else "OLMo (Dense)"
        print(f"Loading {model_type}: {self.model_path}")
        print(f"  → trust_remote_code=True (custom OLMo architecture)")
        
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
        """OLMo base models: raw text."""
        if tokenizer is None:
            return text
        
        if "instruct" in self.model_path.lower() or "sft" in self.model_path.lower():
            messages = [{"role": "user", "content": text}]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return text
        
        return text

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        # OLMo/OLMoE: model.model.norm
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        # OLMo alternate: model.model.transformer.ln_f
        elif hasattr(model, "model") and hasattr(model.model, "transformer"):
            if hasattr(model.model.transformer, "ln_f"):
                return model.model.transformer.ln_f
            if hasattr(model.model.transformer, "ff_out"):
                return None  # OLMo v1 has no final norm
        return None
