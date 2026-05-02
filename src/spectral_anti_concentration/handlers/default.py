import torch
import transformers
from .base import ModelHandler
from typing import Optional

class DefaultHandler(ModelHandler):
    """Handler for standard CausalLM architectures.
    
    Works for: Llama, Mistral, SmolLM, OLMo, JetMoE, and any
    model loadable via AutoModelForCausalLM.
    """
    
    def load(self, device_map: str = "auto"):
        model_lower = self.model_path.lower()
        needs_trust = any(kw in model_lower for kw in [
            "jetmoe", "olmoe", "olmo"
        ])
        
        print(f"Loading CausalLM (Default): {self.model_path}")
        if needs_trust:
            print(f"  → trust_remote_code=True (custom architecture)")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=needs_trust
        )
        
        # Load config first to fix RoPE integer bug in some architectures
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=needs_trust
        )
        if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
            for key in ['beta_fast', 'beta_slow', 'factor', 'mscale', 'mscale_all_dim']:
                if key in config.rope_scaling and isinstance(config.rope_scaling[key], int):
                    config.rope_scaling[key] = float(config.rope_scaling[key])
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            config=config,
            low_cpu_mem_usage=True, 
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=needs_trust
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def format_input(self, text: str, source_lang: str = "en", target_lang_code: str = "fr", tokenizer: Optional[transformers.PreTrainedTokenizer] = None) -> str:
        """Base models: return raw text. No chat template needed."""
        return text

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        # Standard Llama/Mistral/OLMo path
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        elif hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
            return model.model.final_layernorm
        elif hasattr(model, "norm"):
            return model.norm
        # JetMoE / other custom architectures path
        elif hasattr(model, "model") and hasattr(model.model, "final_norm"):
            return model.model.final_norm
        return None
