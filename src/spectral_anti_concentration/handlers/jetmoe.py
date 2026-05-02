import torch
import transformers
from .base import ModelHandler
from typing import Optional

class JetMoEHandler(ModelHandler):
    """Handler for JetMoE models.
    
    Covers: jetmoe/jetmoe-8b
    
    Architecture notes:
    - Novel MoE with both Mixture of Attention (MoA) and Mixture of MLP Experts
    - 8B total / 2.2B active parameters
    - 24 blocks, 8 experts per layer, 2 active per token
    - Requires trust_remote_code=True for custom architecture
    - Uses custom modeling code from the JetMoE team
    """

    def load(self, device_map: str = "auto"):
        print(f"Loading JetMoE: {self.model_path}")
        print(f"  → trust_remote_code=True (custom JetMoE architecture)")
        
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
        """JetMoE base: raw text."""
        return text

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        # JetMoE: model.model.norm
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        # JetMoE alternate: model.transformer.ln_f 
        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            return model.transformer.ln_f
        return None
