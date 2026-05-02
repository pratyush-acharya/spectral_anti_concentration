import torch
import transformers
from .base import ModelHandler
from typing import Optional

class GemmaHandler(ModelHandler):
    """Handler for Gemma model family.
    
    Supports:
    - Gemma-2 (text-only, base): google/gemma-2-2b, gemma-2-9b, gemma-2-27b
    - TranslateGemma (multimodal): google/translategemma-4b-it
    """

    def _is_translate_gemma(self) -> bool:
        return "translategemma" in self.model_path.lower()

    def load(self, device_map: str = "auto"):
        if self._is_translate_gemma():
            return self._load_translate_gemma(device_map)
        return self._load_gemma_text(device_map)

    def _load_gemma_text(self, device_map: str = "auto"):
        """Load standard Gemma-2 text-only models."""
        print(f"Loading Gemma (CausalLM): {self.model_path}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            low_cpu_mem_usage=True, 
            device_map=device_map,
            torch_dtype="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _load_translate_gemma(self, device_map: str = "auto"):
        """Load TranslateGemma multimodal model."""
        print(f"Loading TranslateGemma (ImageTextToText): {self.model_path}")
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            print("Warning: AutoModelForImageTextToText not found, falling back to CausalLM")
            return self._load_gemma_text(device_map)

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, 
            low_cpu_mem_usage=True, 
            device_map=device_map,
            torch_dtype="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def format_input(self, text: str, source_lang: str = "en", target_lang_code: str = "fr", tokenizer: Optional[transformers.PreTrainedTokenizer] = None) -> str:
        """Format input text.
        
        - TranslateGemma: uses multimodal chat template with lang codes
        - Gemma-2 base: returns raw text (no chat template for base models)
        """
        if not self._is_translate_gemma():
            return text  # Base Gemma-2: raw text
            
        if tokenizer is None:
            return text
            
        # TranslateGemma multimodal format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang_code,
                        "text": text,
                    }
                ],
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        # TranslateGemma wraps in text_model
        if hasattr(model, "text_model") and hasattr(model.text_model, "model") and hasattr(model.text_model.model, "norm"):
             return model.text_model.model.norm
        # Standard Gemma-2 path
        elif hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        return None
