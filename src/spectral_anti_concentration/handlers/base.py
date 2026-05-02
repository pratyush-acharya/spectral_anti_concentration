import torch
import transformers
from typing import Optional, List, Union

class ModelHandler:
    """Abstract Base Class for model-specific logic."""
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load(self, device_map: str = "auto"):
        """Loads the model and tokenizer."""
        raise NotImplementedError

    def format_input(self, text: str, source_lang: str = "en", target_lang_code: str = "fr", tokenizer: Optional[transformers.PreTrainedTokenizer] = None) -> str:
        """Formats text for the specific model's chat template.
        For base (pretrained) models, this should return raw text.
        Override in subclass only if the model requires specific formatting.
        """
        return text

    def get_norm_layer(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Identifies the model's final normalization layer."""
        raise NotImplementedError

    def apply_final_norm(self, model: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """Applies the model's final normalization layer to the hidden states."""
        norm_layer = self.get_norm_layer(model)
        if norm_layer is not None:
            return norm_layer(hidden_states)
        return hidden_states

    def _detect_device(self, model) -> str:
        """Detect the device the model is on."""
        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            return "cpu"

    def get_pooled_embeddings(self, model, tokenizer, text_list: List[str], target_words: Optional[List[str]] = None, device: str = "cuda", batch_size: int = 4) -> torch.Tensor:
        """
        Extracts embeddings for specific words (if provided) or last token (default).
        Performs mean-pooling over the word's token span.
        """
        # Auto-detect device from model if default
        if device == "cuda" and not torch.cuda.is_available():
            device = self._detect_device(model)
        
        tokenizer.pad_token = tokenizer.eos_token
        all_embeddings = []

        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i : i + batch_size]
            batch_words = target_words[i : i + batch_size] if target_words else None
            
            max_len = 512
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_len,
                pad_to_multiple_of=8
            ).to(device)
            
            with torch.inference_mode():
                outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
            
            h = outputs.hidden_states[-1] 
            h = self.apply_final_norm(model, h)
            
            batch_emb = []
            for j in range(len(batch_texts)):
                if batch_words and batch_words[j]:
                    word = batch_words[j]
                    text = batch_texts[j]
                    
                    start_char = text.rfind(word)
                    if start_char == -1:
                        pos = inputs.attention_mask[j].sum() - 1
                        batch_emb.append(h[j, pos, :])
                        continue
                    
                    end_char = start_char + len(word)
                    
                    encoding = tokenizer(
                        text, 
                        return_offsets_mapping=True, 
                        add_special_tokens=True, 
                        truncation=True, 
                        max_length=max_len
                    )
                    offsets = encoding['offset_mapping']
                    
                    token_indices = []
                    for idx, (s, e) in enumerate(offsets):
                        if s == e == 0: continue 
                        if (s >= start_char and s < end_char) or (e > start_char and e <= end_char):
                            if idx < h.shape[1]:
                                token_indices.append(idx)
                    
                    if token_indices:
                        word_h = h[j, token_indices, :].mean(dim=0)
                        batch_emb.append(word_h)
                    else:
                        pos = inputs.attention_mask[j].sum() - 1
                        batch_emb.append(h[j, pos, :])
                else:
                    pos = inputs.attention_mask[j].sum() - 1
                    batch_emb.append(h[j, pos, :])
            
            all_embeddings.append(torch.stack(batch_emb))
            
        return torch.cat(all_embeddings)
