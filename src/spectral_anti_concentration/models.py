from .handlers import (
    ModelHandler,
    QwenHandler,
    GemmaHandler,
    LlamaHandler,
    MistralHandler,
    OLMoHandler,
    JetMoEHandler,
    SmolLMHandler,
    DefaultHandler,
)


def get_handler(model_path: str) -> ModelHandler:
    """Factory to get the correct handler based on the model path.
    
    Each model family has a dedicated handler with:
    - Correct model loading (AutoModel class, trust_remote_code, torch_dtype)
    - Proper tokenizer setup (pad_token, special tokens)
    - Family-specific format_input (chat template for instruct, raw text for base)
    - Correct norm layer detection for the architecture
    
    Routing:
        qwen*                → QwenHandler    (Qwen2.5, Qwen3 dense + MoE)
        gemma*               → GemmaHandler   (Gemma-2)
        llama*               → LlamaHandler   (Llama-3.2, Llama-3.1)
        mistral*             → MistralHandler  (Mistral-7B)
        olmo*, olmoe*        → OLMoHandler     (OLMo-2, OLMoE)
        jetmoe*              → JetMoEHandler   (JetMoE-8B)
        smollm*              → SmolLMHandler   (SmolLM2, SmolLM3)
        everything else      → DefaultHandler  (generic CausalLM fallback)
    """
    model_path_lower = model_path.lower()
    
    # Qwen family (dense + MoE: Qwen2, Qwen2.5, Qwen3, Qwen3.5)
    if "qwen" in model_path_lower:
        return QwenHandler(model_path)
    
    # Gemma family (Gemma-2 text-only)
    if "gemma" in model_path_lower:
        return GemmaHandler(model_path)
    
    # Llama family (Meta)
    if "llama" in model_path_lower:
        return LlamaHandler(model_path)
    
    # Mistral family
    if "mistral" in model_path_lower:
        return MistralHandler(model_path)
    
    # OLMo / OLMoE family (Allen AI)
    if "olmo" in model_path_lower:
        return OLMoHandler(model_path)
    
    # JetMoE family
    if "jetmoe" in model_path_lower:
        return JetMoEHandler(model_path)
    
    # SmolLM family (HuggingFace)
    if "smollm" in model_path_lower:
        return SmolLMHandler(model_path)
    
    # Fallback for any other model
    return DefaultHandler(model_path)
