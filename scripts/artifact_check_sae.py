#!/usr/bin/env python3
"""
Experiment 0A: Gemma Scope SAE Artifact Check (PRIMARY)
========================================================

The highest-priority artifact check. Uses Gemma Scope's pre-trained SAE features
as concept directions — these are learned, sparse, and involve NO subtraction.

For each concept:
  1. Feed word pairs through Gemma, get residual stream activations
  2. Pass through SAE encoder to get sparse feature activations
  3. Identify features that activate differentially for concept pairs
  4. Use the SAE decoder column vector as the concept direction
  5. Project onto eigenspace of Σ and compare CDFs with diff-of-means

Usage (on remote GPU):
    uv run python scripts/artifact_check_sae.py
    uv run python scripts/artifact_check_sae.py --model gemma-2-2b
    uv run python scripts/artifact_check_sae.py --model gemma-2-9b --layer 20 --width 16k
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

RESULTS_DIR = Path("results")
SPECTRAL_DIR = Path("results/_spectral")
OUTPUT_DIR = Path("results/_spectral/artifact_check")

# SAE config per model
SAE_CONFIG = {
    "gemma-2-2b": {
        "model_path": "google/gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "default_layer": 20,
        "default_width": "16k",
        "hidden_dim": 2304,
        "family": "gemma",
    },
    "gemma-2-9b": {
        "model_path": "google/gemma-2-9b",
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "default_layer": 31,
        "default_width": "16k",
        "hidden_dim": 3584,
        "family": "gemma",
    },
    "Llama-3.1-8B": {
        "model_path": "meta-llama/Llama-3.1-8B",
        "sae_repo": "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
        "default_layer": 24,
        "default_width": "32k",
        "hidden_dim": 4096,
        "d_sae": 32768,
        "family": "llama-scope",
    },
}


class JumpReLUSAE(torch.nn.Module):
    """
    Minimal JumpReLU SAE implementation for loading Gemma Scope weights.
    Falls back to loading from npz if sae-lens is not available.
    """
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
    
    def encode(self, x):
        pre_act = x @ self.W_enc + self.b_enc
        # JumpReLU: zero out features below threshold
        mask = (pre_act > self.threshold)
        return pre_act * mask
    
    def decode(self, z):
        return z @ self.W_dec + self.b_dec
    
    def forward(self, x):
        z = self.encode(x - self.b_dec)
        return self.decode(z), z
    
    @classmethod
    def from_npz(cls, path, d_model, d_sae):
        """Load from .npz file (manual download format)."""
        data = np.load(path)
        sae = cls(d_model, d_sae)
        sae.W_enc.data = torch.tensor(data['W_enc'], dtype=torch.float32)
        sae.W_dec.data = torch.tensor(data['W_dec'], dtype=torch.float32)
        sae.b_enc.data = torch.tensor(data['b_enc'], dtype=torch.float32)
        sae.b_dec.data = torch.tensor(data['b_dec'], dtype=torch.float32)
        if 'threshold' in data:
            sae.threshold.data = torch.tensor(data['threshold'], dtype=torch.float32)
        return sae


class TopKReLUSAE(torch.nn.Module):
    """
    Minimal TopK-ReLU SAE implementation for loading Llama Scope weights.
    Llama Scope uses TopK activation: keep only the K largest pre-activations.
    Checkpoint format: safetensors with keys from lm_sae convention.
    """
    def __init__(self, d_model, d_sae, k=64):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.k = k

    def encode(self, x):
        pre_act = (x - self.b_dec) @ self.W_enc + self.b_enc
        # TopK: keep only the top-k activations, zero the rest
        topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        z = torch.zeros_like(pre_act)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @classmethod
    def from_safetensors(cls, safetensors_path, hyperparams_path, d_model, d_sae):
        """Load from Llama Scope safetensors checkpoint + hyperparams.json."""
        import json as _json
        from safetensors.torch import load_file

        # Read TopK k value from hyperparams
        with open(hyperparams_path) as f:
            hparams = _json.load(f)
        k = hparams.get("k", hparams.get("top_k", 64))

        sae = cls(d_model, d_sae, k=k)
        state = load_file(safetensors_path)

        # Map lm_sae checkpoint keys to our parameter names.
        # Llama Scope (lm_sae) uses: encoder.weight (d_sae, d_model),
        # decoder.weight (d_model, d_sae), encoder.bias (d_sae), decoder.bias (d_model)
        key_map = {
            "W_enc": ["encoder.weight", "W_enc"],
            "W_dec": ["decoder.weight", "W_dec"],
            "b_enc": ["encoder.bias", "b_enc"],
            "b_dec": ["decoder.bias", "b_dec", "pre_bias"],
        }
        for param_name, candidates in key_map.items():
            for cand in candidates:
                if cand in state:
                    tensor = state[cand].float()
                    # encoder.weight in lm_sae is (d_sae, d_model), we store (d_model, d_sae)
                    if param_name == "W_enc" and tensor.shape == (d_sae, d_model):
                        tensor = tensor.T
                    # decoder.weight in lm_sae is (d_model, d_sae), we store (d_sae, d_model)
                    if param_name == "W_dec" and tensor.shape == (d_model, d_sae):
                        tensor = tensor.T
                    getattr(sae, param_name).data = tensor
                    break

        print(f"      Loaded TopK SAE: d_model={d_model}, d_sae={d_sae}, k={k}")
        print(f"      Checkpoint keys: {list(state.keys())}")
        return sae


def load_sae_llama_scope(model_name: str, layer: int):
    """
    Load a Llama Scope TopK-ReLU SAE from HuggingFace.
    Repo: OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x
    Each layer is stored as Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors
    Returns: (sae_model, method_used)
    """
    from huggingface_hub import hf_hub_download
    config = SAE_CONFIG[model_name]
    repo_id = config["sae_repo"]
    d_model = config["hidden_dim"]
    d_sae = config["d_sae"]

    prefix = f"Llama3_1-8B-Base-L{layer}R-8x"
    safetensors_file = f"{prefix}/checkpoints/final.safetensors"
    hyperparams_file = f"{prefix}/hyperparams.json"

    print(f"    Downloading Llama Scope SAE: {repo_id}/{prefix}")
    st_path = hf_hub_download(repo_id=repo_id, filename=safetensors_file)
    hp_path = hf_hub_download(repo_id=repo_id, filename=hyperparams_file)

    sae = TopKReLUSAE.from_safetensors(st_path, hp_path, d_model, d_sae)
    return sae, "llama-scope-safetensors"


def load_sae(model_name: str, layer: int, width: str):
    """
    Load SAE using sae-lens (preferred) or manual npz fallback.
    Dispatches to Llama Scope loader for Llama family.
    Returns: (sae_model, method_used)
    """
    config = SAE_CONFIG[model_name]

    # Llama Scope: use dedicated loader
    if config.get("family") == "llama-scope":
        return load_sae_llama_scope(model_name, layer)

    # Gemma family — Method 1: sae-lens
    try:
        from sae_lens import SAE
        sae_id = f"layer_{layer}/width_{width}/canonical"
        print(f"    Loading SAE via sae-lens: {config['sae_release']}, {sae_id}")
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=config['sae_release'],
            sae_id=sae_id,
        )
        return sae, "sae-lens"
    except ImportError:
        print("    sae-lens not installed, trying manual loading...")
    except Exception as e:
        print(f"    sae-lens failed: {e}, trying manual loading...")
    
    # Gemma family — Method 2: Direct HuggingFace download
    try:
        from huggingface_hub import hf_hub_download
        repo_id = f"google/gemma-scope-{model_name.split('-')[-1]}-pt-res"
        filename = f"layer_{layer}/width_{width}/average_l0_*/params.npz"
        
        # List available files
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id)
        # Find matching npz
        target = None
        for f in files:
            if f.startswith(f"layer_{layer}/width_{width}/") and f.endswith("params.npz"):
                target = f
                break
        
        if target:
            print(f"    Downloading {repo_id}/{target}...")
            local_path = hf_hub_download(repo_id=repo_id, filename=target)
            d_model = config['hidden_dim']
            # Parse width
            d_sae = int(width.replace('k', '000').replace('K', '000'))
            sae = JumpReLUSAE.from_npz(local_path, d_model, d_sae)
            return sae, "manual-npz"
    except Exception as e:
        print(f"    Manual loading failed: {e}")
    
    raise RuntimeError(f"Could not load SAE for {model_name} layer {layer} width {width}. "
                       f"Install sae-lens: pip install sae-lens")


def get_residual_stream(texts, model, tokenizer, handler, device, layer_idx=None):
    """
    Get residual stream activations at a specific layer.
    Uses a hook to capture intermediate activations.
    """
    activations = []
    
    def hook_fn(module, input, output):
        # output can be tuple (hidden_states, ...) or just tensor
        if isinstance(output, tuple):
            activations.append(output[0].detach())
        else:
            activations.append(output.detach())
    
    # Find the target layer
    model_type = type(model).__name__.lower()
    layers = None
    
    # Try common attribute names
    for attr_chain in [
        ['model', 'layers'],
        ['model', 'decoder', 'layers'],
        ['transformer', 'h'],
        ['gpt_neox', 'layers'],
    ]:
        obj = model
        for attr in attr_chain:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                obj = None
                break
        if obj is not None:
            layers = obj
            break
    
    if layers is None:
        raise RuntimeError(f"Cannot find transformer layers in {model_type}")
    
    if layer_idx is None:
        layer_idx = len(layers) - 2  # second-to-last layer
    
    hook = layers[layer_idx].register_forward_hook(hook_fn)
    
    try:
        # Run texts through model
        for text in texts:
            formatted = handler.format_input(text, "en", "fr", tokenizer) if hasattr(handler, 'format_input') else text
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model(**inputs)
            
            # Take the last token's activation
            act = activations[-1][0, -1, :].cpu().float()
            activations.clear()
            activations.append(act)
        
        result = torch.stack(activations[:-1] if len(activations) > len(texts) else activations[-len(texts):])
    finally:
        hook.remove()
    
    return result


def identify_sae_concept_features(
    base_words, target_words,
    sae, model, tokenizer, handler, device,
    layer_idx, top_k=3
):
    """
    Identify SAE features that activate differentially for concept pairs.
    Returns: (top_feature_indices, diff_activations, decoder_vectors)
    """
    # Get residual stream activations for positive and negative words
    print(f"      Getting activations for {len(base_words)} pairs...")
    
    pos_acts = []
    neg_acts = []
    
    for bw, tw in zip(base_words, target_words):
        try:
            # Get residual stream at target layer
            act_neg = get_residual_stream([bw], model, tokenizer, handler, device, layer_idx)
            act_pos = get_residual_stream([tw], model, tokenizer, handler, device, layer_idx)
            neg_acts.append(act_neg[0])
            pos_acts.append(act_pos[0])
        except Exception as e:
            continue
    
    if not pos_acts:
        return None, None, None
    
    X_pos = torch.stack(pos_acts)  # (n, d_model)
    X_neg = torch.stack(neg_acts)  # (n, d_model)
    
    # Pass through SAE encoder
    sae_device = next(sae.parameters()).device if hasattr(sae, 'parameters') else 'cpu'
    
    # Handle sae-lens SAE vs manual SAE
    if hasattr(sae, 'encode'):
        z_pos = sae.encode(X_pos.to(sae_device)).cpu()
        z_neg = sae.encode(X_neg.to(sae_device)).cpu()
    else:
        # sae-lens format
        z_pos = sae(X_pos.to(sae_device))[1].cpu()
        z_neg = sae(X_neg.to(sae_device))[1].cpu()
    
    # Differential activation
    diff_act = z_pos.mean(dim=0) - z_neg.mean(dim=0)
    
    # Top-k most differentially activating features
    top_indices = torch.argsort(diff_act.abs(), descending=True)[:top_k]
    
    # Get decoder column vectors for these features
    if hasattr(sae, 'W_dec'):
        decoder_vecs = sae.W_dec.data[top_indices].cpu().float()
    elif hasattr(sae, 'W_dec'):
        decoder_vecs = sae.W_dec[top_indices].cpu().float()
    else:
        # sae-lens: try .decoder attribute
        try:
            decoder_vecs = sae.W_dec[top_indices].detach().cpu().float()
        except:
            print("      WARNING: Cannot extract decoder vectors")
            return top_indices, diff_act, None
    
    return top_indices.tolist(), diff_act, decoder_vecs


def eigendecompose_cached(model_name, lang_pair="en-fr"):
    cache_path = SPECTRAL_DIR / "eigen_cache" / model_name / lang_pair / "eigen.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["eigenvalues"], data["eigenvectors"]
    
    cov_path = RESULTS_DIR / model_name / lang_pair / "cov_src.pt"
    if not cov_path.exists():
        cov_path = RESULTS_DIR / model_name / "cov_en.pt"
    cov = torch.load(cov_path, map_location="cpu", weights_only=True).float()
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    idx = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[idx], eigenvectors[:, idx]


def compute_cdf(v, eigenvectors):
    v = v.float()
    proj = eigenvectors.T @ v
    energy = (proj ** 2) / (v @ v).clamp(min=1e-12)
    return torch.cumsum(energy, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Exp 0A: SAE Artifact Check (Gemma Scope + Llama Scope)")
    parser.add_argument("--model", default="gemma-2-2b", choices=list(SAE_CONFIG.keys()))
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--width", default=None)
    parser.add_argument("--top-k", type=int, default=1,
                       help="Use top-K SAE features per concept (default: 1)")
    args = parser.parse_args()
    
    config = SAE_CONFIG[args.model]
    layer = args.layer or config["default_layer"]
    width = args.width or config["default_width"]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"  EXPERIMENT 0A: Gemma Scope SAE Artifact Check")
    print(f"  Model: {args.model}, Layer: {layer}, Width: {width}")
    print("=" * 60)
    
    # Load model
    print("\n[1] Loading model...")
    model, tokenizer, handler = lrg.load_model(config["model_path"])
    device = next(model.parameters()).device
    print(f"    Device: {device}")
    
    # Load SAE
    print("\n[2] Loading SAE...")
    sae, sae_method = load_sae(args.model, layer, width)
    sae = sae.to(device)
    sae.eval()
    print(f"    Loaded via: {sae_method}")
    
    # Load eigendecomposition
    print("\n[3] Loading eigendecomposition...")
    eigenvalues, eigenvectors = eigendecompose_cached(args.model)
    cumvar = np.cumsum(eigenvalues.numpy()) / eigenvalues.numpy().sum()
    d = len(eigenvalues)
    print(f"    Hidden dim: {d}")
    
    # Process concepts
    print("\n[4] Processing concepts...")
    concept_files = sorted(glob.glob("data/word_pairs/*.txt"))
    results = {}
    sae_cdfs = {}
    
    for concept_file in concept_files:
        concept_name = os.path.basename(concept_file).replace('.txt', '')
        print(f"\n    {concept_name}:")
        
        base_words, target_words = lrg.get_counterfactual_pairs(concept_file, tokenizer)
        if len(base_words) < 3:
            print(f"      SKIP ({len(base_words)} pairs)")
            continue
        
        try:
            top_indices, diff_act, decoder_vecs = identify_sae_concept_features(
                base_words, target_words,
                sae, model, tokenizer, handler, device,
                layer_idx=layer, top_k=args.top_k
            )
            
            if decoder_vecs is None or len(decoder_vecs) == 0:
                print(f"      SKIP: no decoder vectors extracted")
                continue
            
            # Use mean of top-k decoder vectors as concept direction
            v_sae = decoder_vecs.mean(dim=0)
            v_sae = v_sae / v_sae.norm().clamp(min=1e-12)
            
            # Ensure dimension matches
            if v_sae.shape[0] != d:
                print(f"      DIM MISMATCH: SAE={v_sae.shape[0]}, eigen={d}")
                continue
            
            # Compute CDF
            cdf = compute_cdf(v_sae, eigenvectors)
            sae_cdfs[concept_name] = cdf
            
            k_half = np.searchsorted(cumvar, 0.5)
            thi = 1.0 - cdf[min(k_half, len(cdf)-1)]
            gini = float(np.trapezoid(cdf - cumvar, cumvar))
            
            results[concept_name] = {
                "top_features": top_indices,
                "n_pairs": len(base_words),
                "top_k": args.top_k,
                "thi": float(thi),
                "gini_deviation": gini,
            }
            
            print(f"      Features: {top_indices[:3]}, THI={thi:.4f}, Gini={gini:+.4f}")
        
        except Exception as e:
            print(f"      ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    del model, sae
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Plot
    if sae_cdfs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
        for cname, cdf in sae_cdfs.items():
            short = cname.replace("[", "").replace("]", "").strip()
            ax.plot(cumvar, cdf, lw=1, alpha=0.6, label=short)
        ax.set_xlabel("Cumulative Variance V(k)")
        ax.set_ylabel("Cumulative Concept Energy C(k)")
        ax.set_title(f"{args.model} — SAE-Derived Concept CDFs")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.2)
        
        ax = axes[1]
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4, label="Diagonal")
        
        arr = np.array(list(sae_cdfs.values()))
        ax.plot(cumvar, arr.mean(axis=0), "m-", lw=2.5, label=f"SAE (N={len(sae_cdfs)})")
        ax.fill_between(cumvar, arr.min(axis=0), arr.max(axis=0), color="magenta", alpha=0.1)
        
        # Load diff-of-means
        cdf_path = SPECTRAL_DIR / "cdfs" / f"{args.model}_en-fr.json"
        if cdf_path.exists():
            with open(cdf_path) as f:
                exp1 = json.load(f)
            all_dm = []
            for k, v in exp1.items():
                if k.startswith("group_"):
                    all_dm.extend([np.array(c) for c in v])
            if all_dm:
                cumvar_dm = np.array(exp1["cumvar"])
                arr_dm = np.array(all_dm)
                if len(cumvar_dm) != len(cumvar):
                    arr_dm = np.array([np.interp(cumvar, cumvar_dm, c) for c in arr_dm])
                ax.plot(cumvar, arr_dm.mean(axis=0), "b-", lw=2.5, label=f"Diff-of-Means (N={len(all_dm)})")
        
        ax.set_xlabel("Cumulative Variance V(k)")
        ax.set_ylabel("Cumulative Concept Energy C(k)")
        ax.set_title(f"{args.model} — SAE vs. Diff-of-Means")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"artifact_check_sae_{args.model}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved: {save_path}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("  SAE ARTIFACT CHECK VERDICT")
    print("=" * 60)
    
    if results:
        all_ginis = [r["gini_deviation"] for r in results.values()]
        from scipy import stats as sp_stats
        mean_gini = np.mean(all_ginis)
        t_stat, p_val = sp_stats.ttest_1samp(all_ginis, 0.0)
        
        print(f"  N: {len(all_ginis)}")
        print(f"  Mean Gini deviation: {mean_gini:+.4f}")
        print(f"  t={t_stat:.3f}, p={p_val:.2e}")
        
        if p_val < 0.05 and mean_gini < 0:
            print(f"\n  ✅ SAE FEATURES ALSO ANTI-CONCENTRATE.")
            print(f"     Learned sparse features — no subtraction involved — still land in tail.")
            print(f"     → STRONGEST evidence that anti-concentration is real.")
        elif p_val < 0.05 and mean_gini > 0:
            print(f"\n  ⚠️  SAE features show CONCENTRATION (above diagonal).")
            print(f"     → anti-concentration may be a diff-of-means artifact.")
        else:
            print(f"\n  ❌ No significant deviation from uniform.")
    
    save_path = OUTPUT_DIR / f"artifact_check_sae_{args.model}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
