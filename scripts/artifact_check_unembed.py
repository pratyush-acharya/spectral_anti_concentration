#!/usr/bin/env python3
"""
Experiment 0B: Unembedding-Derived Artifact Check
==================================================

Tests whether spectral anti-concentration is real or an artifact of
difference-of-means extraction, by deriving concept directions from the
unembedding matrix directly (no contextualized activations).

For each model:
  1. Load tokenizer + unembedding matrix (lm_head weights)
  2. For word pairs with clean single-token mappings, compute:
     v_unembed = mean(γ_pos − γ_neg) where γ is the unembedding row
  3. Project onto eigenspace of Σ and compare CDFs with Exp 1 (diff-of-means)

Usage:
    # On remote GPU (loads model weights, but no inference needed)
    uv run python scripts/artifact_check_unembed.py
    
    # Specific models
    uv run python scripts/artifact_check_unembed.py --models Qwen2.5-0.5B Llama-3.2-1B
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

# Which concepts have clean single-token word pairs
# (we verify at runtime, but these are the candidates)
SINGLE_TOKEN_CANDIDATE_CONCEPTS = [
    "[noun - plural]", "[verb - Ved]", "[verb - Ving]", "[verb - 3pSg]",
    "[adj - comparative]", "[adj - superlative]",
    "[male - female]", "[lower - upper]", "[small - big]",
]

MODEL_PATHS = {
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
}


def load_word_pairs(concept_file: str) -> list:
    """Load word pairs from file. Returns [(base, target), ...]."""
    pairs = []
    with open(concept_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def filter_single_token_pairs(pairs: list, tokenizer) -> list:
    """Keep only pairs where both words are single tokens."""
    clean = []
    for base, target in pairs:
        # Try with and without space prefix
        base_ids = tokenizer.encode(base, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        # Also try with leading space (common in BPE tokenizers)
        base_ids_sp = tokenizer.encode(" " + base, add_special_tokens=False)
        target_ids_sp = tokenizer.encode(" " + target, add_special_tokens=False)
        
        # Accept if either version is single-token
        base_ok = len(base_ids) == 1 or len(base_ids_sp) == 1
        target_ok = len(target_ids) == 1 or len(target_ids_sp) == 1
        
        if base_ok and target_ok:
            # Use the single-token version
            b_id = base_ids[0] if len(base_ids) == 1 else base_ids_sp[-1]
            t_id = target_ids[0] if len(target_ids) == 1 else target_ids_sp[-1]
            clean.append((base, target, b_id, t_id))
    
    return clean


def compute_unembed_concept_direction(clean_pairs: list, W_U: torch.Tensor) -> torch.Tensor:
    """
    Compute concept direction from unembedding rows:
    v = mean(W_U[target_id] - W_U[base_id]) for all pairs, normalized.
    """
    diffs = []
    for _, _, base_id, target_id in clean_pairs:
        diff = W_U[target_id] - W_U[base_id]
        diffs.append(diff)
    
    if not diffs:
        return None
    
    v = torch.stack(diffs).mean(dim=0)
    norm = v.norm()
    if norm > 0:
        v = v / norm
    return v.float()


def eigendecompose_cached(model_name: str, lang_pair: str = "en-fr"):
    """Load cached eigendecomposition from Exp 1, or compute from cov."""
    cache_path = SPECTRAL_DIR / "eigen_cache" / model_name / lang_pair / "eigen.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["eigenvalues"], data["eigenvectors"]
    
    # Compute from covariance
    cov_path = RESULTS_DIR / model_name / lang_pair / "cov_src.pt"
    if not cov_path.exists():
        cov_path = RESULTS_DIR / model_name / "cov_en.pt"
    
    cov = torch.load(cov_path, map_location="cpu", weights_only=True).float()
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}, cache_path)
    
    return eigenvalues, eigenvectors


def compute_cdf(v: torch.Tensor, eigenvectors: torch.Tensor) -> np.ndarray:
    """Compute spectral energy CDF of v projected onto eigenbasis."""
    v = v.float()
    proj = eigenvectors.T @ v
    energy = (proj ** 2) / (v @ v).clamp(min=1e-12)
    return torch.cumsum(energy, dim=0).numpy()


def load_diffmeans_cdf(model_name: str, concept_name: str, lang_pair: str = "en-fr"):
    """Load the difference-of-means CDF from Exp 1 output for comparison."""
    cdf_path = SPECTRAL_DIR / "cdfs" / f"{model_name}_{lang_pair}.json"
    if not cdf_path.exists():
        return None
    
    with open(cdf_path) as f:
        data = json.load(f)
    
    # Find which group this concept belongs to
    from spectral_analysis import CONCEPT_GROUPS
    for group_key, concepts in CONCEPT_GROUPS.items():
        if concept_name in concepts:
            idx = concepts.index(concept_name)
            group_data = data.get(f"group_{group_key}", [])
            if idx < len(group_data):
                return np.array(group_data[idx])
    
    return None


def run_model(model_name: str, model_path: str):
    """Process one model: load, extract unembed directions, compute CDFs."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_name} ({model_path})")
    print(f"{'='*60}")
    
    # Load model (just for tokenizer + W_U, then delete the model)
    print("  Loading model for tokenizer + unembedding...")
    model, tokenizer, handler = lrg.load_model(model_path)
    W_U = lrg.get_lm_head(model).weight.detach().cpu().float()
    print(f"  Unembedding shape: {W_U.shape}")
    
    # Free model memory
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load eigendecomposition
    eigenvalues, eigenvectors = eigendecompose_cached(model_name)
    cumvar = np.cumsum(eigenvalues.numpy()) / eigenvalues.numpy().sum()
    d = len(eigenvalues)
    print(f"  Hidden dim: {d}")
    
    # Process each concept
    concept_files = sorted(glob.glob("data/word_pairs/*.txt"))
    results = {}
    unembed_cdfs = {}
    
    for concept_file in concept_files:
        concept_name = os.path.basename(concept_file).replace('.txt', '')
        
        # Load and filter word pairs
        pairs = load_word_pairs(concept_file)
        clean_pairs = filter_single_token_pairs(pairs, tokenizer)
        
        if len(clean_pairs) < 3:
            print(f"    {concept_name}: SKIP ({len(clean_pairs)}/{len(pairs)} single-token pairs)")
            continue
        
        # Compute unembedding-derived direction
        v_unembed = compute_unembed_concept_direction(clean_pairs, W_U)
        if v_unembed is None:
            continue
        
        # Ensure dimension matches
        if v_unembed.shape[0] != d:
            print(f"    {concept_name}: dim mismatch (unembed {v_unembed.shape[0]} vs eigen {d})")
            continue
        
        # Compute CDF
        cdf = compute_cdf(v_unembed, eigenvectors)
        unembed_cdfs[concept_name] = cdf
        
        # Compute SCM and Gini
        k_half = np.searchsorted(cumvar, 0.5)
        thi = 1.0 - cdf[min(k_half, len(cdf)-1)]
        gini = float(np.trapezoid(cdf - cumvar, cumvar))
        
        results[concept_name] = {
            "n_pairs_total": len(pairs),
            "n_pairs_single_token": len(clean_pairs),
            "thi": float(thi),
            "gini_deviation": gini,
        }
        
        print(f"    {concept_name}: {len(clean_pairs)}/{len(pairs)} pairs, "
              f"THI={thi:.4f}, Gini={gini:+.4f}")
    
    del W_U, tokenizer
    gc.collect()
    
    return results, unembed_cdfs, cumvar


def plot_comparison(model_name: str, unembed_cdfs: dict, cumvar: np.ndarray):
    """Plot unembed CDFs vs diff-of-means CDFs for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: All unembed CDFs
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4, label="Diagonal")
    for cname, cdf in unembed_cdfs.items():
        short = cname.replace("[", "").replace("]", "").strip()
        ax.plot(cumvar, cdf, lw=1, alpha=0.6, label=short)
    ax.set_xlabel("Cumulative Variance V(k)")
    ax.set_ylabel("Cumulative Concept Energy C(k)")
    ax.set_title(f"{model_name}\nUnembedding-Derived Concept CDFs")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.2)
    
    # Panel 2: Aggregate comparison (unembed mean vs diffmeans mean)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4, label="Diagonal")
    
    # Unembed aggregate
    if unembed_cdfs:
        arr = np.array(list(unembed_cdfs.values()))
        mean_cdf = arr.mean(axis=0)
        std_cdf = arr.std(axis=0)
        ax.plot(cumvar, mean_cdf, "r-", lw=2, label=f"Unembedding (N={len(unembed_cdfs)})")
        ax.fill_between(cumvar, mean_cdf - std_cdf, mean_cdf + std_cdf, color="red", alpha=0.1)
    
    # Diff-of-means aggregate (from Exp 1)
    cdf_path = SPECTRAL_DIR / "cdfs" / f"{model_name}_en-fr.json"
    if cdf_path.exists():
        with open(cdf_path) as f:
            exp1_data = json.load(f)
        all_dm_cdfs = []
        for key, cdfs in exp1_data.items():
            if key.startswith("group_"):
                all_dm_cdfs.extend([np.array(c) for c in cdfs])
        if all_dm_cdfs:
            arr_dm = np.array(all_dm_cdfs)
            # Interpolate to same cumvar grid if needed
            cumvar_dm = np.array(exp1_data["cumvar"])
            if len(cumvar_dm) != len(cumvar):
                arr_dm_interp = np.array([np.interp(cumvar, cumvar_dm, c) for c in arr_dm])
            else:
                arr_dm_interp = arr_dm
            mean_dm = arr_dm_interp.mean(axis=0)
            std_dm = arr_dm_interp.std(axis=0)
            ax.plot(cumvar, mean_dm, "b-", lw=2, label=f"Diff-of-Means (N={len(all_dm_cdfs)})")
            ax.fill_between(cumvar, mean_dm - std_dm, mean_dm + std_dm, color="blue", alpha=0.1)
    
    ax.set_xlabel("Cumulative Variance V(k)")
    ax.set_ylabel("Cumulative Concept Energy C(k)")
    ax.set_title(f"{model_name}\nUnembedding vs. Diff-of-Means")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"artifact_check_{model_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp 0B: Unembedding Artifact Check")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to process (default: all in MODEL_PATHS)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    models = args.models or list(MODEL_PATHS.keys())
    
    print("=" * 60)
    print("  EXPERIMENT 0B: Unembedding-Derived Artifact Check")
    print(f"  Models: {len(models)}")
    print("=" * 60)
    
    all_results = {}
    all_ginis_unembed = []
    all_ginis_diffmeans = []
    
    for model_name in models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            print(f"\n  SKIP {model_name}: no model path configured")
            continue
        
        try:
            results, unembed_cdfs, cumvar = run_model(model_name, model_path)
            all_results[model_name] = results
            
            if unembed_cdfs:
                plot_comparison(model_name, unembed_cdfs, cumvar)
                
                # Collect Gini deviations
                for cname, info in results.items():
                    all_ginis_unembed.append(info["gini_deviation"])
                
        except Exception as e:
            print(f"\n  ERROR {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # ── Aggregate verdict ──
    print("\n" + "=" * 60)
    print("  ARTIFACT CHECK VERDICT")
    print("=" * 60)
    
    if all_ginis_unembed:
        from scipy import stats as sp_stats
        mean_gini = np.mean(all_ginis_unembed)
        t_stat, p_val = sp_stats.ttest_1samp(all_ginis_unembed, 0.0)
        
        print(f"\n  Unembedding-derived concepts:")
        print(f"    N observations: {len(all_ginis_unembed)}")
        print(f"    Mean Gini deviation: {mean_gini:+.4f}")
        print(f"    t-test vs 0: t={t_stat:.3f}, p={p_val:.2e}")
        
        if p_val < 0.05 and mean_gini < 0:
            print(f"\n  ✅ ANTI-CONCENTRATION CONFIRMED via unembedding extraction.")
            print(f"     Concepts derived without contextualized subtraction STILL anti-concentrate.")
            print(f"     → The finding is NOT an artifact of difference-of-means.")
        elif p_val < 0.05 and mean_gini > 0:
            print(f"\n  ⚠️  Unembedding shows CONCENTRATION (opposite direction).")
            print(f"     → Difference-of-means may be creating artificial anti-concentration.")
        else:
            print(f"\n  ❌ No significant deviation from uniform.")
            print(f"     → Cannot confirm anti-concentration with this method alone.")
    
    # Save
    save_path = OUTPUT_DIR / "artifact_check_unembed.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {save_path}")


if __name__ == "__main__":
    main()
