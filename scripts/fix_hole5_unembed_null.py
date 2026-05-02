#!/usr/bin/env python3
"""
Hole 5 Fix: Random Unembedding Null Model
===========================================

The Exp 0B unembed check found CONCENTRATION (positive Gini). But we don't know
if this is trivially expected — maybe all unembedding difference vectors concentrate.

This fix:
  1. For each model, sample 1000 random token PAIRS from the vocabulary
  2. Compute their unembedding difference vector (W_U[t1] - W_U[t2])
  3. Project onto eigenbasis and compute Gini deviation
  4. Compare: concept unembed Gini vs random unembed Gini

If random pairs concentrate MORE than concept pairs → concept-specific structure exists.
If random pairs concentrate EQUALLY → concentration is trivially expected from W_U.
Either way, the interesting finding is that contextualized representations REVERSE this.

Usage (on remote GPU — loads model weights):
    uv run python scripts/fix_hole5_unembed_null.py
    uv run python scripts/fix_hole5_unembed_null.py --models Qwen2.5-0.5B Llama-3.2-1B
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

RESULTS_DIR = Path("results")
SPECTRAL_DIR = Path("results/_spectral")
OUTPUT_DIR = Path("results/_spectral/hole_fixes")

MODEL_PATHS = {
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
}


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


def compute_gini(v, eigenvectors, cumvar):
    v = v.float()
    proj = eigenvectors.T @ v
    energy = (proj ** 2) / (v @ v).clamp(min=1e-12)
    cdf = torch.cumsum(energy, dim=0).numpy()
    return float(np.trapezoid(cdf - cumvar, cumvar)), cdf


def random_unembed_null(W_U, eigenvectors, cumvar, n_samples=1000, seed=42):
    """
    Sample random token pairs and compute Gini of their unembed difference.
    """
    V, d = W_U.shape
    rng = np.random.default_rng(seed)
    
    ginis = []
    cdfs_all = []
    
    for _ in range(n_samples):
        # Sample two random token IDs
        t1, t2 = rng.choice(V, size=2, replace=False)
        v = (W_U[t1] - W_U[t2]).float()
        norm = v.norm()
        if norm < 1e-12:
            continue
        v = v / norm
        
        gini, cdf = compute_gini(v, eigenvectors, cumvar)
        ginis.append(gini)
        cdfs_all.append(cdf)
    
    return {
        "gini_mean": float(np.mean(ginis)),
        "gini_std": float(np.std(ginis)),
        "gini_median": float(np.median(ginis)),
        "n": len(ginis),
        "mean_cdf": np.mean(cdfs_all, axis=0) if cdfs_all else None,
        "ci_lower": np.percentile(cdfs_all, 2.5, axis=0) if cdfs_all else None,
        "ci_upper": np.percentile(cdfs_all, 97.5, axis=0) if cdfs_all else None,
    }


def load_concept_unembed_results(model_name):
    """Load Exp 0B results for this model."""
    path = SPECTRAL_DIR / "artifact_check" / "artifact_check_unembed.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(model_name)


def run_model(model_name, model_path):
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"{'=' * 60}")
    
    # Load model weights
    print("  Loading model for unembedding matrix...")
    model, tokenizer, handler = lrg.load_model(model_path)
    W_U = lrg.get_lm_head(model).weight.detach().cpu().float()
    print(f"  W_U shape: {W_U.shape}")
    
    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load eigendecomposition
    eigenvalues, eigenvectors = eigendecompose_cached(model_name)
    cumvar = np.cumsum(eigenvalues.numpy()) / eigenvalues.numpy().sum()
    d = len(eigenvalues)
    
    # Dim check
    if W_U.shape[1] != d:
        print(f"  ⚠️  W_U dim {W_U.shape[1]} ≠ eigen dim {d}. Skipping.")
        return None
    
    # Random null
    print("  Computing random token-pair null (N=1000)...")
    null = random_unembed_null(W_U, eigenvectors, cumvar, n_samples=1000)
    
    # Load concept unembed results
    concept_results = load_concept_unembed_results(model_name)
    concept_ginis = []
    if concept_results:
        for info in concept_results.values():
            concept_ginis.append(info["gini_deviation"])
    
    print(f"\n  {'':>25} {'N':>5} {'Mean Gini':>12}")
    print(f"  {'-' * 45}")
    print(f"  {'Random unembed pairs':<25} {null['n']:>5} {null['gini_mean']:>+12.4f}")
    if concept_ginis:
        print(f"  {'Concept unembed pairs':<25} {len(concept_ginis):>5} {np.mean(concept_ginis):>+12.4f}")
    
    # Statistical test: are concept ginis different from random ginis?
    result = {
        "model": model_name,
        "vocab_size": int(W_U.shape[0]),
        "hidden_dim": d,
        "null_gini_mean": null["gini_mean"],
        "null_gini_std": null["gini_std"],
        "null_n": null["n"],
    }
    
    if concept_ginis:
        from scipy import stats
        # One-sample t-test: is concept Gini different from null mean?
        t_stat, p_val = stats.ttest_1samp(concept_ginis, null["gini_mean"])
        result["concept_gini_mean"] = float(np.mean(concept_ginis))
        result["concept_vs_null_t"] = float(t_stat)
        result["concept_vs_null_p"] = float(p_val)
        
        print(f"\n  Concept vs Random: t={t_stat:.3f}, p={p_val:.4f}")
        
        if p_val < 0.05 and np.mean(concept_ginis) < null["gini_mean"]:
            print(f"  → Concept pairs concentrate LESS than random → concept structure exists")
        elif p_val < 0.05 and np.mean(concept_ginis) > null["gini_mean"]:
            print(f"  → Concept pairs concentrate MORE than random → unexpected")
        else:
            print(f"  → No significant difference → concentration is trivially expected")
    
    return result, null, cumvar


def main():
    parser = argparse.ArgumentParser(description="Hole 5 Fix: Random Unembedding Null")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    models = args.models or list(MODEL_PATHS.keys())
    
    print("=" * 70)
    print("  HOLE 5 FIX: Random Unembedding Null Model")
    print("=" * 70)
    
    all_results = []
    plot_data = {}
    
    for model_name in models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            continue
        
        try:
            out = run_model(model_name, model_path)
            if out:
                result, null, cumvar = out
                all_results.append(result)
                plot_data[model_name] = (null, cumvar)
        except Exception as e:
            print(f"  ERROR {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # ── Aggregate ──
    print(f"\n\n{'=' * 70}")
    print("  HOLE 5 AGGREGATE")
    print(f"{'=' * 70}")
    
    if all_results:
        null_ginis = [r["null_gini_mean"] for r in all_results]
        concept_ginis = [r["concept_gini_mean"] for r in all_results if "concept_gini_mean" in r]
        
        print(f"\n  Across {len(all_results)} models:")
        print(f"    Mean random-pair Gini:  {np.mean(null_ginis):+.4f}")
        if concept_ginis:
            print(f"    Mean concept-pair Gini: {np.mean(concept_ginis):+.4f}")
            print(f"    Difference:             {np.mean(concept_ginis) - np.mean(null_ginis):+.4f}")
    
    # ── Summary Plot ──
    if all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: Comparison bar chart
        ax = axes[0]
        model_names = [r["model"] for r in all_results if "concept_gini_mean" in r]
        concept_vals = [r["concept_gini_mean"] for r in all_results if "concept_gini_mean" in r]
        random_vals = [r["null_gini_mean"] for r in all_results if "concept_gini_mean" in r]
        
        x = np.arange(len(model_names))
        w = 0.35
        ax.bar(x - w/2, concept_vals, w, label="Concept pairs", color="#E63946", alpha=0.8)
        ax.bar(x + w/2, random_vals, w, label="Random pairs", color="#457B9D", alpha=0.8)
        ax.axhline(0, color="black", ls="-", lw=0.5)
        ax.set_ylabel("Mean Gini Deviation")
        ax.set_title("Unembedding Concentration:\nConcept vs Random Token Pairs")
        ax.set_xticks(x)
        ax.set_xticklabels([m[:10] for m in model_names], fontsize=7, rotation=45, ha="right")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")
        
        # Panel 2: Example CDF comparison (first model with data)
        ax = axes[1]
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4, label="Diagonal")
        
        for model_name, (null, cumvar) in list(plot_data.items())[:3]:
            if null["mean_cdf"] is not None:
                ax.plot(cumvar, null["mean_cdf"], lw=1.5, alpha=0.7, label=f"{model_name[:10]} random")
                ax.fill_between(cumvar, null["ci_lower"], null["ci_upper"], alpha=0.05)
        
        ax.set_xlabel("Cumulative Variance V(k)")
        ax.set_ylabel("Cumulative Energy C(k)")
        ax.set_title("Random Token-Pair\nUnembedding CDFs")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / "hole5_unembed_null.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved: {save_path}")
    
    # Save
    save_path = OUTPUT_DIR / "hole5_unembed_null.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  Results saved: {save_path}")


if __name__ == "__main__":
    main()
