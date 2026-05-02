#!/usr/bin/env python3
"""
Experiment 0C: Linear Probe Artifact Check
===========================================

Trains logistic regression probes on raw (UN-centered) activations to get
concept directions that are independent of the difference-of-means pipeline.

L2-regularized logistic regression has an OPPOSING bias — it tends toward
high-variance directions. If probe directions STILL anti-concentrate,
the evidence is very strong.

Usage (on remote GPU):
    uv run python scripts/artifact_check_probe.py
    uv run python scripts/artifact_check_probe.py --models gemma-2-9b Llama-3.2-3B Qwen2.5-3B
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

MODEL_PATHS = {
    "gemma-2-9b": "google/gemma-2-9b",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
}


def get_activations_for_word_list(words, model, tokenizer, handler, device, batch_size=4):
    """Get residual stream activations for a list of words."""
    return handler.get_pooled_embeddings(
        model, tokenizer, words,
        target_words=words,
        device=device,
        batch_size=batch_size,
    )


def train_probe(X_pos, X_neg):
    """
    Train L2-regularized logistic regression on RAW (un-centered) activations.
    Returns the weight vector (concept direction).
    
    CRITICAL: Do NOT mean-center X. That reintroduces the subtraction artifact.
    """
    from sklearn.linear_model import LogisticRegression
    
    X = torch.cat([X_pos, X_neg], dim=0).cpu().numpy().astype(np.float32)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    clf = LogisticRegression(
        C=1.0, fit_intercept=True,
        max_iter=1000, solver='lbfgs',
        l1_ratio=0
    )
    clf.fit(X, y)
    
    v_probe = torch.tensor(clf.coef_[0], dtype=torch.float32)
    v_probe = v_probe / v_probe.norm().clamp(min=1e-12)
    
    accuracy = clf.score(X, y)
    return v_probe, accuracy


def eigendecompose_cached(model_name, lang_pair="en-fr"):
    """Load cached eigendecomposition from Exp 1."""
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


def run_model(model_name, model_path):
    """Process one model."""
    import gc
    
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print("  Loading model...")
    model, tokenizer, handler = lrg.load_model(model_path)
    device = next(model.parameters()).device
    print(f"  Device: {device}")
    
    # Load eigendecomposition
    eigenvalues, eigenvectors = eigendecompose_cached(model_name)
    cumvar = np.cumsum(eigenvalues.numpy()) / eigenvalues.numpy().sum()
    d = len(eigenvalues)
    
    # Process concepts
    concept_files = sorted(glob.glob("data/word_pairs/*.txt"))
    results = {}
    probe_cdfs = {}
    
    for concept_file in concept_files:
        concept_name = os.path.basename(concept_file).replace('.txt', '')
        
        # Load word pairs
        base_words, target_words = lrg.get_counterfactual_pairs(concept_file, tokenizer)
        if len(base_words) < 5:
            print(f"    {concept_name}: SKIP (only {len(base_words)} pairs)")
            continue
        
        try:
            # Get activations (NO mean-centering!)
            with torch.no_grad():
                X_pos = get_activations_for_word_list(
                    target_words, model, tokenizer, handler, device
                ).cpu().float()
                X_neg = get_activations_for_word_list(
                    base_words, model, tokenizer, handler, device
                ).cpu().float()
            
            if X_pos.shape[1] != d:
                print(f"    {concept_name}: dim mismatch ({X_pos.shape[1]} vs {d})")
                continue
            
            # Train probe
            v_probe, accuracy = train_probe(X_pos, X_neg)
            
            # Compute CDF
            cdf = compute_cdf(v_probe, eigenvectors)
            probe_cdfs[concept_name] = cdf
            
            # Metrics
            k_half = np.searchsorted(cumvar, 0.5)
            thi = 1.0 - cdf[min(k_half, len(cdf)-1)]
            gini = float(np.trapezoid(cdf - cumvar, cumvar))
            
            results[concept_name] = {
                "n_pairs": len(base_words),
                "probe_accuracy": float(accuracy),
                "thi": float(thi),
                "gini_deviation": gini,
            }
            
            print(f"    {concept_name}: acc={accuracy:.3f}, "
                  f"THI={thi:.4f}, Gini={gini:+.4f}")
        
        except Exception as e:
            print(f"    {concept_name}: ERROR — {e}")
    
    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results, probe_cdfs, cumvar


def plot_probe_comparison(model_name, probe_cdfs, cumvar):
    """Plot probe CDFs alongside diff-of-means CDFs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Probe CDFs
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
    for cname, cdf in probe_cdfs.items():
        short = cname.replace("[", "").replace("]", "").strip()
        ax.plot(cumvar, cdf, lw=1, alpha=0.6, label=short)
    ax.set_xlabel("Cumulative Variance V(k)")
    ax.set_ylabel("Cumulative Concept Energy C(k)")
    ax.set_title(f"{model_name}\nLinear Probe Concept CDFs")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.2)
    
    # Panel 2: Aggregate comparison
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4, label="Diagonal")
    
    if probe_cdfs:
        arr = np.array(list(probe_cdfs.values()))
        mean_cdf = arr.mean(axis=0)
        ax.plot(cumvar, mean_cdf, "g-", lw=2.5, label=f"Probe (N={len(probe_cdfs)})")
        ax.fill_between(cumvar, arr.min(axis=0), arr.max(axis=0), color="green", alpha=0.1)
    
    # Load diff-of-means from Exp 1
    cdf_path = SPECTRAL_DIR / "cdfs" / f"{model_name}_en-fr.json"
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
    
    # Load unembed from Exp 0B if available
    unembed_path = OUTPUT_DIR / f"artifact_check_{model_name}.json"
    # (not plotting unembed here to avoid circular dependency, keep it simple)
    
    ax.set_xlabel("Cumulative Variance V(k)")
    ax.set_ylabel("Cumulative Concept Energy C(k)")
    ax.set_title(f"{model_name}\nProbe vs. Diff-of-Means")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"artifact_check_probe_{model_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp 0C: Linear Probe Artifact Check")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    models = args.models or list(MODEL_PATHS.keys())
    
    print("=" * 60)
    print("  EXPERIMENT 0C: Linear Probe Artifact Check")
    print("=" * 60)
    
    all_ginis = []
    all_results = {}
    
    for model_name in models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            print(f"\n  SKIP {model_name}: no model path")
            continue
        
        try:
            results, probe_cdfs, cumvar = run_model(model_name, model_path)
            all_results[model_name] = results
            
            if probe_cdfs:
                plot_probe_comparison(model_name, probe_cdfs, cumvar)
                for info in results.values():
                    all_ginis.append(info["gini_deviation"])
        
        except Exception as e:
            print(f"\n  ERROR {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Verdict
    print("\n" + "=" * 60)
    print("  PROBE ARTIFACT CHECK VERDICT")
    print("=" * 60)
    
    if all_ginis:
        from scipy import stats as sp_stats
        mean_gini = np.mean(all_ginis)
        t_stat, p_val = sp_stats.ttest_1samp(all_ginis, 0.0)
        
        print(f"  N: {len(all_ginis)}")
        print(f"  Mean Gini deviation: {mean_gini:+.4f}")
        print(f"  t={t_stat:.3f}, p={p_val:.2e}")
        
        if p_val < 0.05 and mean_gini < 0:
            print(f"\n  ✅ PROBE DIRECTIONS ALSO ANTI-CONCENTRATE.")
            print(f"     Despite L2 bias toward high-variance, concepts still land in the tail.")
            print(f"     → VERY strong evidence that anti-concentration is real.")
        else:
            print(f"\n  ⚠️  Probes do NOT show anti-concentration.")
            print(f"     This may be due to L2 bias pulling toward high-variance directions.")
    
    save_path = OUTPUT_DIR / "artifact_check_probe.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
