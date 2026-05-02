#!/usr/bin/env python3
"""
Hole 3 Fix: SAE Feature Collapse
==================================

The original SAE artifact check (Exp 0A) used top-1 feature selection per concept.
This caused feature collapse: feature #11527 was selected for 6/27 concepts.

This fix:
  1. Reports unique feature counts for top-k=1 (the original run)
  2. Re-runs with top-k=5, averaging decoder vectors of the 5 most
     differential features per concept
  3. Compares CDFs and Gini deviations between top-1 and top-5
  4. Verifies that anti-concentration holds with the more robust extraction

Usage (on remote GPU):
    uv run python scripts/fix_hole3_sae_collapse.py --model gemma-2-2b
    uv run python scripts/fix_hole3_sae_collapse.py --model gemma-2-2b --top-k 5
    uv run python scripts/fix_hole3_sae_collapse.py --model gemma-2-9b --top-k 5
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from collections import Counter, defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

# Import from existing script
sys.path.append(os.path.join(os.path.dirname(__file__)))
from artifact_check_sae import (
    SAE_CONFIG, JumpReLUSAE, load_sae,
    get_residual_stream, eigendecompose_cached, compute_cdf,
)

RESULTS_DIR = Path("results")
SPECTRAL_DIR = Path("results/_spectral")
OUTPUT_DIR = Path("results/_spectral/hole_fixes")


def identify_sae_concept_features_v2(
    base_words, target_words,
    sae, model, tokenizer, handler, device,
    layer_idx, top_k=5
):
    """
    Improved feature identification:
    - Returns top-k feature indices (not just top-1)
    - Also returns per-feature statistics for quality assessment
    """
    print(f"      Getting activations for {len(base_words)} pairs...")
    
    pos_acts, neg_acts = [], []
    for bw, tw in zip(base_words, target_words):
        try:
            act_neg = get_residual_stream([bw], model, tokenizer, handler, device, layer_idx)
            act_pos = get_residual_stream([tw], model, tokenizer, handler, device, layer_idx)
            neg_acts.append(act_neg[0])
            pos_acts.append(act_pos[0])
        except Exception:
            continue
    
    if not pos_acts:
        return None, None, None, None
    
    X_pos = torch.stack(pos_acts)
    X_neg = torch.stack(neg_acts)
    
    sae_device = next(sae.parameters()).device
    with torch.no_grad():
        z_pos = sae.encode(X_pos.to(sae_device)).cpu().float()
        z_neg = sae.encode(X_neg.to(sae_device)).cpu().float()
    
    # Differential activation (mean)
    diff_act = z_pos.mean(dim=0) - z_neg.mean(dim=0)
    
    # Also compute effect size (Cohen's d) per feature for quality
    z_diff = z_pos - z_neg  # (n, d_sae)
    z_mean = z_diff.mean(dim=0)
    z_std = z_diff.std(dim=0).clamp(min=1e-8)
    effect_size = z_mean / z_std
    
    # Top-k by absolute differential activation
    top_indices = torch.argsort(diff_act.abs(), descending=True)[:top_k].tolist()
    
    # Get decoder vectors
    decoder_vecs = sae.W_dec.data[top_indices].cpu().float()
    
    # Feature quality stats
    feature_stats = {}
    for i, idx in enumerate(top_indices):
        feature_stats[idx] = {
            "rank": i,
            "diff_activation": float(diff_act[idx]),
            "abs_diff": float(diff_act[idx].abs()),
            "effect_size": float(effect_size[idx]),
            # Activation rate (fraction of samples where feature fires)
            "pos_fire_rate": float((z_pos[:, idx] > 0).float().mean()),
            "neg_fire_rate": float((z_neg[:, idx] > 0).float().mean()),
        }
    
    return top_indices, diff_act, decoder_vecs, feature_stats


def main():
    parser = argparse.ArgumentParser(description="Hole 3 Fix: SAE Feature Collapse")
    parser.add_argument("--model", default="gemma-2-2b", choices=list(SAE_CONFIG.keys()))
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--width", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    
    config = SAE_CONFIG[args.model]
    layer = args.layer or config["default_layer"]
    width = args.width or config["default_width"]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"  HOLE 3 FIX: SAE Feature Collapse (top-k={args.top_k})")
    print(f"  Model: {args.model}, Layer: {layer}, Width: {width}")
    print("=" * 70)
    
    # Load model
    print("\n[1] Loading model...")
    model, tokenizer, handler = lrg.load_model(config["model_path"])
    device = next(model.parameters()).device
    
    # Load SAE
    print("[2] Loading SAE...")
    sae, sae_method = load_sae(args.model, layer, width)
    sae = sae.to(device)
    sae.eval()
    
    # Load eigendecomposition
    eigenvalues, eigenvectors = eigendecompose_cached(args.model)
    cumvar = np.cumsum(eigenvalues.numpy()) / eigenvalues.numpy().sum()
    d = len(eigenvalues)
    
    # Process concepts
    print(f"\n[3] Processing concepts with top-k={args.top_k}...")
    concept_files = sorted(glob.glob("data/word_pairs/*.txt"))
    
    results = {}
    all_top1_features = []
    all_topk_features = []
    cdfs_topk = {}
    cdfs_top1 = {}
    
    for concept_file in concept_files:
        concept_name = os.path.basename(concept_file).replace('.txt', '')
        
        base_words, target_words = lrg.get_counterfactual_pairs(concept_file, tokenizer)
        if len(base_words) < 3:
            continue
        
        try:
            top_indices, diff_act, decoder_vecs, feature_stats = \
                identify_sae_concept_features_v2(
                    base_words, target_words,
                    sae, model, tokenizer, handler, device,
                    layer_idx=layer, top_k=args.top_k
                )
            
            if decoder_vecs is None:
                continue
            
            # Record top-1 feature
            all_top1_features.append(top_indices[0])
            all_topk_features.extend(top_indices)
            
            # Top-1 CDF (for comparison)
            v_top1 = decoder_vecs[0] / decoder_vecs[0].norm().clamp(min=1e-12)
            if v_top1.shape[0] == d:
                cdf_top1 = compute_cdf(v_top1, eigenvectors)
                cdfs_top1[concept_name] = cdf_top1
            
            # Top-k averaged CDF
            v_topk = decoder_vecs.mean(dim=0)
            v_topk = v_topk / v_topk.norm().clamp(min=1e-12)
            if v_topk.shape[0] == d:
                cdf_topk = compute_cdf(v_topk, eigenvectors)
                cdfs_topk[concept_name] = cdf_topk
            
            k_half = np.searchsorted(cumvar, 0.5)
            thi_top1 = 1.0 - cdf_top1[min(k_half, d-1)] if concept_name in cdfs_top1 else None
            thi_topk = 1.0 - cdf_topk[min(k_half, d-1)]
            gini_top1 = float(np.trapezoid(cdf_top1 - cumvar, cumvar)) if concept_name in cdfs_top1 else None
            gini_topk = float(np.trapezoid(cdf_topk - cumvar, cumvar))
            
            results[concept_name] = {
                "top1_feature": top_indices[0],
                "topk_features": top_indices,
                "n_pairs": len(base_words),
                "top_k": args.top_k,
                "gini_top1": gini_top1,
                "gini_topk": gini_topk,
                "thi_top1": thi_top1,
                "thi_topk": thi_topk,
                "feature_stats": {str(k): v for k, v in feature_stats.items()},
            }
            
            print(f"    {concept_name}: features={top_indices}, "
                  f"Gini(k=1)={gini_top1:+.4f}, Gini(k={args.top_k})={gini_topk:+.4f}")
        
        except Exception as e:
            print(f"    {concept_name}: ERROR — {e}")
    
    # Cleanup
    del model, sae
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ── Feature Collapse Analysis ──
    print(f"\n\n{'=' * 70}")
    print("  FEATURE COLLAPSE ANALYSIS")
    print(f"{'=' * 70}")
    
    n_concepts = len(all_top1_features)
    n_unique_top1 = len(set(all_top1_features))
    feature_counts = Counter(all_top1_features)
    
    print(f"\n  Top-1 Features:")
    print(f"    N concepts:        {n_concepts}")
    print(f"    N unique features: {n_unique_top1}")
    print(f"    Collapse ratio:    {n_unique_top1}/{n_concepts} = {n_unique_top1/n_concepts:.1%}")
    
    if n_unique_top1 < n_concepts:
        print(f"\n    Collapsed features (appearing >1 time):")
        for feat_id, count in feature_counts.most_common():
            if count > 1:
                concepts_with_feat = [c for c, r in results.items() if r["top1_feature"] == feat_id]
                print(f"      Feature #{feat_id}: {count}× → {concepts_with_feat}")
    
    n_unique_topk = len(set(all_topk_features))
    print(f"\n  Top-{args.top_k} Features:")
    print(f"    Total feature slots: {len(all_topk_features)}")
    print(f"    N unique features:   {n_unique_topk}")
    
    # ── Compare Gini: top-1 vs top-k ──
    ginis_top1 = [r["gini_top1"] for r in results.values() if r["gini_top1"] is not None]
    ginis_topk = [r["gini_topk"] for r in results.values()]
    
    from scipy import stats as sp_stats
    
    print(f"\n\n{'=' * 70}")
    print("  GINI COMPARISON: Top-1 vs Top-k")
    print(f"{'=' * 70}")
    
    print(f"\n  {'':>20} {'N':>5} {'Mean Gini':>12} {'t-stat':>10} {'p-value':>12}")
    print(f"  {'-' * 60}")
    
    t1, p1 = sp_stats.ttest_1samp(ginis_top1, 0.0) if ginis_top1 else (0, 1)
    tk, pk = sp_stats.ttest_1samp(ginis_topk, 0.0) if ginis_topk else (0, 1)
    
    print(f"  {'Top-1':<20} {len(ginis_top1):>5} {np.mean(ginis_top1):>+12.4f} "
          f"{t1:>10.3f} {p1:>12.2e}")
    print(f"  {'Top-' + str(args.top_k):<20} {len(ginis_topk):>5} {np.mean(ginis_topk):>+12.4f} "
          f"{tk:>10.3f} {pk:>12.2e}")
    
    # Excluding collapsed features
    non_collapsed = [c for c, r in results.items() if feature_counts[r["top1_feature"]] == 1]
    ginis_non_collapsed = [results[c]["gini_top1"] for c in non_collapsed
                          if results[c]["gini_top1"] is not None]
    if ginis_non_collapsed:
        tnc, pnc = sp_stats.ttest_1samp(ginis_non_collapsed, 0.0)
        print(f"  {'Top-1 (no collapse)':<20} {len(ginis_non_collapsed):>5} "
              f"{np.mean(ginis_non_collapsed):>+12.4f} {tnc:>10.3f} {pnc:>12.2e}")
    
    # ── Verdict ──
    print(f"\n\n{'=' * 70}")
    print("  HOLE 3 VERDICT")
    print(f"{'=' * 70}")
    
    if pk < 0.05 and np.mean(ginis_topk) < 0:
        print(f"\n  ✅ ANTI-CONCENTRATION HOLDS WITH TOP-{args.top_k}.")
        print(f"     Mean Gini (top-{args.top_k}): {np.mean(ginis_topk):+.4f} (p={pk:.2e})")
        print(f"     Feature collapse was a methodological weakness but NOT the driver.")
    else:
        print(f"\n  ⚠️  Top-{args.top_k} results are weaker or non-significant.")
        print(f"     Mean Gini: {np.mean(ginis_topk):+.4f} (p={pk:.2e})")
    
    # ── Comparison Plot ──
    if cdfs_top1 and cdfs_topk:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Panel 1: Top-1 CDFs
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
        for cname, cdf in cdfs_top1.items():
            ax.plot(cumvar, cdf, lw=0.8, alpha=0.5)
        if cdfs_top1:
            arr = np.array(list(cdfs_top1.values()))
            ax.plot(cumvar, arr.mean(axis=0), "r-", lw=2.5, label=f"Mean (N={len(cdfs_top1)})")
        ax.set_xlabel("Cumulative Variance V(k)")
        ax.set_ylabel("Cumulative Concept Energy C(k)")
        ax.set_title(f"Top-1 (Original)\nGini={np.mean(ginis_top1):+.4f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        
        # Panel 2: Top-k CDFs
        ax = axes[1]
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
        for cname, cdf in cdfs_topk.items():
            ax.plot(cumvar, cdf, lw=0.8, alpha=0.5)
        if cdfs_topk:
            arr = np.array(list(cdfs_topk.values()))
            ax.plot(cumvar, arr.mean(axis=0), "m-", lw=2.5, label=f"Mean (N={len(cdfs_topk)})")
        ax.set_xlabel("Cumulative Variance V(k)")
        ax.set_ylabel("Cumulative Concept Energy C(k)")
        ax.set_title(f"Top-{args.top_k}\nGini={np.mean(ginis_topk):+.4f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        
        # Panel 3: Feature usage histogram
        ax = axes[2]
        counts = sorted(feature_counts.values(), reverse=True)
        ax.bar(range(len(counts)), counts, color="#457B9D", alpha=0.8)
        ax.set_xlabel("Feature Rank (by # concepts using it)")
        ax.set_ylabel("# Concepts")
        ax.set_title(f"Feature Collapse: {n_unique_top1}/{n_concepts} unique (top-1)")
        ax.grid(True, alpha=0.2, axis="y")
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"hole3_sae_collapse_{args.model}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved: {save_path}")
    
    # Save
    save_path = OUTPUT_DIR / f"hole3_sae_collapse_{args.model}.json"
    with open(save_path, "w") as f:
        json.dump({
            "model": args.model,
            "top_k": args.top_k,
            "n_concepts": n_concepts,
            "n_unique_top1": n_unique_top1,
            "collapse_ratio": n_unique_top1 / n_concepts if n_concepts > 0 else 0,
            "mean_gini_top1": float(np.mean(ginis_top1)) if ginis_top1 else None,
            "mean_gini_topk": float(np.mean(ginis_topk)) if ginis_topk else None,
            "results": results,
        }, f, indent=2, default=float)
    print(f"  Results saved: {save_path}")


if __name__ == "__main__":
    main()
