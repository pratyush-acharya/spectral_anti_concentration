#!/usr/bin/env python3
"""
Hole 1 Fix: SCM Ceiling Effect Confound
=========================================

The SCM (Spectral Center of Mass) can be inflated toward 1.0 when eigenspectra
become more top-heavy at larger scales. This script computes the SCM of random
baseline vectors alongside concept vectors to check whether the scaling trend
(Exp 4) is real or an artifact.

Reports: concept SCM, random SCM, and the GAP = (concept SCM - random SCM)
at each model scale. Also recomputes using Gini deviation (which is already
baseline-corrected against the diagonal) as the primary metric.

Usage:
    uv run python scripts/fix_hole1_scm_ceiling.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__)))
from spectral_analysis import (
    load_covariance, eigendecompose, compute_cumulative_variance,
    compute_concept_energy_cdf, compute_spectral_center_of_mass,
    compute_gini_deviation, load_concept_vectors,
    CONCEPT_GROUPS, GROUP_COLORS, EIGEN_CACHE_DIR, PLOTS_DIR,
    ALL_MODELS, LANG_PAIR, QWEN25_SCALE, QWEN25_PARAMS, QWEN3_SCALE, QWEN3_PARAMS,
    RESULTS_DIR,
)

OUTPUT_DIR = Path("results/_spectral/hole_fixes")


def compute_random_scm(eigenvectors: torch.Tensor, cumvar: np.ndarray,
                       n_samples: int = 5000, seed: int = 42) -> dict:
    """
    Compute SCM and Gini deviation for random unit vectors.
    Returns {"scm_mean", "scm_std", "gini_mean", "gini_std", "scm_values"}.
    """
    d = eigenvectors.shape[0]
    rng = np.random.default_rng(seed)
    
    scms = []
    ginis = []
    
    for _ in range(n_samples):
        v = torch.tensor(rng.standard_normal(d), dtype=torch.float32)
        v = v / v.norm()
        cdf = compute_concept_energy_cdf(v, eigenvectors)
        scms.append(compute_spectral_center_of_mass(cdf, cumvar))
        ginis.append(compute_gini_deviation(cdf, cumvar))
    
    return {
        "scm_mean": float(np.mean(scms)),
        "scm_std": float(np.std(scms)),
        "scm_median": float(np.median(scms)),
        "scm_p95": float(np.percentile(scms, 95)),
        "gini_mean": float(np.mean(ginis)),
        "gini_std": float(np.std(ginis)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  HOLE 1 FIX: SCM Ceiling Effect Check")
    print("=" * 70)
    
    # ── Part 1: All models SCM comparison ──
    print("\n[1] Computing random baseline SCM for all models...")
    
    header = (f"  {'Model':<22} {'Dim':>5} {'Concept SCM':>12} {'Random SCM':>12} "
              f"{'GAP':>8} {'Concept Gini':>14} {'Random Gini':>13}")
    print(header)
    print(f"  {'-' * len(header)}")
    
    all_data = []
    
    for model in ALL_MODELS:
        try:
            cov = load_covariance(model)
            d = cov.shape[0]
            cache_path = EIGEN_CACHE_DIR / model / LANG_PAIR / "eigen.pt"
            eigenvalues, eigenvectors = eigendecompose(cov, cache_path)
            cumvar = compute_cumulative_variance(eigenvalues)
            
            # Concept SCM/Gini
            concepts = load_concept_vectors(model)
            concept_scms = []
            concept_ginis = []
            for _, v in concepts.items():
                cdf = compute_concept_energy_cdf(v, eigenvectors)
                concept_scms.append(compute_spectral_center_of_mass(cdf, cumvar))
                concept_ginis.append(compute_gini_deviation(cdf, cumvar))
            
            # Random SCM/Gini
            rand = compute_random_scm(eigenvectors, cumvar, n_samples=5000)
            
            gap = np.mean(concept_scms) - rand["scm_mean"]
            
            row = {
                "model": model, "d": d,
                "concept_scm": float(np.mean(concept_scms)),
                "concept_scm_std": float(np.std(concept_scms)),
                "random_scm": rand["scm_mean"],
                "random_scm_std": rand["scm_std"],
                "scm_gap": gap,
                "concept_gini": float(np.mean(concept_ginis)),
                "random_gini": rand["gini_mean"],
            }
            all_data.append(row)
            
            print(f"  {model:<22} {d:>5} {row['concept_scm']:>12.4f} "
                  f"{row['random_scm']:>12.4f} {gap:>+8.4f} "
                  f"{row['concept_gini']:>+14.4f} {rand['gini_mean']:>+13.6f}")
        
        except Exception as e:
            print(f"  {model:<22} ERROR: {e}")
    
    # ── Part 2: Qwen Scaling with corrected metric ──
    print(f"\n\n[2] Qwen Scaling — Corrected Metrics")
    print("=" * 70)
    
    for family, model_list, param_list in [
        ("Qwen2.5", QWEN25_SCALE, QWEN25_PARAMS),
        ("Qwen3", QWEN3_SCALE, QWEN3_PARAMS),
    ]:
        print(f"\n  {family} Scaling:")
        print(f"  {'Params':>8} | {'Concept SCM':>12} {'Random SCM':>12} {'SCM Gap':>10} | "
              f"{'Concept Gini':>13} {'Random Gini':>12} {'Gini Gap':>10}")
        print(f"  {'-' * 90}")
        
        scaling_gaps = defaultdict(list)  # group -> [(param, gap_mean, gap_std)]
        scaling_gini = defaultdict(list)  # group -> [(param, gini_mean, gini_std)]
        
        for model, param_B in zip(model_list, param_list):
            row = next((r for r in all_data if r["model"] == model), None)
            if row is None:
                print(f"  {param_B:>7.1f}B | N/A")
                continue
            
            gini_gap = row["concept_gini"] - row["random_gini"]
            print(f"  {param_B:>7.1f}B | {row['concept_scm']:>12.4f} {row['random_scm']:>12.4f} "
                  f"{row['scm_gap']:>+10.4f} | {row['concept_gini']:>+13.4f} "
                  f"{row['random_gini']:>+12.6f} {gini_gap:>+10.4f}")
        
        # Plot corrected scaling with both SCM Gap and Gini
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        family_models = [(m, p) for m, p in zip(model_list, param_list)]
        family_data = [(next((r for r in all_data if r["model"] == m), None), p)
                       for m, p in family_models]
        family_data = [(r, p) for r, p in family_data if r is not None]
        
        if family_data:
            params = [p for _, p in family_data]
            concept_scms = [r["concept_scm"] for r, _ in family_data]
            random_scms = [r["random_scm"] for r, _ in family_data]
            scm_gaps = [r["scm_gap"] for r, _ in family_data]
            concept_ginis = [r["concept_gini"] for r, _ in family_data]
            random_ginis = [r["random_gini"] for r, _ in family_data]
            
            # Panel 1: SCM with random baseline
            ax = axes[0]
            ax.plot(params, concept_scms, "o-", color="#E63946", lw=2, ms=8,
                    label="Concept SCM", zorder=3)
            ax.plot(params, random_scms, "s--", color="grey", lw=2, ms=8,
                    label="Random SCM", zorder=3)
            ax.fill_between(params, random_scms, concept_scms,
                           color="#E63946", alpha=0.15, label="Gap")
            ax.set_xlabel("Parameters (B)", fontsize=12)
            ax.set_ylabel("Spectral Center of Mass", fontsize=12)
            ax.set_title(f"{family} — SCM with Random Baseline", fontsize=13)
            ax.legend(fontsize=9)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.2)
            ax.set_ylim(0, 1.05)
            
            # Panel 2: Gini deviation (already corrected)
            ax = axes[1]
            ax.plot(params, concept_ginis, "o-", color="#457B9D", lw=2, ms=8,
                    label="Concept Gini Dev")
            ax.axhline(0, color="grey", ls="--", lw=1, alpha=0.5, label="Random baseline (≈0)")
            ax.set_xlabel("Parameters (B)", fontsize=12)
            ax.set_ylabel("Gini Deviation (negative = anti-concentrated)", fontsize=12)
            ax.set_title(f"{family} — Gini Deviation (baseline-corrected)", fontsize=13)
            ax.legend(fontsize=9)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            save_path = OUTPUT_DIR / f"hole1_scaling_corrected_{family}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  Saved: {save_path}")
    
    # ── Part 3: Verdict ──
    print(f"\n\n{'=' * 70}")
    print("  HOLE 1 VERDICT")
    print(f"{'=' * 70}")
    
    if all_data:
        scm_gaps = [r["scm_gap"] for r in all_data]
        random_scms = [r["random_scm"] for r in all_data]
        concept_scms = [r["concept_scm"] for r in all_data]
        
        print(f"\n  Across all {len(all_data)} models:")
        print(f"    Mean concept SCM:  {np.mean(concept_scms):.4f}")
        print(f"    Mean random SCM:   {np.mean(random_scms):.4f}")
        print(f"    Mean SCM gap:      {np.mean(scm_gaps):+.4f}")
        
        # Correlation between random SCM and concept SCM
        from scipy import stats
        r, p = stats.pearsonr(random_scms, concept_scms)
        print(f"\n    Corr(random SCM, concept SCM): r={r:.4f}, p={p:.2e}")
        
        if r > 0.8 and p < 0.05:
            print(f"\n  ⚠️  HIGH CORRELATION: concept SCM tracks random SCM.")
            print(f"     The SCM scaling trend is CONFOUNDED by eigenspectrum shape.")
            print(f"     → Use Gini deviation as the primary metric instead.")
        else:
            print(f"\n  ✅ SCM gap is meaningful — concepts move independently of baseline.")
    
    # Save
    save_path = OUTPUT_DIR / "hole1_scm_ceiling.json"
    with open(save_path, "w") as f:
        json.dump(all_data, f, indent=2, default=float)
    print(f"\n  Results saved: {save_path}")


if __name__ == "__main__":
    main()
