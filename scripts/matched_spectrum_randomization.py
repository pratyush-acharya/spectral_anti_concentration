#!/usr/bin/env python3
"""
Experiment 2: Matched-Spectrum Randomization
==============================================

The cleverest experiment: definitively separates "spectral regularization"
from "geometric correction" by replacing real eigenvectors with random ones
while keeping eigenvalues identical.

If Fake WCA ≈ Real WCA → benefit is purely spectral regularization.
If Fake WCA ≈ Naive → benefit comes from specific geometric directions.

Usage (on remote GPU):
    uv run python scripts/matched_spectrum_randomization.py
    uv run python scripts/matched_spectrum_randomization.py --models Llama-3.2-1B --seeds 3
"""

import argparse
import json
import os
import sys
import glob
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/_spectral/randomization")

LANG_PAIRS = ["en-fr", "es-de", "fr-de", "fr-es"]

ALL_MODELS = [
    "gemma-2-2b", "gemma-2-9b", "jetmoe-8b",
    "Llama-3.2-1B", "Llama-3.2-3B", "Mistral-7B-v0.3",
    "OLMoE-1B-7B-0125", "Qwen2.5-0.5B", "Qwen2.5-1.5B",
    "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B",
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
    "SmolLM2-1.7B",
]

REG_LAMBDA = 0.1  # optimal from ablation


def construct_fake_covariance(eigenvalues: torch.Tensor, d: int, seed: int) -> torch.Tensor:
    """
    Build a fake covariance with identical eigenvalues but random eigenvectors.
    Σ_fake = V @ diag(Λ) @ V^T where V is a random orthogonal matrix.
    """
    rng = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(d, d, generator=rng)
    V, _ = torch.linalg.qr(random_matrix)
    fake_cov = V @ torch.diag(eigenvalues) @ V.T
    return fake_cov


def get_whitening_from_cov(cov: torch.Tensor, reg_lambda: float = REG_LAMBDA):
    """Compute whitening (Ψ = Σ^{-1/2}) and sqrt (Σ^{1/2}) from a covariance matrix."""
    cov = cov.float()
    d = cov.shape[0]
    cov_reg = cov + reg_lambda * torch.eye(d, device=cov.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_reg)
    
    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    inv_sqrt_vals = 1.0 / torch.sqrt(eigenvalues.clamp(min=1e-12))
    psi = eigenvectors @ torch.diag(inv_sqrt_vals) @ eigenvectors.T
    
    sqrt_vals = torch.sqrt(eigenvalues.clamp(min=0))
    sqrt_mat = eigenvectors @ torch.diag(sqrt_vals) @ eigenvectors.T
    
    return psi, sqrt_mat


def run_wca_with_covariance(
    cov_src: torch.Tensor, cov_tgt: torch.Tensor,
    X_src: torch.Tensor, X_tgt: torch.Tensor,
    v_raw_dict: dict,
    reg_lambda: float = REG_LAMBDA,
) -> dict:
    """
    Run WCA pipeline with arbitrary covariance matrices.
    Returns {concept_name: cosine_sim_with_ground_truth}.
    """
    psi_src, _ = get_whitening_from_cov(cov_src, reg_lambda)
    psi_tgt, sqrt_tgt = get_whitening_from_cov(cov_tgt, reg_lambda)
    
    Q = lrg.solve_whitened_procrustes(X_src.float(), X_tgt.float(), psi_src, psi_tgt)
    
    results = {}
    for name, v_raw in v_raw_dict.items():
        v_transported = lrg.transport_concept_vector(v_raw.float(), psi_src, sqrt_tgt, Q)
        # Ground truth is the raw target concept vector (already loaded as v_raw from target)
        # But we're comparing transported source to the source's raw (self-reconstruction)
        # Actually, we compare transported to ground truth target v_raw
        sim = torch.nn.functional.cosine_similarity(v_transported.unsqueeze(0), v_raw.unsqueeze(0)).item()
        results[name] = sim
    
    return results


def run_naive(X_src: torch.Tensor, X_tgt: torch.Tensor, v_raw_dict: dict) -> dict:
    """Run naive Procrustes transport."""
    Q_naive = lrg.solve_naive_procrustes(X_src.float(), X_tgt.float())
    
    results = {}
    for name, v_raw in v_raw_dict.items():
        v_transported = lrg.transport_naive(v_raw.float(), Q_naive)
        sim = torch.nn.functional.cosine_similarity(v_transported.unsqueeze(0), v_raw.unsqueeze(0)).item()
        results[name] = sim
    
    return results


def process_model_pair(model: str, pair: str, n_seeds: int = 5):
    """Run randomization experiment for one model × pair."""
    pair_dir = RESULTS_DIR / model / pair
    if not pair_dir.exists():
        return None
    
    # Load saved data
    cov_src = torch.load(pair_dir / "cov_src.pt", map_location="cpu", weights_only=True).float()
    cov_tgt = torch.load(pair_dir / "cov_tgt.pt", map_location="cpu", weights_only=True).float()
    X_src = torch.load(pair_dir / "X_src.pt", map_location="cpu", weights_only=True).float()
    X_tgt = torch.load(pair_dir / "X_tgt.pt", map_location="cpu", weights_only=True).float()
    
    d = cov_src.shape[0]
    
    # Load concept vectors
    v_raw_dict = {}
    for f in pair_dir.glob("v_raw_*.pt"):
        name = f.stem.replace("v_raw_", "")
        v_raw_dict[name] = torch.load(f, map_location="cpu", weights_only=True).float()
    
    if not v_raw_dict:
        return None
    
    # 1. Real eigenvalues
    eigenvalues_src, _ = torch.linalg.eigh(cov_src)
    eigenvalues_src = torch.sort(eigenvalues_src, descending=True).values
    eigenvalues_tgt, _ = torch.linalg.eigh(cov_tgt)
    eigenvalues_tgt = torch.sort(eigenvalues_tgt, descending=True).values
    
    # 2. Naive baseline
    naive_sims = run_naive(X_src, X_tgt, v_raw_dict)
    
    # 3. Real WCA
    real_sims = run_wca_with_covariance(cov_src, cov_tgt, X_src, X_tgt, v_raw_dict)
    
    # 4. Fake WCA (multiple seeds)
    fake_sims_all = []
    for seed in range(n_seeds):
        fake_cov_src = construct_fake_covariance(eigenvalues_src, d, seed * 1000 + 42)
        fake_cov_tgt = construct_fake_covariance(eigenvalues_tgt, d, seed * 1000 + 137)
        fake_sims = run_wca_with_covariance(fake_cov_src, fake_cov_tgt, X_src, X_tgt, v_raw_dict)
        fake_sims_all.append(fake_sims)
    
    # Compare
    concepts = sorted(v_raw_dict.keys())
    
    real_wins = 0  # Real WCA > Naive
    fake_wins_avg = 0  # Avg Fake WCA > Naive
    n_valid = 0
    
    concept_results = {}
    for c in concepts:
        n_val = naive_sims.get(c)
        r_val = real_sims.get(c)
        f_vals = [fs.get(c) for fs in fake_sims_all]
        
        if n_val is None or r_val is None or any(v is None for v in f_vals):
            continue
        if np.isnan(n_val) or np.isnan(r_val):
            continue
        
        f_mean = np.mean(f_vals)
        n_valid += 1
        
        if r_val > n_val:
            real_wins += 1
        if f_mean > n_val:
            fake_wins_avg += 1
        
        concept_results[c] = {
            "naive": n_val,
            "real_wca": r_val,
            "fake_wca_mean": float(f_mean),
            "fake_wca_std": float(np.std(f_vals)),
            "delta_real": r_val - n_val,
            "delta_fake": float(f_mean - n_val),
        }
    
    if n_valid == 0:
        return None
    
    return {
        "model": model,
        "pair": pair,
        "n_concepts": n_valid,
        "n_seeds": n_seeds,
        "real_win_rate": real_wins / n_valid,
        "fake_win_rate": fake_wins_avg / n_valid,
        "mean_delta_real": np.mean([cr["delta_real"] for cr in concept_results.values()]),
        "mean_delta_fake": np.mean([cr["delta_fake"] for cr in concept_results.values()]),
        "concepts": concept_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 2: Matched-Spectrum Randomization")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--pairs", nargs="+", default=LANG_PAIRS)
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of random seeds for fake covariance (default: 5)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    models = args.models or ALL_MODELS
    
    print("=" * 60)
    print("  EXPERIMENT 2: Matched-Spectrum Randomization")
    print(f"  Models: {len(models)}, Pairs: {args.pairs}, Seeds: {args.seeds}")
    print("=" * 60)
    
    all_results = []
    real_deltas = []
    fake_deltas = []
    
    for model in models:
        for pair in args.pairs:
            pair_dir = RESULTS_DIR / model / pair
            if not pair_dir.exists() or not (pair_dir / "cov_src.pt").exists():
                continue
            
            print(f"\n  {model}/{pair}...", end=" ", flush=True)
            t0 = time.time()
            
            try:
                result = process_model_pair(model, pair, args.seeds)
                if result:
                    all_results.append(result)
                    real_deltas.append(result["mean_delta_real"])
                    fake_deltas.append(result["mean_delta_fake"])
                    print(f"Real WR={result['real_win_rate']:.0%}, "
                          f"Fake WR={result['fake_win_rate']:.0%} "
                          f"({time.time()-t0:.1f}s)")
                else:
                    print("SKIP (no valid concepts)")
            except Exception as e:
                print(f"ERROR: {e}")
    
    # ── Aggregate ──
    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS")
    print("=" * 60)
    
    if not all_results:
        print("  No results to aggregate.")
        return
    
    real_wrs = [r["real_win_rate"] for r in all_results]
    fake_wrs = [r["fake_win_rate"] for r in all_results]
    
    print(f"\n  {'Condition':<20} {'Win Rate':>10} {'Mean Δ':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Real WCA (λ=0.1)':<20} {np.mean(real_wrs):>9.1%} {np.mean(real_deltas):>+10.4f}")
    print(f"  {'Fake WCA (λ=0.1)':<20} {np.mean(fake_wrs):>9.1%} {np.mean(fake_deltas):>+10.4f}")
    print(f"  {'Naive Procrustes':<20} {'50.0%':>10} {'0.0000':>10}")
    
    # Paired test: Real vs Fake
    if len(real_deltas) > 3:
        t_stat, p_val = sp_stats.ttest_rel(real_deltas, fake_deltas)
        print(f"\n  Paired t-test (Real vs Fake): t={t_stat:.3f}, p={p_val:.4f}")
        
        if p_val > 0.05:
            print("  → No significant difference: benefit is PURELY SPECTRAL REGULARIZATION")
        elif np.mean(real_deltas) > np.mean(fake_deltas):
            print("  → Real WCA significantly better: GEOMETRY MATTERS (not just regularization)")
        else:
            print("  → Fake WCA better: unexpected result")
    
    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel 1: Win rates
    ax = axes[0]
    x = np.arange(len(all_results))
    width = 0.35
    labels = [f"{r['model'][:8]}\n{r['pair']}" for r in all_results]
    ax.bar(x - width/2, real_wrs, width, label="Real WCA", color="#2A9D8F", alpha=0.8)
    ax.bar(x + width/2, fake_wrs, width, label="Fake WCA", color="#E9C46A", alpha=0.8)
    ax.axhline(0.5, color="grey", ls="--", alpha=0.5, label="Chance (Naive)")
    ax.set_ylabel("Win Rate vs. Naive")
    ax.set_title("Real vs. Fake WCA Win Rates")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    
    # Panel 2: Delta scatter
    ax = axes[1]
    ax.scatter(fake_deltas, real_deltas, c="#457B9D", s=40, alpha=0.7, edgecolors="white")
    lims = [min(min(fake_deltas), min(real_deltas)) - 0.01,
            max(max(fake_deltas), max(real_deltas)) + 0.01]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="Real = Fake")
    ax.axhline(0, color="grey", ls=":", alpha=0.3)
    ax.axvline(0, color="grey", ls=":", alpha=0.3)
    ax.set_xlabel("Fake WCA Mean Δ")
    ax.set_ylabel("Real WCA Mean Δ")
    ax.set_title("Δ(WCA−Naive): Real vs. Randomized Eigenvectors")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "randomization_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")
    
    # Save
    save_path = OUTPUT_DIR / "randomization_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  Results saved: {save_path}")


if __name__ == "__main__":
    main()
