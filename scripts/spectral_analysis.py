#!/usr/bin/env python3
"""
Spectral Analysis of Concept Representations
=============================================

Covers:
  - Experiment 1: Spectral Stratification CDFs
  - Experiment 4: Qwen Scaling Curves
  - Experiment 5: Gemma vs Llama Spectral Comparison

All operations use saved .pt files (cov_src.pt, v_raw_*.pt).
No GPU or model loading required.

Usage:
    uv run python scripts/spectral_analysis.py
    uv run python scripts/spectral_analysis.py --models gemma-2-2b gemma-2-9b
    uv run python scripts/spectral_analysis.py --experiment qwen_scaling
    uv run python scripts/spectral_analysis.py --experiment gemma_vs_llama
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
import matplotlib.gridspec as gridspec
from scipy import stats

# ─── Configuration ───────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/_spectral")
EIGEN_CACHE_DIR = OUTPUT_DIR / "eigen_cache"
CDFS_DIR = OUTPUT_DIR / "cdfs"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Concept grouping (from experimental-plan.md)
CONCEPT_GROUPS = {
    "Verb Morphology": [
        "[verb - Ved]", "[verb - Ving]", "[verb - 3pSg]",
        "[verb - V + er]", "[verb - V + able]", "[verb - V + ment]",
        "[verb - V + tion]", "[Ving - Ved]", "[Ving - 3pSg]", "[3pSg - Ved]",
    ],
    "Semantic": [
        "[male - female]", "[country - capital]", "[thing - color]",
        "[thing - part]", "[small - big]", "[lower - upper]",
        "[frequent - infrequent]",
    ],
    "Grammatical": [
        "[adj - comparative]", "[adj - superlative]", "[adj - un + adj]",
        "[adj - adj + ly]", "[noun - plural]", "[pronoun - possessive]",
    ],
    "Language Pairs": [
        "[English - French]", "[French - German]",
        "[French - Spanish]", "[German - Spanish]",
    ],
}

# Flat lookup: concept_name -> group
CONCEPT_TO_GROUP = {}
for group, concepts in CONCEPT_GROUPS.items():
    for c in concepts:
        CONCEPT_TO_GROUP[c] = group

GROUP_COLORS = {
    "Verb Morphology": "#E63946",   # red
    "Semantic": "#457B9D",           # steel blue
    "Grammatical": "#2A9D8F",       # teal
    "Language Pairs": "#E9C46A",    # gold
}

# Qwen scaling models (Exp 4)
QWEN25_SCALE = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B"]
QWEN25_PARAMS = [0.5, 1.5, 3.0, 7.0, 14.0]  # in billions

QWEN3_SCALE = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
QWEN3_PARAMS = [0.6, 1.7, 4.0, 8.0]

# All models expected
ALL_MODELS = [
    "gemma-2-2b", "gemma-2-9b", "jetmoe-8b",
    "Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.2-3B", "Mistral-7B-v0.3",
    "OLMoE-1B-7B-0125", "Qwen2.5-0.5B", "Qwen2.5-1.5B",
    "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B",
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
    "SmolLM2-1.7B",
]

LANG_PAIR = "en-fr"  # primary analysis pair


# ─── Core Functions ──────────────────────────────────────────────────────────

def load_covariance(model_name: str, lang_pair: str = LANG_PAIR) -> torch.Tensor:
    """Load covariance matrix, trying new format first, then old."""
    new_path = RESULTS_DIR / model_name / lang_pair / "cov_src.pt"
    old_path = RESULTS_DIR / model_name / "cov_en.pt"

    if new_path.exists():
        return torch.load(new_path, map_location="cpu", weights_only=True).float()
    elif old_path.exists():
        return torch.load(old_path, map_location="cpu", weights_only=True).float()
    else:
        raise FileNotFoundError(f"No covariance found for {model_name}/{lang_pair}")


def eigendecompose(cov: torch.Tensor, cache_path: Path = None) -> tuple:
    """
    Eigendecompose Σ = U Λ U^T, sorted descending.
    Returns (eigenvalues_desc, eigenvectors) where eigenvectors[:,i] corresponds to eigenvalues[i].
    Caches to disk if cache_path is provided.
    """
    if cache_path and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["eigenvalues"], data["eigenvectors"]

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}, cache_path)

    return eigenvalues, eigenvectors


def compute_cumulative_variance(eigenvalues: torch.Tensor) -> np.ndarray:
    """Compute V(k) = Σ_{i=1}^{k} λ_i / Tr(Σ), the cumulative variance fraction."""
    evals = eigenvalues.numpy()
    cumvar = np.cumsum(evals) / np.sum(evals)
    return cumvar


def compute_concept_energy_cdf(v: torch.Tensor, eigenvectors: torch.Tensor) -> np.ndarray:
    """
    Compute spectral energy CDF of concept vector v against eigenvectors.
    E_i = (v^T u_i)^2 / ||v||^2
    C(k) = Σ_{i=1}^{k} E_i
    """
    v = v.float()
    projections = eigenvectors.T @ v  # (d,) — projection onto each eigenvector
    energy = (projections ** 2) / (v @ v).clamp(min=1e-12)  # normalize
    cdf = torch.cumsum(energy, dim=0).detach().cpu().numpy()
    return cdf


def generate_random_baseline(d: int, n_samples: int = 10000, eigenvectors: torch.Tensor = None) -> dict:
    """
    Generate random baseline CDFs.
    Returns dict with 'mean', 'ci_lower' (2.5th), 'ci_upper' (97.5th).
    """
    if eigenvectors is None:
        # For random vectors projected onto an orthonormal basis,
        # the CDF is just the cumulative sum of squared projections of random unit vectors.
        # Theoretically this should be close to y=x (uniform), but we compute empirically.
        eigenvectors = torch.eye(d)

    rng = np.random.default_rng(42)
    all_cdfs = np.zeros((n_samples, d))

    for i in range(n_samples):
        v = torch.tensor(rng.standard_normal(d), dtype=torch.float32)
        v = v / v.norm()
        cdf = compute_concept_energy_cdf(v, eigenvectors)
        all_cdfs[i] = cdf

    return {
        "mean": np.mean(all_cdfs, axis=0),
        "ci_lower": np.percentile(all_cdfs, 2.5, axis=0),
        "ci_upper": np.percentile(all_cdfs, 97.5, axis=0),
    }


def load_concept_vectors(model_name: str, lang_pair: str = LANG_PAIR) -> dict:
    """Load all v_raw_*.pt concept vectors for a model. Returns {concept_name: tensor}."""
    # Try new format first
    search_dir = RESULTS_DIR / model_name / lang_pair
    if not search_dir.exists():
        search_dir = RESULTS_DIR / model_name

    vectors = {}
    for f in sorted(search_dir.glob("v_raw_*.pt")):
        # Extract concept name: v_raw_[noun - plural].pt -> [noun - plural]
        name = f.stem.replace("v_raw_", "")
        vectors[name] = torch.load(f, map_location="cpu", weights_only=True).float()

    return vectors


def compute_spectral_center_of_mass(concept_cdf: np.ndarray, cumvar: np.ndarray) -> float:
    """
    Compute the spectral center of mass: the V(k) value at which
    cumulative concept energy C(k) reaches 50%.
    Low SCM = concept in high-variance (top of spectrum)
    High SCM = concept in low-variance (tail)
    """
    idx = np.searchsorted(concept_cdf, 0.5)
    if idx >= len(cumvar):
        return 1.0
    return float(cumvar[idx])


def compute_gini_deviation(concept_cdf: np.ndarray, cumvar: np.ndarray) -> float:
    """
    Compute area between CDF and diagonal.
    Negative = anti-concentrated (below diagonal = tail-heavy)
    Positive = concentrated (above diagonal = top-heavy)
    Uses trapezoidal integration on V(k) x-axis.
    """
    # Interpolate concept CDF onto uniform V(k) grid
    deviation = concept_cdf - cumvar  # positive = above diagonal (top-concentrated)
    # Integrate using trapezoidal rule with V(k) as x-axis
    area = np.trapezoid(deviation, cumvar)
    return float(area)


# ─── Plotting Functions ─────────────────────────────────────────────────────

def plot_stratification_single_model(
    cdfs_by_group: dict,
    cumvar: np.ndarray,
    random_baseline: dict,
    model_name: str,
    save_path: Path,
):
    """Plot stratification CDFs for one model. One curve per concept group + random baseline."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Random baseline band
    ax.fill_between(
        cumvar, random_baseline["ci_lower"], random_baseline["ci_upper"],
        color="grey", alpha=0.15, label="Random 95% CI"
    )
    ax.plot(cumvar, random_baseline["mean"], color="grey", ls="--", lw=1, alpha=0.5)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)

    # Group CDFs (mean ± std across concepts in group)
    for group_name, concept_cdfs in cdfs_by_group.items():
        if not concept_cdfs:
            continue
        arr = np.array(concept_cdfs)
        mean_cdf = arr.mean(axis=0)
        std_cdf = arr.std(axis=0)
        color = GROUP_COLORS.get(group_name, "black")

        ax.plot(cumvar, mean_cdf, color=color, lw=2, label=f"{group_name} (N={len(concept_cdfs)})")
        ax.fill_between(cumvar, mean_cdf - std_cdf, mean_cdf + std_cdf, color=color, alpha=0.1)

    ax.set_xlabel("Cumulative Variance V(k)", fontsize=12)
    ax.set_ylabel("Cumulative Concept Energy C(k)", fontsize=12)
    ax.set_title(f"Spectral Stratification — {model_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_aggregate_stratification(
    all_model_cdfs: dict,
    save_path: Path,
):
    """
    Plot aggregate CDFs across all models.
    all_model_cdfs: {group_name: list of (cumvar, cdf) tuples across all models×concepts}
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Diagonal
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4, label="Uniform (diagonal)")

    for group_name, entries in all_model_cdfs.items():
        if not entries:
            continue
        color = GROUP_COLORS.get(group_name, "black")

        # Interpolate all CDFs onto a common V(k) grid
        v_grid = np.linspace(0, 1, 500)
        interp_cdfs = []
        for cumvar, cdf in entries:
            interp_cdf = np.interp(v_grid, cumvar, cdf)
            interp_cdfs.append(interp_cdf)

        arr = np.array(interp_cdfs)
        mean_cdf = arr.mean(axis=0)
        ci_lo = np.percentile(arr, 2.5, axis=0)
        ci_hi = np.percentile(arr, 97.5, axis=0)

        ax.plot(v_grid, mean_cdf, color=color, lw=2.5, label=f"{group_name} (N={len(entries)})")
        ax.fill_between(v_grid, ci_lo, ci_hi, color=color, alpha=0.12)

    ax.set_xlabel("Cumulative Variance V(k)", fontsize=12)
    ax.set_ylabel("Cumulative Concept Energy C(k)", fontsize=12)
    ax.set_title("Aggregate Spectral Stratification (All Models)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_qwen_scaling(scaling_data: dict, save_path: Path, family: str = "Qwen2.5"):
    """
    Plot spectral center of mass vs. model size.
    scaling_data: {group_name: [(param_B, scm_mean, scm_std), ...]}
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for group_name, points in scaling_data.items():
        if not points:
            continue
        params, means, stds = zip(*points)
        color = GROUP_COLORS.get(group_name, "black")
        ax.errorbar(params, means, yerr=stds, color=color, marker="o", lw=2,
                    capsize=4, label=group_name)

    ax.set_xlabel("Parameters (B)", fontsize=12)
    ax.set_ylabel("Spectral Center of Mass (V at 50% energy)", fontsize=12)
    ax.set_title(f"{family} Scaling — Concept Spectral Position", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.2)

    # Higher SCM = deeper in tail
    ax.annotate("← High-variance\n(top of spectrum)", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=8, color="grey")
    ax.annotate("Low-variance →\n(spectral tail)", xy=(0.02, 0.9),
                xycoords="axes fraction", fontsize=8, color="grey")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_gemma_vs_llama(eigenvalues_dict: dict, cdfs_dict: dict, save_path: Path):
    """
    Side-by-side comparison of Gemma vs Llama eigenspectra and concept CDFs.
    """
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

    # Panel 1: Eigenvalue spectra (log-log)
    ax1 = fig.add_subplot(gs[0])
    colors = {"gemma-2-2b": "#E63946", "gemma-2-9b": "#A8201A",
              "Llama-3.2-1B": "#457B9D", "Llama-3.2-3B": "#1D3557"}
    for model, evals in eigenvalues_dict.items():
        color = colors.get(model, "grey")
        ax1.plot(np.arange(1, len(evals) + 1), evals, color=color, lw=1.5, label=model)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Eigenvalue Rank")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Eigenvalue Spectra")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Concept CDFs for Gemma
    ax2 = fig.add_subplot(gs[1])
    ax2.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
    for model in ["gemma-2-2b", "gemma-2-9b"]:
        if model not in cdfs_dict:
            continue
        cumvar, group_cdfs = cdfs_dict[model]
        for gname, mean_cdf in group_cdfs.items():
            color = GROUP_COLORS.get(gname, "grey")
            ls = "-" if model == "gemma-2-2b" else "--"
            ax2.plot(cumvar, mean_cdf, color=color, ls=ls, lw=1.5)
    ax2.set_xlabel("Cumulative Variance V(k)")
    ax2.set_ylabel("Cumulative Concept Energy C(k)")
    ax2.set_title("Gemma Concept CDFs")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.2)

    # Panel 3: Concept CDFs for Llama
    ax3 = fig.add_subplot(gs[2])
    ax3.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
    for model in ["Llama-3.2-1B", "Llama-3.2-3B"]:
        if model not in cdfs_dict:
            continue
        cumvar, group_cdfs = cdfs_dict[model]
        for gname, mean_cdf in group_cdfs.items():
            color = GROUP_COLORS.get(gname, "grey")
            ls = "-" if model == "Llama-3.2-1B" else "--"
            ax3.plot(cumvar, mean_cdf, color=color, ls=ls, lw=1.5)
    ax3.set_xlabel("Cumulative Variance V(k)")
    ax3.set_ylabel("Cumulative Concept Energy C(k)")
    ax3.set_title("Llama Concept CDFs")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.2)

    plt.suptitle("Gemma vs. Llama: Spectral Structure Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Analysis Runners ────────────────────────────────────────────────────────

def run_single_model(model_name: str, lang_pair: str = LANG_PAIR) -> dict:
    """
    Run full spectral analysis for one model.
    Returns dict with per-concept CDFs, SCMs, Gini deviations, etc.
    """
    print(f"\n{'='*60}")
    print(f"  Model: {model_name} | Pair: {lang_pair}")
    print(f"{'='*60}")

    # Load data
    cov = load_covariance(model_name, lang_pair)
    d = cov.shape[0]
    print(f"  Hidden dim: {d}")

    # Eigendecompose (with caching)
    cache_path = EIGEN_CACHE_DIR / model_name / lang_pair / "eigen.pt"
    eigenvalues, eigenvectors = eigendecompose(cov, cache_path)
    print(f"  Eigenvalues: min={eigenvalues[-1]:.2e}, max={eigenvalues[0]:.2e}, "
          f"cond={eigenvalues[0]/eigenvalues[-1].clamp(min=1e-12):.2e}")

    # Cumulative variance
    cumvar = compute_cumulative_variance(eigenvalues)

    # Random baseline
    print(f"  Computing random baseline (d={d})...")
    baseline = generate_random_baseline(d, n_samples=5000, eigenvectors=eigenvectors)

    # Load concepts
    concepts = load_concept_vectors(model_name, lang_pair)
    print(f"  Loaded {len(concepts)} concept vectors")

    # Compute CDFs and statistics for each concept
    results = {
        "model": model_name,
        "lang_pair": lang_pair,
        "hidden_dim": d,
        "condition_number": float(eigenvalues[0] / eigenvalues[-1].clamp(min=1e-12)),
        "concepts": {},
    }

    cdfs_by_group = defaultdict(list)
    all_scms = defaultdict(list)
    all_ginis = defaultdict(list)

    for cname, v in concepts.items():
        cdf = compute_concept_energy_cdf(v, eigenvectors)
        scm = compute_spectral_center_of_mass(cdf, cumvar)
        gini = compute_gini_deviation(cdf, cumvar)
        group = CONCEPT_TO_GROUP.get(cname, "Unknown")

        results["concepts"][cname] = {
            "group": group,
            "scm": scm,
            "gini_deviation": gini,
        }

        cdfs_by_group[group].append(cdf)
        all_scms[group].append(scm)
        all_ginis[group].append(gini)

    # Print summary
    print(f"\n  {'Group':<20} {'SCM (mean±std)':<20} {'Gini Dev (mean)':<18} N")
    print(f"  {'-'*70}")
    for group in CONCEPT_GROUPS:
        if group in all_scms:
            scms = all_scms[group]
            ginis = all_ginis[group]
            print(f"  {group:<20} {np.mean(scms):.4f} ± {np.std(scms):.4f}    "
                  f"{np.mean(ginis):+.4f}          {len(scms)}")

    # Overall
    all_scm_flat = [s for v in all_scms.values() for s in v]
    all_gini_flat = [g for v in all_ginis.values() for g in v]
    print(f"  {'OVERALL':<20} {np.mean(all_scm_flat):.4f} ± {np.std(all_scm_flat):.4f}    "
          f"{np.mean(all_gini_flat):+.4f}          {len(all_scm_flat)}")

    # One-sample t-test on Gini deviation (is it significantly different from 0?)
    t_stat, p_val = stats.ttest_1samp(all_gini_flat, 0.0)
    print(f"\n  Gini deviation t-test: t={t_stat:.3f}, p={p_val:.2e} "
          f"({'SIGNIFICANT' if p_val < 0.05 else 'not significant'})")

    # Plot
    plot_path = PLOTS_DIR / f"stratification_{model_name}.png"
    plot_stratification_single_model(cdfs_by_group, cumvar, baseline, model_name, plot_path)

    # Save CDF data
    cdf_save = {
        "cumvar": cumvar.tolist(),
        "baseline_mean": baseline["mean"].tolist(),
        "baseline_ci_lower": baseline["ci_lower"].tolist(),
        "baseline_ci_upper": baseline["ci_upper"].tolist(),
    }
    for group, concept_cdfs in cdfs_by_group.items():
        cdf_save[f"group_{group}"] = [c.tolist() for c in concept_cdfs]

    cdf_path = CDFS_DIR / f"{model_name}_{lang_pair}.json"
    cdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cdf_path, "w") as f:
        json.dump(cdf_save, f)

    return {
        "results": results,
        "cdfs_by_group": cdfs_by_group,
        "cumvar": cumvar,
        "eigenvalues": eigenvalues.numpy(),
    }


def run_experiment_1(models: list = None, lang_pair: str = LANG_PAIR):
    """
    Experiment 1: Spectral Stratification CDFs.
    Runs all models and produces per-model + aggregate plots.
    """
    if models is None:
        models = ALL_MODELS

    print("=" * 60)
    print("  EXPERIMENT 1: Spectral Stratification CDFs")
    print("=" * 60)

    all_results = {}
    aggregate_cdfs = defaultdict(list)  # group -> [(cumvar, cdf)]
    summary_rows = []

    for model in models:
        if not (RESULTS_DIR / model / lang_pair / "cov_src.pt").exists() and \
           not (RESULTS_DIR / model / "cov_en.pt").exists():
            print(f"\n  SKIP {model}: no covariance data")
            continue

        try:
            out = run_single_model(model, lang_pair)
            all_results[model] = out

            # Collect for aggregate
            for group, concept_cdfs in out["cdfs_by_group"].items():
                for cdf in concept_cdfs:
                    aggregate_cdfs[group].append((out["cumvar"], cdf))

            # Summary row
            r = out["results"]
            all_ginis = [c["gini_deviation"] for c in r["concepts"].values()]
            all_scms = [c["scm"] for c in r["concepts"].values()]
            summary_rows.append({
                "model": model,
                "hidden_dim": r["hidden_dim"],
                "cond_number": r["condition_number"],
                "mean_scm": float(np.mean(all_scms)),
                "mean_gini": float(np.mean(all_ginis)),
                "n_concepts": len(all_ginis),
            })

        except Exception as e:
            print(f"\n  ERROR {model}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate plot
    if aggregate_cdfs:
        plot_aggregate_stratification(aggregate_cdfs, PLOTS_DIR / "aggregate_stratification.png")

    # Summary table
    print("\n" + "=" * 80)
    print("  AGGREGATE SUMMARY")
    print("=" * 80)
    print(f"  {'Model':<22} {'Dim':>5} {'Cond#':>10} {'Mean SCM':>10} {'Mean Gini':>10}")
    print(f"  {'-'*62}")
    for row in summary_rows:
        print(f"  {row['model']:<22} {row['hidden_dim']:>5} {row['cond_number']:>10.1e} "
              f"{row['mean_scm']:>10.4f} {row['mean_gini']:>+10.4f}")

    # Cross-model statistical test per group
    print(f"\n  CROSS-MODEL GROUP COMPARISONS (Gini deviation)")
    print(f"  {'-'*60}")

    group_ginis = defaultdict(list)
    for model, out in all_results.items():
        for cname, info in out["results"]["concepts"].items():
            group_ginis[info["group"]].append(info["gini_deviation"])

    for group in CONCEPT_GROUPS:
        if group in group_ginis:
            gvals = group_ginis[group]
            t_stat, p_val = stats.ttest_1samp(gvals, 0.0)
            print(f"  {group:<20}: mean={np.mean(gvals):+.4f}, "
                  f"t={t_stat:.2f}, p={p_val:.2e} "
                  f"{'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

    # Pairwise group comparisons (is Verb different from Semantic?)
    print(f"\n  PAIRWISE GROUP COMPARISONS (Wilcoxon)")
    print(f"  {'-'*60}")
    group_names = [g for g in CONCEPT_GROUPS if g in group_ginis]
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            try:
                stat, p_val = stats.mannwhitneyu(group_ginis[g1], group_ginis[g2], alternative="two-sided")
                print(f"  {g1} vs {g2}: U={stat:.0f}, p={p_val:.3f} "
                      f"{'*' if p_val < 0.05 else 'ns'}")
            except Exception:
                print(f"  {g1} vs {g2}: insufficient data")

    # Save summary
    summary_path = OUTPUT_DIR / "experiment1_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "models": summary_rows,
            "group_statistics": {
                g: {
                    "mean_gini": float(np.mean(group_ginis[g])),
                    "std_gini": float(np.std(group_ginis[g])),
                    "n": len(group_ginis[g]),
                    "t_stat": float(stats.ttest_1samp(group_ginis[g], 0.0)[0]),
                    "p_value": float(stats.ttest_1samp(group_ginis[g], 0.0)[1]),
                }
                for g in CONCEPT_GROUPS if g in group_ginis
            },
        }, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    return all_results


def run_experiment_4(all_results: dict = None):
    """
    Experiment 4: Qwen Scaling Curves.
    Shows how spectral position changes with model size.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Qwen Scaling Curves")
    print("=" * 60)

    for family, model_list, param_list in [
        ("Qwen2.5", QWEN25_SCALE, QWEN25_PARAMS),
        ("Qwen3", QWEN3_SCALE, QWEN3_PARAMS),
    ]:
        scaling_data = defaultdict(list)  # group -> [(param_B, scm_mean, scm_std)]

        for model, param_B in zip(model_list, param_list):
            if all_results and model in all_results:
                out = all_results[model]
            else:
                try:
                    out = run_single_model(model)
                except Exception as e:
                    print(f"  SKIP {model}: {e}")
                    continue

            # Compute per-group SCM
            for group in CONCEPT_GROUPS:
                scms = [
                    info["scm"]
                    for info in out["results"]["concepts"].values()
                    if info["group"] == group
                ]
                if scms:
                    scaling_data[group].append((param_B, np.mean(scms), np.std(scms)))

        if scaling_data:
            plot_qwen_scaling(scaling_data, PLOTS_DIR / f"scaling_{family}.png", family)

            # Print scaling table
            print(f"\n  {family} Scaling Table:")
            print(f"  {'Params':>8} | ", end="")
            for g in CONCEPT_GROUPS:
                print(f"{g[:12]:>14}", end="")
            print()
            for i, (model, param_B) in enumerate(zip(model_list, param_list)):
                print(f"  {param_B:>7.1f}B | ", end="")
                for g in CONCEPT_GROUPS:
                    points = scaling_data.get(g, [])
                    match = [p for p in points if p[0] == param_B]
                    if match:
                        print(f"  {match[0][1]:.4f}±{match[0][2]:.3f}", end="")
                    else:
                        print(f"  {'N/A':>12}", end="")
                print()


def run_experiment_5(all_results: dict = None):
    """
    Experiment 5: Gemma vs Llama Spectral Comparison.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: Gemma vs Llama Comparison")
    print("=" * 60)

    target_models = ["gemma-2-2b", "gemma-2-9b", "Llama-3.2-1B", "Llama-3.2-3B"]
    eigenvalues_dict = {}
    cdfs_dict = {}

    for model in target_models:
        if all_results and model in all_results:
            out = all_results[model]
        else:
            try:
                out = run_single_model(model)
            except Exception as e:
                print(f"  SKIP {model}: {e}")
                continue

        eigenvalues_dict[model] = out["eigenvalues"]

        # Compute mean CDF per group
        cumvar = out["cumvar"]
        group_mean_cdfs = {}
        for group, concept_cdfs in out["cdfs_by_group"].items():
            if concept_cdfs:
                group_mean_cdfs[group] = np.mean(concept_cdfs, axis=0)
        cdfs_dict[model] = (cumvar, group_mean_cdfs)

    if eigenvalues_dict:
        plot_gemma_vs_llama(eigenvalues_dict, cdfs_dict, PLOTS_DIR / "gemma_vs_llama.png")

        # Print spectral comparison
        print(f"\n  {'Model':<18} {'Dim':>5} {'Cond#':>12} {'Top 10% Energy':>16} {'Top 50% Energy':>16}")
        print(f"  {'-'*72}")
        for model in target_models:
            if model in eigenvalues_dict:
                evals = eigenvalues_dict[model]
                d = len(evals)
                total = evals.sum()
                top10 = evals[:int(0.1*d)].sum() / total
                top50 = evals[:int(0.5*d)].sum() / total
                cond = evals[0] / evals[-1] if evals[-1] > 0 else float("inf")
                print(f"  {model:<18} {d:>5} {cond:>12.1e} {top10:>15.1%} {top50:>15.1%}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spectral Analysis of Concept Representations")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to analyze (default: all)")
    parser.add_argument("--experiment", choices=["all", "exp1", "qwen_scaling", "gemma_vs_llama"],
                       default="all", help="Which experiment to run")
    parser.add_argument("--lang-pair", default=LANG_PAIR, help="Language pair (default: en-fr)")
    args = parser.parse_args()

    # Create output dirs
    for d in [OUTPUT_DIR, EIGEN_CACHE_DIR, CDFS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    models = args.models or ALL_MODELS

    if args.experiment in ("all", "exp1"):
        all_results = run_experiment_1(models, args.lang_pair)
    else:
        all_results = None

    if args.experiment in ("all", "qwen_scaling"):
        run_experiment_4(all_results)

    if args.experiment in ("all", "gemma_vs_llama"):
        run_experiment_5(all_results)

    print("\n" + "=" * 60)
    print("  DONE. All outputs in: results/_spectral/")
    print("=" * 60)


if __name__ == "__main__":
    main()
