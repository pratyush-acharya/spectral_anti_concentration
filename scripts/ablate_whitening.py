#!/usr/bin/env python3
"""
WCA Hyperparameter Ablation Study.

Sweeps reg_lambda, truncate_k, and centering to find optimal WCA configuration.
Runs fully offline from saved .pt files; computes covariances from model if needed.

Usage:
    python scripts/ablate_whitening.py results/Qwen3.5-0.8B
    python scripts/ablate_whitening.py results/Qwen3.5-0.8B --model Qwen/Qwen3.5-0.8B
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
import os
import glob
import csv
import argparse
import warnings
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

sns.set_theme(context="paper", style="white", palette="colorblind", font="sans-serif", font_scale=1.3)

# ──────────────────────────────────────────────────────────
# Ablation Grid
# ──────────────────────────────────────────────────────────
REG_LAMBDAS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
REG_LABELS = ['0', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1']
TRUNCATE_FRACS = [1.0, 0.9, 0.75, 0.5]
CENTERING_OPTIONS = [False, True]

# ──────────────────────────────────────────────────────────
# Concept Grouping (shared with analyze_raid.py)
# ──────────────────────────────────────────────────────────
CONCEPT_GROUPS = {
    'Verb Morphology': ['[verb - Ved]', '[verb - Ving]', '[verb - 3pSg]', '[verb - V + er]',
                        '[verb - V + able]', '[verb - V + ment]', '[verb - V + tion]',
                        '[Ving - Ved]', '[Ving - 3pSg]', '[3pSg - Ved]'],
    'Adj Morphology':  ['[adj - comparative]', '[adj - superlative]', '[adj - un + adj]', '[adj - adj + ly]'],
    'Noun Morphology': ['[noun - plural]', '[pronoun - possessive]'],
    'Language Pairs':   ['[English - French]', '[French - German]', '[French - Spanish]', '[German - Spanish]'],
    'Semantic':        ['[male - female]', '[country - capital]', '[thing - color]', '[thing - part]',
                        '[small - big]', '[lower - upper]', '[frequent - infrequent]'],
}
GROUP_COLORS = {
    'Verb Morphology': '#e74c3c',
    'Adj Morphology':  '#3498db',
    'Noun Morphology': '#2ecc71',
    'Language Pairs':   '#9b59b6',
    'Semantic':        '#f39c12',
}


def get_group(name):
    for g, members in CONCEPT_GROUPS.items():
        if name in members:
            return g
    return 'Other'


def sort_concept_names(names):
    ordered = []
    for group_concepts in CONCEPT_GROUPS.values():
        for c in group_concepts:
            if c in names:
                ordered.append(c)
    for c in names:
        if c not in ordered:
            ordered.append(c)
    return ordered


def cosine_sim(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    denom = torch.norm(a) * torch.norm(b)
    if denom < 1e-12:
        return 0.0
    return (torch.dot(a, b) / denom).item()


# ──────────────────────────────────────────────────────────
# Quiet Whitening (no print statements)
# ──────────────────────────────────────────────────────────
def whitening_transform_quiet(cov, reg_lambda=1e-4, truncate_k=None):
    """Compute whitening + sqrt transforms without printing."""
    cov = cov.to(torch.float32).clone()
    eps = max(reg_lambda, 1e-12)
    cov += eps * torch.eye(cov.shape[0], device=cov.device)

    vals, vecs = torch.linalg.eigh(cov)
    # Sort descending
    vals = torch.flip(vals, [0])
    vecs = torch.flip(vecs, [1])

    if truncate_k is not None and truncate_k < len(vals):
        vals = vals[:truncate_k]
        vecs = vecs[:, :truncate_k]

    inv_sqrt_vals = 1.0 / torch.sqrt(torch.clamp(vals, min=1e-12))
    inv_sqrt_mat = vecs @ torch.diag(inv_sqrt_vals) @ vecs.T

    sqrt_vals = torch.sqrt(torch.clamp(vals, min=0))
    sqrt_mat = vecs @ torch.diag(sqrt_vals) @ vecs.T

    return inv_sqrt_mat, sqrt_mat


# ──────────────────────────────────────────────────────────
# Covariance Loading / Extraction
# ──────────────────────────────────────────────────────────
def load_covariances(result_dir, model_path=None):
    """Load covariances, computing from model if necessary."""
    # Try new naming first, then legacy
    for src_name, tgt_name in [("cov_src.pt", "cov_tgt.pt"), ("cov_en.pt", "cov_fr.pt")]:
        src_path = os.path.join(result_dir, src_name)
        tgt_path = os.path.join(result_dir, tgt_name)
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            print(f"  Loading covariance matrices ({src_name}, {tgt_name})...")
            return (torch.load(src_path, map_location='cpu', weights_only=True),
                    torch.load(tgt_path, map_location='cpu', weights_only=True))

    # Need to compute from model
    if model_path is None:
        model_name = os.path.basename(result_dir)
        model_path = f"Qwen/{model_name}"
        print(f"  No covariances found. Trying model: {model_path}")

    print(f"  Loading {model_path} on CPU for covariance extraction...")
    import json
    from tqdm import tqdm

    model, tokenizer, handler = lrg.load_model(model_path, device_map="cpu")

    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "paired_contexts", "en-fr.jsonl")
    pairs = []
    with open(data_file, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))

    vocab_en, vocab_fr = set(), set()
    for p in tqdm(pairs, desc="  Tokenizing for vocab"):
        for text in p["contexts0"]:
            fmt = handler.format_input(text, "en", "fr", tokenizer)
            vocab_en.update(tokenizer.encode(fmt, add_special_tokens=False))
        for text in p["contexts1"]:
            fmt = handler.format_input(text, "fr", "en", tokenizer)
            vocab_fr.update(tokenizer.encode(fmt, add_special_tokens=False))

    W_U = lrg.get_lm_head(model).weight.detach().float()
    cov_en = lrg.compute_covariance_matrix(W_U, token_indices=list(vocab_en))
    cov_fr = lrg.compute_covariance_matrix(W_U, token_indices=list(vocab_fr))

    torch.save(cov_en.cpu(), cov_en_path)
    torch.save(cov_fr.cpu(), cov_fr_path)
    print(f"  Saved covariances to {result_dir}")

    del model, tokenizer, handler, W_U
    import gc; gc.collect()

    return cov_en.cpu(), cov_fr.cpu()


# ──────────────────────────────────────────────────────────
# Naive Baseline
# ──────────────────────────────────────────────────────────
def compute_naive(X_en, X_fr, raw_vectors, concept_names, centered=False):
    """Compute naive Procrustes cosine sims."""
    if centered:
        X_en = X_en - X_en.mean(dim=0)
        X_fr = X_fr - X_fr.mean(dim=0)
    Q = lrg.solve_naive_procrustes(X_en, X_fr)
    return {name: cosine_sim(raw_vectors[name], lrg.transport_naive(raw_vectors[name], Q))
            for name in concept_names}


# ──────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────
def sweep(cov_en, cov_fr, X_en, X_fr, raw_vectors, concept_names, d,
          reg_lambdas, truncate_fracs, centering_options):
    """
    Sweep all (reg_lambda, truncate_k, centering) configurations.
    Returns: dict mapping (reg_lambda, truncate_k_int, centered) -> {concept: sim}
    """
    X_en_c = X_en - X_en.mean(dim=0)
    X_fr_c = X_fr - X_fr.mean(dim=0)

    truncate_ks = [None if f == 1.0 else int(d * f) for f in truncate_fracs]
    configs = list(product(reg_lambdas, truncate_ks, centering_options))
    total = len(configs)

    print(f"  Sweeping {total} configurations...")
    results = {}

    for i, (rl, tk, center) in enumerate(configs):
        tk_key = tk if tk is not None else d
        config_key = (rl, tk_key, center)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                psi_en, _ = whitening_transform_quiet(cov_en, reg_lambda=rl, truncate_k=tk)
                psi_fr, sqrt_fr = whitening_transform_quiet(cov_fr, reg_lambda=rl, truncate_k=tk)

            Xs, Xf = (X_en_c, X_fr_c) if center else (X_en, X_fr)
            Q = lrg.solve_whitened_procrustes(Xs, Xf, psi_en, psi_fr)

            sims = {}
            for name in concept_names:
                v_raid = lrg.transport_concept_vector(raw_vectors[name], psi_en, sqrt_fr, Q)
                sims[name] = cosine_sim(raw_vectors[name], v_raid)

            results[config_key] = sims

        except Exception as e:
            print(f"    Config ({rl}, {tk}, {center}) FAILED: {e}")
            results[config_key] = {name: float('nan') for name in concept_names}

        if (i + 1) % 14 == 0 or (i + 1) == total:
            print(f"    {i+1}/{total} done")

    return results


def find_best(results, concept_names):
    best_config, best_mean = None, -float('inf')
    for config, sims in results.items():
        m = np.nanmean([sims[c] for c in concept_names])
        if m > best_mean:
            best_mean, best_config = m, config
    return best_config, best_mean


# ──────────────────────────────────────────────────────────
# CSV Output
# ──────────────────────────────────────────────────────────
def save_csv(results, concept_names, naive_sims, naive_c_sims, reg_lambdas, output_dir):
    path = os.path.join(output_dir, "ablation_results.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['reg_lambda', 'truncate_k', 'centered', 'mean_sim'] + concept_names)

        # Naive baselines
        nv = [naive_sims[c] for c in concept_names]
        w.writerow(['naive', 'N/A', False, f"{np.mean(nv):.6f}"] + [f"{v:.6f}" for v in nv])
        nvc = [naive_c_sims[c] for c in concept_names]
        w.writerow(['naive', 'N/A', True, f"{np.mean(nvc):.6f}"] + [f"{v:.6f}" for v in nvc])

        for config in sorted(results.keys()):
            sims = results[config]
            vals = [sims[c] for c in concept_names]
            m = np.nanmean(vals)
            w.writerow([config[0], config[1], config[2], f"{m:.6f}"] + [f"{v:.6f}" for v in vals])

    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────
def _build_grid(results, concept_names, reg_lambdas, d, truncate_fracs, centered):
    """Build 2D numpy array: rows=reg_lambda, cols=truncate_frac, values=mean sim."""
    grid = np.zeros((len(reg_lambdas), len(truncate_fracs)))
    for i, rl in enumerate(reg_lambdas):
        for j, frac in enumerate(truncate_fracs):
            tk = d if frac == 1.0 else int(d * frac)
            config = (rl, tk, centered)
            if config in results:
                grid[i, j] = np.nanmean([results[config][c] for c in concept_names])
            else:
                grid[i, j] = float('nan')
    return grid


def plot_heatmaps(results, concept_names, reg_lambdas, reg_labels, d,
                  truncate_fracs, naive_mean, naive_c_mean, output_dir):
    """Two-panel heatmap: uncentered (left) and centered (right)."""
    trunc_labels = [f"{d}" if f == 1.0 else f"{int(d*f)}" for f in truncate_fracs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for ax_idx, centered in enumerate([False, True]):
        ax = axes[ax_idx]
        grid = _build_grid(results, concept_names, reg_lambdas, d, truncate_fracs, centered)
        baseline = naive_c_mean if centered else naive_mean

        sns.heatmap(grid, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    xticklabels=trunc_labels, yticklabels=reg_labels if ax_idx == 0 else False,
                    linewidths=0.5, linecolor='white',
                    vmin=min(grid.min(), baseline) - 0.02,
                    vmax=max(grid.max(), baseline) + 0.02)

        ax.set_xlabel('truncate_k')
        if ax_idx == 0:
            ax.set_ylabel('reg_lambda')
        title = 'Centered' if centered else 'Uncentered'
        ax.set_title(f'{title}\n(Naive baseline: {baseline:.4f})', fontsize=11)

        # Mark current config (1e-3, full, this centering)
        curr_rl_idx = reg_lambdas.index(1e-3)
        curr_tk_idx = 0  # full
        ax.add_patch(plt.Rectangle((curr_tk_idx, curr_rl_idx), 1, 1,
                                    fill=False, edgecolor='cyan', linewidth=3))

    fig.suptitle('WCA Mean Cosine Similarity Across Hyperparameters', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_heatmap.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: ablation_heatmap.png")


def plot_lines(results, concept_names, reg_lambdas, reg_labels, d,
               truncate_fracs, naive_mean, naive_c_mean, output_dir):
    """Line chart: mean sim vs reg_lambda, one line per truncate_k."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x = np.arange(len(reg_lambdas))
    trunc_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for ax_idx, centered in enumerate([False, True]):
        ax = axes[ax_idx]
        baseline = naive_c_mean if centered else naive_mean

        for j, frac in enumerate(truncate_fracs):
            tk = d if frac == 1.0 else int(d * frac)
            label = f"k={d}" if frac == 1.0 else f"k={int(d*frac)}"
            y_vals = []
            for rl in reg_lambdas:
                config = (rl, tk, centered)
                if config in results:
                    y_vals.append(np.nanmean([results[config][c] for c in concept_names]))
                else:
                    y_vals.append(float('nan'))
            ax.plot(x, y_vals, 'o-', color=trunc_colors[j], label=label, linewidth=2, markersize=5)

        ax.axhline(y=baseline, color='black', linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'Naive ({baseline:.4f})')

        # Mark current config
        curr_idx = reg_lambdas.index(1e-3)
        curr_config = (1e-3, d, centered)
        if curr_config in results:
            curr_val = np.nanmean([results[curr_config][c] for c in concept_names])
            ax.plot(curr_idx, curr_val, '*', color='cyan', markersize=15, zorder=5,
                    markeredgecolor='black', markeredgewidth=0.5, label='Current (1e-3)')

        ax.set_xticks(x)
        ax.set_xticklabels(reg_labels, rotation=45, ha='right')
        ax.set_xlabel('reg_lambda')
        if ax_idx == 0:
            ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('Centered' if centered else 'Uncentered', fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle('WCA Performance vs Regularization Strength', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_lines.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: ablation_lines.png")


def plot_best_vs_naive(best_sims, naive_sims, concept_names, best_config, output_dir):
    """Per-concept comparison: WCA (best config) vs Naive."""
    # Sort by WCA advantage
    advantages = {c: best_sims[c] - naive_sims[c] for c in concept_names}
    sorted_concepts = sorted(concept_names, key=lambda c: advantages[c], reverse=True)

    n = len(sorted_concepts)
    fig, ax = plt.subplots(figsize=(12, max(7, n * 0.35)))
    y = np.arange(n)
    h = 0.35

    wca_vals = [best_sims[c] for c in sorted_concepts]
    naive_vals = [naive_sims[c] for c in sorted_concepts]
    colors = [GROUP_COLORS.get(get_group(c), '#95a5a6') for c in sorted_concepts]

    ax.barh(y - h/2, wca_vals, h, label=f'WCA (best)', color=colors, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.barh(y + h/2, naive_vals, h, label='Naive', color=colors, alpha=0.35,
            edgecolor='white', linewidth=0.5, hatch='///')

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_concepts, fontsize=9)
    ax.set_xlabel('Cosine Similarity (Raw vs Transported)')
    rl, tk, cen = best_config
    ax.set_title(f'WCA (λ={rl}, k={tk}, {"centered" if cen else "uncentered"}) vs Naive Procrustes',
                 fontsize=11)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    # Annotate advantage
    for i, c in enumerate(sorted_concepts):
        adv = advantages[c]
        color = '#2ecc71' if adv > 0 else '#e74c3c'
        ax.annotate(f'{adv:+.3f}', xy=(max(wca_vals[i], naive_vals[i]) + 0.005, i),
                    fontsize=7, color=color, va='center')

    ax.legend(loc='lower right', fontsize=9)

    # Group legend
    for g, clr in GROUP_COLORS.items():
        ax.plot([], [], 's', color=clr, label=g, markersize=6)
    ax.legend(fontsize=7, loc='lower right', ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_vs_naive.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: best_vs_naive.png")


def plot_advantage_heatmap(results, concept_names, reg_lambdas, reg_labels, d,
                           truncate_fracs, naive_sims, output_dir):
    """Heatmap of (WCA_mean - Naive_mean) — green = WCA wins."""
    naive_mean = np.mean(list(naive_sims.values()))
    trunc_labels = [f"{d}" if f == 1.0 else f"{int(d*f)}" for f in truncate_fracs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for ax_idx, centered in enumerate([False, True]):
        ax = axes[ax_idx]
        grid = _build_grid(results, concept_names, reg_lambdas, d, truncate_fracs, centered)
        advantage = grid - naive_mean

        vabs = max(abs(advantage.min()), abs(advantage.max()), 0.01)
        sns.heatmap(advantage, annot=True, fmt="+.3f", cmap="PiYG", center=0, ax=ax,
                    xticklabels=trunc_labels,
                    yticklabels=reg_labels if ax_idx == 0 else False,
                    linewidths=0.5, linecolor='white',
                    vmin=-vabs, vmax=vabs)

        ax.set_xlabel('truncate_k')
        if ax_idx == 0:
            ax.set_ylabel('reg_lambda')
        ax.set_title(f'{"Centered" if centered else "Uncentered"}\nGreen = WCA wins', fontsize=11)

        # Mark current config
        curr_rl_idx = reg_lambdas.index(1e-3)
        ax.add_patch(plt.Rectangle((0, curr_rl_idx), 1, 1,
                                    fill=False, edgecolor='cyan', linewidth=3))

    fig.suptitle('WCA Advantage Over Naive Procrustes', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_advantage.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: ablation_advantage.png")


def plot_dashboard(results, concept_names, reg_lambdas, reg_labels, d,
                   truncate_fracs, naive_sims, naive_c_sims, best_config, output_dir):
    """Combined summary dashboard."""
    best_sims = results[best_config]
    naive_mean = np.mean(list(naive_sims.values()))
    best_mean = np.nanmean([best_sims[c] for c in concept_names])

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # ── Panel 1: Best heatmap (uncentered) ──
    ax1 = fig.add_subplot(gs[0, 0])
    grid = _build_grid(results, concept_names, reg_lambdas, d, truncate_fracs, False)
    trunc_labels = [f"{d}" if f == 1.0 else f"{int(d*f)}" for f in truncate_fracs]
    sns.heatmap(grid, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax1,
                xticklabels=trunc_labels, yticklabels=reg_labels,
                linewidths=0.5, linecolor='white', annot_kws={"size": 8})
    ax1.set_xlabel('truncate_k', fontsize=9)
    ax1.set_ylabel('reg_lambda', fontsize=9)
    ax1.set_title('Ablation Landscape (Uncentered)', fontsize=10)

    # ── Panel 2: Line chart ──
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(reg_lambdas))
    trunc_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for j, frac in enumerate(truncate_fracs):
        tk = d if frac == 1.0 else int(d * frac)
        label = f"k={tk}"
        y_vals = [np.nanmean([results.get((rl, tk, False), {}).get(c, float('nan'))
                              for c in concept_names]) for rl in reg_lambdas]
        ax2.plot(x, y_vals, 'o-', color=trunc_colors[j], label=label, linewidth=1.5, markersize=4)
    ax2.axhline(y=naive_mean, color='black', linestyle='--', alpha=0.7, label=f'Naive ({naive_mean:.3f})')
    ax2.set_xticks(x)
    ax2.set_xticklabels(reg_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean Cosine Sim', fontsize=9)
    ax2.set_title('WCA vs Regularization', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Top/Bottom concepts at best config ──
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_c = sorted(concept_names, key=lambda c: best_sims.get(c, 0), reverse=True)
    top5 = sorted_c[:5]
    bot5 = sorted_c[-5:]
    display = top5 + ['---'] + bot5
    vals = [best_sims.get(c, 0) for c in top5] + [0] + [best_sims.get(c, 0) for c in bot5]
    clrs = [GROUP_COLORS.get(get_group(c), '#95a5a6') for c in top5] + \
           ['white'] + [GROUP_COLORS.get(get_group(c), '#95a5a6') for c in bot5]
    yp = np.arange(len(display))
    ax3.barh(yp, vals, color=clrs, edgecolor='white', linewidth=0.5)
    ax3.set_yticks(yp)
    ax3.set_yticklabels([c.replace('[', '').replace(']', '') for c in display], fontsize=8)
    ax3.set_title('Top 5 / Bottom 5 (Best Config)', fontsize=10)
    ax3.invert_yaxis()
    ax3.axvline(x=0, color='black', linewidth=0.5)

    # ── Panel 4: Per-group comparison ──
    ax4 = fig.add_subplot(gs[1, 0])
    groups = list(CONCEPT_GROUPS.keys())
    wca_group_means = []
    naive_group_means = []
    for g in groups:
        members = [c for c in CONCEPT_GROUPS[g] if c in concept_names]
        if members:
            wca_group_means.append(np.nanmean([best_sims.get(c, 0) for c in members]))
            naive_group_means.append(np.mean([naive_sims[c] for c in members]))
        else:
            wca_group_means.append(0)
            naive_group_means.append(0)
    gx = np.arange(len(groups))
    w = 0.35
    ax4.bar(gx - w/2, wca_group_means, w, label='WCA (best)', color=[GROUP_COLORS[g] for g in groups], alpha=0.85)
    ax4.bar(gx + w/2, naive_group_means, w, label='Naive', color=[GROUP_COLORS[g] for g in groups], alpha=0.35, hatch='///')
    ax4.set_xticks(gx)
    ax4.set_xticklabels([g.replace(' ', '\n') for g in groups], fontsize=7)
    ax4.set_ylabel('Mean Cosine Sim', fontsize=9)
    ax4.set_title('By Concept Group', fontsize=10)
    ax4.legend(fontsize=7)
    ax4.axhline(y=0, color='black', linewidth=0.5)

    # ── Panel 5: Summary stats table ──
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    rl, tk, cen = best_config
    # Current config
    curr_config = (1e-3, d, False)
    curr_mean = np.nanmean([results.get(curr_config, {}).get(c, float('nan')) for c in concept_names])

    # Win counts
    wca_wins = sum(1 for c in concept_names if best_sims.get(c, 0) > naive_sims[c])
    naive_wins = len(concept_names) - wca_wins

    table_data = [
        ['Model', os.path.basename(os.path.dirname(output_dir))],
        ['Concepts', str(len(concept_names))],
        ['Best Config', f'λ={rl}, k={tk}, {"centered" if cen else "uncentered"}'],
        ['Best WCA Mean', f'{best_mean:.4f}'],
        ['Current Config (λ=1e-3)', f'{curr_mean:.4f}'],
        ['Naive Baseline', f'{naive_mean:.4f}'],
        ['Best WCA Advantage', f'{best_mean - naive_mean:+.4f}'],
        ['Current WCA Advantage', f'{curr_mean - naive_mean:+.4f}'],
        ['WCA wins / Naive wins', f'{wca_wins} / {naive_wins}'],
        ['Configs Evaluated', str(len(results))],
    ]

    table = ax5.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left', colWidths=[0.45, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif 'Advantage' in table_data[row-1][0]:
            val = float(table_data[row-1][1])
            cell.set_facecolor('#d5f5e3' if val > 0 else '#fadbd8')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')
    ax5.set_title('Ablation Summary', fontsize=11, pad=20)

    fig.suptitle('WCA Hyperparameter Ablation Dashboard', fontsize=15, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(output_dir, "ablation_dashboard.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: ablation_dashboard.png")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='WCA Hyperparameter Ablation Study')
    parser.add_argument('result_dir', help='Path to model results directory')
    parser.add_argument('--model', default=None, help='Model path (for covariance extraction if needed)')
    args = parser.parse_args()

    result_dir = args.result_dir
    ablation_dir = os.path.join(result_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  WCA Hyperparameter Ablation Study")
    print(f"  Input:  {result_dir}")
    print(f"  Output: {ablation_dir}")
    print(f"{'='*60}\n")

    # ── 1. Load inputs ──
    print("[1/5] Loading inputs...")
    cov_src, cov_tgt = load_covariances(result_dir, args.model)

    # Load activations (try new naming, then legacy)
    for src_name, tgt_name in [("X_src.pt", "X_tgt.pt"), ("X_en.pt", "X_fr.pt")]:
        src_path = os.path.join(result_dir, src_name)
        tgt_path = os.path.join(result_dir, tgt_name)
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            X_src = torch.load(src_path, map_location='cpu', weights_only=True).float()
            X_tgt = torch.load(tgt_path, map_location='cpu', weights_only=True).float()
            break
    else:
        print("  ERROR: No activation files found (X_src.pt/X_tgt.pt or X_en.pt/X_fr.pt)")
        sys.exit(1)

    raw_vectors = {}
    for f in glob.glob(os.path.join(result_dir, "v_raw_*.pt")):
        name = os.path.basename(f).replace("v_raw_", "").replace(".pt", "")
        raw_vectors[name] = torch.load(f, map_location='cpu', weights_only=True).float()

    concept_names = sort_concept_names(list(raw_vectors.keys()))
    d = cov_src.shape[0]
    print(f"  Hidden dim: {d}")
    print(f"  Concepts: {len(concept_names)}")
    print(f"  Paired activations: {X_src.shape[0]}")

    # ── 2. Compute naive baselines ──
    print("\n[2/5] Computing naive baselines...")
    naive_sims = compute_naive(X_src, X_tgt, raw_vectors, concept_names, centered=False)
    naive_c_sims = compute_naive(X_src, X_tgt, raw_vectors, concept_names, centered=True)
    naive_mean = np.mean(list(naive_sims.values()))
    naive_c_mean = np.mean(list(naive_c_sims.values()))
    print(f"  Naive (uncentered): {naive_mean:.4f}")
    print(f"  Naive (centered):   {naive_c_mean:.4f}")

    # ── 3. Sweep ──
    print(f"\n[3/5] Running sweep ({len(REG_LAMBDAS)} × {len(TRUNCATE_FRACS)} × {len(CENTERING_OPTIONS)} = {len(REG_LAMBDAS)*len(TRUNCATE_FRACS)*len(CENTERING_OPTIONS)} configs)...")
    results = sweep(cov_src, cov_tgt, X_src, X_tgt, raw_vectors, concept_names, d,
                    REG_LAMBDAS, TRUNCATE_FRACS, CENTERING_OPTIONS)

    # ── 4. Find best ──
    print("\n[4/5] Analyzing results...")
    best_config, best_mean = find_best(results, concept_names)
    rl, tk, cen = best_config

    # Current config for comparison
    curr_config = (1e-3, d, False)
    curr_mean = np.nanmean([results.get(curr_config, {}).get(c, float('nan')) for c in concept_names])

    print(f"\n  {'─'*50}")
    print(f"  BEST:    λ={rl}, k={tk}, {'centered' if cen else 'uncentered'} → {best_mean:.4f}")
    print(f"  CURRENT: λ=1e-3, k={d}, uncentered → {curr_mean:.4f}")
    print(f"  NAIVE:   uncentered → {naive_mean:.4f}")
    print(f"  ")
    print(f"  Best vs Naive:    {best_mean - naive_mean:+.4f}")
    print(f"  Current vs Naive: {curr_mean - naive_mean:+.4f}")
    print(f"  Best vs Current:  {best_mean - curr_mean:+.4f}")
    print(f"  {'─'*50}")

    # ── 5. Generate outputs ──
    print(f"\n[5/5] Generating outputs...")
    save_csv(results, concept_names, naive_sims, naive_c_sims, REG_LAMBDAS, ablation_dir)

    plot_heatmaps(results, concept_names, REG_LAMBDAS, REG_LABELS, d,
                  TRUNCATE_FRACS, naive_mean, naive_c_mean, ablation_dir)
    plot_lines(results, concept_names, REG_LAMBDAS, REG_LABELS, d,
               TRUNCATE_FRACS, naive_mean, naive_c_mean, ablation_dir)
    plot_best_vs_naive(results[best_config], naive_sims, concept_names, best_config, ablation_dir)
    plot_advantage_heatmap(results, concept_names, REG_LAMBDAS, REG_LABELS, d,
                           TRUNCATE_FRACS, naive_sims, ablation_dir)
    plot_dashboard(results, concept_names, REG_LAMBDAS, REG_LABELS, d,
                   TRUNCATE_FRACS, naive_sims, naive_c_sims, best_config, ablation_dir)

    print(f"\n{'='*60}")
    print(f"  Ablation complete! All outputs in: {ablation_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
