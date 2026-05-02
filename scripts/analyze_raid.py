"""
Comprehensive post-hoc analysis of RAID results.
Generates 6 visualizations from saved .pt files (no model needed).

Usage: python scripts/analyze_raid.py results/Qwen3.5-0.8B
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
import os
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

sns.set_theme(context="paper", style="white", palette="colorblind", font="sans-serif", font_scale=1.5)

# ──────────────────────────────────────────────────────────
# Concept grouping (shared with perform_raid.py)
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


def get_group(concept_name):
    for group, members in CONCEPT_GROUPS.items():
        if concept_name in members:
            return group
    return 'Other'


def sort_concept_names(names):
    sorted_names = []
    for group_concepts in CONCEPT_GROUPS.values():
        for c in group_concepts:
            if c in names:
                sorted_names.append(c)
    for c in names:
        if c not in sorted_names:
            sorted_names.append(c)
    return sorted_names


def load_vectors(result_dir):
    """Load all saved vectors from a result directory."""
    raw_files = glob.glob(os.path.join(result_dir, "v_raw_*.pt"))

    data = {'raw': {}, 'raid': {}, 'naive': {}}
    for raw_file in raw_files:
        name = os.path.basename(raw_file).replace("v_raw_", "").replace(".pt", "")
        data['raw'][name] = torch.load(raw_file, map_location='cpu', weights_only=True).to(torch.float32)

        raid_file = os.path.join(result_dir, f"v_raid_{name}.pt")
        if os.path.exists(raid_file):
            data['raid'][name] = torch.load(raid_file, map_location='cpu', weights_only=True).to(torch.float32)

        naive_file = os.path.join(result_dir, f"v_naive_{name}.pt")
        if os.path.exists(naive_file):
            data['naive'][name] = torch.load(naive_file, map_location='cpu', weights_only=True).to(torch.float32)

    return data


def cosine_sim(a, b):
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b)).clamp(min=1e-10)).item()


# ──────────────────────────────────────────────────────────
# Visualization 1: Cosine Similarity Bar Chart
# ──────────────────────────────────────────────────────────
def plot_cosine_bars(data, concept_names, save_path):
    """Bar chart comparing WCA vs Naive cosine similarity per concept."""
    has_naive = len(data['naive']) > 0

    n = len(concept_names)
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.35)))

    y = np.arange(n)
    bar_height = 0.35 if has_naive else 0.6

    wca_sims = [cosine_sim(data['raw'][c], data['raid'][c]) for c in concept_names]
    colors = [GROUP_COLORS.get(get_group(c), '#95a5a6') for c in concept_names]

    ax.barh(y, wca_sims, bar_height, label='WCA (Whitened)', color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    if has_naive:
        naive_sims = [cosine_sim(data['raw'][c], data['naive'][c]) for c in concept_names]
        ax.barh(y + bar_height, naive_sims, bar_height, label='Naive Procrustes', color=colors, alpha=0.35,
                edgecolor='white', linewidth=0.5, hatch='///')

    ax.set_yticks(y + (bar_height / 2 if has_naive else 0))
    ax.set_yticklabels(concept_names, fontsize=9)
    ax.set_xlabel('Cosine Similarity (Raw vs Transported)')
    ax.set_title('Concept Alignment: WCA vs Naive Procrustes')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    # Add group labels
    for group, color in GROUP_COLORS.items():
        ax.plot([], [], 's', color=color, label=group, markersize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# Visualization 2: WCA vs Naive Scatter
# ──────────────────────────────────────────────────────────
def plot_wca_vs_naive_scatter(data, concept_names, save_path):
    """Scatter: x=naive_sim, y=wca_sim. Above diagonal = WCA wins."""
    if not data['naive']:
        print("  Skipping scatter: no naive vectors found.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for c in concept_names:
        naive_s = cosine_sim(data['raw'][c], data['naive'][c])
        wca_s = cosine_sim(data['raw'][c], data['raid'][c])
        group = get_group(c)
        color = GROUP_COLORS.get(group, '#95a5a6')
        ax.scatter(naive_s, wca_s, c=color, s=60, zorder=3, edgecolors='white', linewidth=0.5)
        ax.annotate(c.replace('[', '').replace(']', ''),
                    (naive_s, wca_s), fontsize=6, alpha=0.7,
                    textcoords="offset points", xytext=(4, 4))

    lims = [-0.3, 0.5]
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1, label='y = x (equal)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Naive Procrustes Cosine Sim')
    ax.set_ylabel('WCA Cosine Sim')
    ax.set_title('WCA vs Naive: Points Above Line → WCA Wins')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Color legend
    for group, color in GROUP_COLORS.items():
        ax.plot([], [], 'o', color=color, label=group, markersize=6)
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# Visualization 3: Norm Preservation Chart
# ──────────────────────────────────────────────────────────
def plot_norm_chart(data, concept_names, save_path):
    """Bar chart of transported vector norms relative to raw norms."""
    has_naive = len(data['naive']) > 0

    n = len(concept_names)
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.35)))

    y = np.arange(n)
    bar_height = 0.35 if has_naive else 0.6

    wca_ratios = [torch.norm(data['raid'][c]).item() / max(torch.norm(data['raw'][c]).item(), 1e-10) for c in concept_names]
    colors = [GROUP_COLORS.get(get_group(c), '#95a5a6') for c in concept_names]

    ax.barh(y, wca_ratios, bar_height, label='WCA', color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    if has_naive:
        naive_ratios = [torch.norm(data['naive'][c]).item() / max(torch.norm(data['raw'][c]).item(), 1e-10) for c in concept_names]
        ax.barh(y + bar_height, naive_ratios, bar_height, label='Naive', color=colors, alpha=0.35,
                edgecolor='white', linewidth=0.5, hatch='///')

    ax.set_yticks(y + (bar_height / 2 if has_naive else 0))
    ax.set_yticklabels(concept_names, fontsize=9)
    ax.set_xlabel('||v_transported|| / ||v_raw||')
    ax.set_title('Norm Preservation After Transport')
    ax.axvline(x=1.0, color='red', linewidth=1, linestyle='--', alpha=0.7, label='Perfect preservation')
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# Visualization 4: Language Leakage
# ──────────────────────────────────────────────────────────
def plot_language_leakage(data, concept_names, save_path):
    """Project each transported vector onto the [English-French] direction."""
    lang_key = '[English - French]'
    if lang_key not in data['raw']:
        print(f"  Skipping language leakage: {lang_key} not found in concepts.")
        return

    v_lang = data['raw'][lang_key]
    v_lang_norm = v_lang / torch.norm(v_lang)

    # Exclude the language concept itself
    filtered = [c for c in concept_names if c != lang_key]

    n = len(filtered)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.35)))

    y = np.arange(n)
    has_naive = len(data['naive']) > 0
    bar_height = 0.35 if has_naive else 0.6

    wca_proj = [torch.dot(data['raid'][c], v_lang_norm).item() for c in filtered]
    colors = [GROUP_COLORS.get(get_group(c), '#95a5a6') for c in filtered]

    ax.barh(y, wca_proj, bar_height, label='WCA (Raid)', color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    if has_naive:
        naive_proj = [torch.dot(data['naive'][c], v_lang_norm).item() for c in filtered]
        ax.barh(y + bar_height, naive_proj, bar_height, label='Naive', color=colors, alpha=0.35,
                edgecolor='white', linewidth=0.5, hatch='///')

    # Also show raw projections as reference markers
    raw_proj = [torch.dot(data['raw'][c], v_lang_norm).item() for c in filtered]
    ax.scatter(raw_proj, y + bar_height / 2, marker='|', color='black', s=100, zorder=5, label='Raw (reference)')

    ax.set_yticks(y + (bar_height / 2 if has_naive else 0))
    ax.set_yticklabels(filtered, fontsize=9)
    ax.set_xlabel('Projection onto [English - French] direction')
    ax.set_title('Language Leakage: How Much "English→French" Bleeds Into Concepts')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.legend(loc='lower right', fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# Visualization 5: Delta Similarity Matrix
# ──────────────────────────────────────────────────────────
def plot_delta_sim(data, concept_names, save_path):
    """Heatmap of (sim_raid − sim_raw) showing structure distortion."""
    raw_vecs = torch.stack([data['raw'][c] for c in concept_names])
    raid_vecs = torch.stack([data['raid'][c] for c in concept_names])

    raw_norm = torch.nn.functional.normalize(raw_vecs, p=2, dim=1)
    raid_norm = torch.nn.functional.normalize(raid_vecs, p=2, dim=1)

    sim_raw = torch.mm(raw_norm, raw_norm.T).detach().numpy()
    sim_raid = torch.mm(raid_norm, raid_norm.T).detach().numpy()
    delta = sim_raid - sim_raw

    n = len(concept_names)
    fig_size = max(12, n * 0.55)
    annot_size = max(6, 12 - n // 5)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    sns.heatmap(delta, annot=True, fmt=".2f", cmap="PiYG", center=0,
                xticklabels=concept_names, yticklabels=concept_names,
                annot_kws={"size": annot_size}, ax=ax,
                linewidths=0.5, linecolor='white', vmin=-0.5, vmax=0.5)
    ax.set_title('Δ Similarity (Raid − Raw): Green = More Similar After Transport', fontsize=13, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# Visualization 6: Summary Dashboard
# ──────────────────────────────────────────────────────────
def plot_dashboard(data, concept_names, result_dir, save_path):
    """Combined summary figure with key metrics."""
    has_naive = len(data['naive']) > 0

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # ── Panel 1: Group-level mean cosine similarity ──
    ax1 = fig.add_subplot(gs[0, 0])
    groups = list(CONCEPT_GROUPS.keys())
    wca_means = []
    naive_means = []
    for group in groups:
        members = [c for c in CONCEPT_GROUPS[group] if c in data['raw'] and c in data['raid']]
        if members:
            wca_means.append(np.mean([cosine_sim(data['raw'][c], data['raid'][c]) for c in members]))
            if has_naive:
                naive_means.append(np.mean([cosine_sim(data['raw'][c], data['naive'][c]) for c in members]))
            else:
                naive_means.append(0)
        else:
            wca_means.append(0)
            naive_means.append(0)

    x = np.arange(len(groups))
    w = 0.35
    ax1.bar(x - w/2, wca_means, w, label='WCA', color=[GROUP_COLORS[g] for g in groups], alpha=0.85, edgecolor='white')
    if has_naive:
        ax1.bar(x + w/2, naive_means, w, label='Naive', color=[GROUP_COLORS[g] for g in groups], alpha=0.35, hatch='///', edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels([g.replace(' ', '\n') for g in groups], fontsize=8)
    ax1.set_ylabel('Mean Cosine Sim')
    ax1.set_title('Alignment by Concept Group', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.axhline(y=0, color='black', linewidth=0.5)

    # ── Panel 2: Distribution of cosine similarities ──
    ax2 = fig.add_subplot(gs[0, 1])
    wca_all = [cosine_sim(data['raw'][c], data['raid'][c]) for c in concept_names]
    ax2.hist(wca_all, bins=15, alpha=0.7, label='WCA', color='#e74c3c', edgecolor='white')
    if has_naive:
        naive_all = [cosine_sim(data['raw'][c], data['naive'][c]) for c in concept_names]
        ax2.hist(naive_all, bins=15, alpha=0.4, label='Naive', color='#3498db', edgecolor='white', hatch='///')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Alignment Scores', fontsize=11)
    ax2.legend(fontsize=8)

    # ── Panel 3: Top & Bottom concepts ──
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_concepts = sorted(concept_names, key=lambda c: cosine_sim(data['raw'][c], data['raid'][c]), reverse=True)
    top5 = sorted_concepts[:5]
    bot5 = sorted_concepts[-5:]
    display = top5 + ['---'] + bot5
    values = [cosine_sim(data['raw'][c], data['raid'][c]) for c in top5] + [0] + [cosine_sim(data['raw'][c], data['raid'][c]) for c in bot5]
    colors = [GROUP_COLORS.get(get_group(c), '#95a5a6') for c in top5] + ['white'] + [GROUP_COLORS.get(get_group(c), '#95a5a6') for c in bot5]

    y_pos = np.arange(len(display))
    ax3.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([c.replace('[', '').replace(']', '') for c in display], fontsize=8)
    ax3.set_xlabel('Cosine Sim (WCA)')
    ax3.set_title('Top 5 & Bottom 5 Concepts', fontsize=11)
    ax3.invert_yaxis()
    ax3.axvline(x=0, color='black', linewidth=0.5)

    # ── Panel 4: Norm ratios ──
    ax4 = fig.add_subplot(gs[1, 0])
    wca_norms = [torch.norm(data['raid'][c]).item() / max(torch.norm(data['raw'][c]).item(), 1e-10) for c in concept_names]
    ax4.scatter(range(len(concept_names)), wca_norms, c=[GROUP_COLORS.get(get_group(c), '#95a5a6') for c in concept_names],
                s=40, zorder=3, edgecolors='white', linewidth=0.5)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_ylabel('||v_raid|| / ||v_raw||')
    ax4.set_title('Norm Preservation', fontsize=11)
    ax4.set_xticks([])
    ax4.set_xlabel(f'{len(concept_names)} concepts')

    # ── Panel 5: Summary statistics table ──
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    model_name = os.path.basename(result_dir)
    wca_mean = np.mean(wca_all)
    wca_std = np.std(wca_all)
    wca_max_c = sorted_concepts[0]
    wca_min_c = sorted_concepts[-1]

    table_data = [
        ['Model', model_name],
        ['Concepts', str(len(concept_names))],
        ['Mean WCA Cosine Sim', f'{wca_mean:.4f} ± {wca_std:.4f}'],
        ['Best Concept (WCA)', f'{wca_max_c}: {cosine_sim(data["raw"][wca_max_c], data["raid"][wca_max_c]):.4f}'],
        ['Worst Concept (WCA)', f'{wca_min_c}: {cosine_sim(data["raw"][wca_min_c], data["raid"][wca_min_c]):.4f}'],
    ]
    if has_naive:
        naive_mean = np.mean(naive_all)
        naive_std = np.std(naive_all)
        table_data.append(['Mean Naive Cosine Sim', f'{naive_mean:.4f} ± {naive_std:.4f}'])
        table_data.append(['WCA Advantage', f'{wca_mean - naive_mean:+.4f}'])

    table = ax5.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left', colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#ecf0f1' if row % 2 == 0 else 'white')
    ax5.set_title('Summary Statistics', fontsize=11, pad=20)

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def analyze(result_dir):
    print(f"=== Analyzing: {result_dir} ===")

    data = load_vectors(result_dir)
    concept_names = sort_concept_names(list(data['raw'].keys()))

    if not concept_names:
        print("No vectors found!")
        return

    has_naive = len(data['naive']) > 0
    print(f"Found {len(concept_names)} concepts, naive baseline: {'Yes' if has_naive else 'No'}")

    # Generate all visualizations
    print("Generating visualizations...")
    plot_cosine_bars(data, concept_names, os.path.join(result_dir, "cosine_bars.png"))
    plot_wca_vs_naive_scatter(data, concept_names, os.path.join(result_dir, "wca_vs_naive.png"))
    plot_norm_chart(data, concept_names, os.path.join(result_dir, "norm_chart.png"))
    plot_language_leakage(data, concept_names, os.path.join(result_dir, "language_leakage.png"))
    plot_delta_sim(data, concept_names, os.path.join(result_dir, "delta_sim.png"))
    plot_dashboard(data, concept_names, result_dir, os.path.join(result_dir, "dashboard.png"))

    # Also regenerate sorted similarity matrices
    raw_vectors = [data['raw'][n] for n in concept_names]
    raid_vectors = [data['raid'][n] for n in concept_names]
    lrg.plot_similarity_matrix(raw_vectors, concept_names, "Concept Similarity Matrix (English/Raw)",
                               os.path.join(result_dir, "sim_matrix_raw.png"))
    lrg.plot_similarity_matrix(raid_vectors, concept_names, "Concept Similarity Matrix (French/Raid)",
                               os.path.join(result_dir, "sim_matrix_raid.png"))

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_raid.py <result_dir>")
        sys.exit(1)
    analyze(sys.argv[1])
