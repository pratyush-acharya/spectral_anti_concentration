#!/usr/bin/env python3
"""
Experiment 3: Concept-Level Utility Correlation
================================================

Tests whether a concept's spectral position (Tail-Heaviness Index)
predicts how much it benefits from WCA transport.

Depends on:
  - Experiment 1 output: results/_spectral/cdfs/<model>_en-fr.json
  - RAID results: results/<model>/<pair>/ablation/ablation_results.csv OR stats.txt

Usage:
    uv run python scripts/utility_correlation.py
"""

import argparse
import json
import os
import re
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

RESULTS_DIR = Path("results")
SPECTRAL_DIR = Path("results/_spectral")
OUTPUT_DIR = Path("results/_spectral/utility")

# Same concept groups as spectral_analysis.py
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

CONCEPT_TO_GROUP = {}
for group, concepts in CONCEPT_GROUPS.items():
    for c in concepts:
        CONCEPT_TO_GROUP[c] = group

GROUP_COLORS = {
    "Verb Morphology": "#E63946",
    "Semantic": "#457B9D",
    "Grammatical": "#2A9D8F",
    "Language Pairs": "#E9C46A",
}

ALL_MODELS = [
    "gemma-2-2b", "gemma-2-9b", "jetmoe-8b",
    "Llama-3.2-1B", "Llama-3.2-3B", "Mistral-7B-v0.3",
    "OLMoE-1B-7B-0125", "Qwen2.5-0.5B", "Qwen2.5-1.5B",
    "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B",
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
    "SmolLM2-1.7B",
]

LANG_PAIRS = ["en-fr", "es-de", "fr-de", "fr-es"]


def compute_tail_heaviness(cdf_data: dict, threshold: float = 0.5) -> dict:
    """
    Compute Tail-Heaviness Index (THI) for each concept from Exp 1 CDF data.
    THI = fraction of concept energy below the V(k)=threshold cumulative variance.
    High THI = concept lives deep in the spectral tail.
    
    Returns: {concept_name: THI}
    """
    cumvar = np.array(cdf_data["cumvar"])
    # Find index where cumvar >= threshold
    k_star = np.searchsorted(cumvar, threshold)
    if k_star >= len(cumvar):
        k_star = len(cumvar) - 1
    
    thi_by_concept = {}
    
    for key, concept_cdfs in cdf_data.items():
        if not key.startswith("group_"):
            continue
        group_name = key.replace("group_", "")
        
        # Get the concept names for this group
        if group_name in CONCEPT_GROUPS:
            concepts = CONCEPT_GROUPS[group_name]
        else:
            continue
        
        for i, cdf in enumerate(concept_cdfs):
            if i < len(concepts):
                concept_name = concepts[i]
                cdf_arr = np.array(cdf)
                # THI = 1 - C(k*), where C(k*) is energy captured by top variance directions
                thi = 1.0 - cdf_arr[k_star]
                thi_by_concept[concept_name] = thi
    
    return thi_by_concept


def load_raid_deltas_from_stats(model: str, pair: str) -> dict:
    """
    Parse stats.txt for Δ(WCA−Naive) per concept.
    Returns {concept_name: delta}.
    """
    stats_path = RESULTS_DIR / model / pair / "stats.txt"
    if not stats_path.exists():
        return {}
    
    deltas = {}
    current_concept = None
    sim_wca = None
    sim_naive = None
    
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Concept: (.+)", line)
            if m:
                current_concept = m.group(1)
                sim_wca = None
                sim_naive = None
                continue
            
            m = re.match(r"Cosine Sim \(Raw vs WCA\): (.+)", line)
            if m and current_concept:
                try:
                    sim_wca = float(m.group(1))
                except ValueError:
                    sim_wca = None
                continue
            
            m = re.match(r"Cosine Sim \(Raw vs Naive\): (.+)", line)
            if m and current_concept:
                try:
                    sim_naive = float(m.group(1))
                except ValueError:
                    sim_naive = None
                
                if sim_wca is not None and sim_naive is not None:
                    if not (np.isnan(sim_wca) or np.isnan(sim_naive)):
                        deltas[current_concept] = sim_wca - sim_naive
                current_concept = None
    
    return deltas


def load_raid_deltas_from_ablation(model: str, pair: str, target_lambda: float = 0.1) -> dict:
    """
    Load Δ(WCA−Naive) from ablation CSV at a specific lambda.
    Falls back to stats.txt (default lambda=0.001).
    """
    ablation_path = RESULTS_DIR / model / pair / "ablation" / "ablation_results.csv"
    if not ablation_path.exists():
        return load_raid_deltas_from_stats(model, pair)
    
    naive_row = None
    wca_row = None
    
    with open(ablation_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers:
            return load_raid_deltas_from_stats(model, pair)
        
        concept_cols = [h for h in headers if h not in ("reg_lambda", "truncate_k", "centered", "mean_sim")]
        
        for row in reader:
            lam = row.get("reg_lambda", "")
            trunc = row.get("truncate_k", "")
            centered = row.get("centered", "")
            
            if lam == "naive" and centered == "False":
                naive_row = row
            elif centered == "False" and trunc == "N/A":
                try:
                    if abs(float(lam) - target_lambda) < 1e-6:
                        wca_row = row
                except ValueError:
                    pass
    
    if naive_row is None or wca_row is None:
        return load_raid_deltas_from_stats(model, pair)
    
    deltas = {}
    for concept in concept_cols:
        try:
            wca_val = float(wca_row[concept])
            naive_val = float(naive_row[concept])
            if not (np.isnan(wca_val) or np.isnan(naive_val)):
                deltas[concept] = wca_val - naive_val
        except (ValueError, KeyError):
            pass
    
    return deltas


def main():
    parser = argparse.ArgumentParser(description="Concept-Level Utility Correlation (Exp 3)")
    parser.add_argument("--lambda-val", type=float, default=0.1,
                       help="Lambda value for WCA deltas (default: 0.1)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Cumulative variance threshold for THI (default: 0.5)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  EXPERIMENT 3: Concept-Level Utility Correlation")
    print(f"  Lambda: {args.lambda_val}, THI threshold: {args.threshold}")
    print("=" * 60)
    
    # ── Step 1: Load THI from Exp 1 CDFs ──
    print("\n[1] Loading spectral CDFs from Experiment 1...")
    
    thi_by_model_concept = defaultdict(dict)  # model -> {concept: THI}
    
    for model in ALL_MODELS:
        cdf_path = SPECTRAL_DIR / "cdfs" / f"{model}_en-fr.json"
        if not cdf_path.exists():
            print(f"  SKIP {model}: no CDF data (run spectral_analysis.py first)")
            continue
        
        with open(cdf_path) as f:
            cdf_data = json.load(f)
        
        thi = compute_tail_heaviness(cdf_data, args.threshold)
        thi_by_model_concept[model] = thi
        print(f"  {model}: {len(thi)} concepts with THI")
    
    if not thi_by_model_concept:
        print("\n  ERROR: No CDF data found. Run spectral_analysis.py --experiment exp1 first.")
        return
    
    # ── Step 2: Load RAID deltas ──
    print("\n[2] Loading RAID deltas...")
    
    delta_by_model_pair_concept = defaultdict(lambda: defaultdict(dict))
    
    for model in thi_by_model_concept:
        for pair in LANG_PAIRS:
            deltas = load_raid_deltas_from_ablation(model, pair, args.lambda_val)
            if deltas:
                delta_by_model_pair_concept[model][pair] = deltas
                print(f"  {model}/{pair}: {len(deltas)} concepts with deltas")
    
    # ── Step 3: Compute correlations ──
    print("\n[3] Computing correlations...")
    
    # 3a: Concept-level (average across models and pairs)
    concept_thi_avg = defaultdict(list)
    concept_delta_avg = defaultdict(list)
    
    for model in thi_by_model_concept:
        thi = thi_by_model_concept[model]
        for pair in LANG_PAIRS:
            deltas = delta_by_model_pair_concept.get(model, {}).get(pair, {})
            for concept in thi:
                if concept in deltas:
                    concept_thi_avg[concept].append(thi[concept])
                    concept_delta_avg[concept].append(deltas[concept])
    
    # Average per concept
    concept_names = sorted([c for c in concept_thi_avg if len(concept_thi_avg[c]) >= 3])
    thi_means = [np.mean(concept_thi_avg[c]) for c in concept_names]
    delta_means = [np.mean(concept_delta_avg[c]) for c in concept_names]
    
    if len(concept_names) < 5:
        print("  ERROR: Insufficient data for correlation. Need at least 5 concepts.")
        return
    
    r_concept, p_concept = sp_stats.pearsonr(thi_means, delta_means)
    rho_concept, p_rho = sp_stats.spearmanr(thi_means, delta_means)
    
    print(f"\n  CONCEPT-LEVEL CORRELATION (N={len(concept_names)}):")
    print(f"    Pearson r  = {r_concept:+.4f}, p = {p_concept:.4e}")
    print(f"    Spearman ρ = {rho_concept:+.4f}, p = {p_rho:.4e}")
    
    # 3b: Instance-level (all model×concept observations)
    all_thi = []
    all_delta = []
    all_groups = []
    
    for model in thi_by_model_concept:
        thi = thi_by_model_concept[model]
        for pair in LANG_PAIRS:
            deltas = delta_by_model_pair_concept.get(model, {}).get(pair, {})
            for concept in thi:
                if concept in deltas:
                    all_thi.append(thi[concept])
                    all_delta.append(deltas[concept])
                    all_groups.append(CONCEPT_TO_GROUP.get(concept, "Unknown"))
    
    if len(all_thi) > 10:
        r_instance, p_instance = sp_stats.pearsonr(all_thi, all_delta)
        print(f"\n  INSTANCE-LEVEL CORRELATION (N={len(all_thi)}):")
        print(f"    Pearson r  = {r_instance:+.4f}, p = {p_instance:.4e}")
        print(f"    (Note: instances are not independent — cluster-robust SEs needed)")
    
    # ── Step 4: Plots ──
    print("\n[4] Generating plots...")
    
    # Plot 1: Concept-level scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    
    for i, cname in enumerate(concept_names):
        group = CONCEPT_TO_GROUP.get(cname, "Unknown")
        color = GROUP_COLORS.get(group, "grey")
        ax.scatter(thi_means[i], delta_means[i], c=color, s=60, zorder=3, edgecolors="white", linewidths=0.5)
        
        # Label a subset of interesting points
        if abs(delta_means[i]) > np.std(delta_means) or abs(thi_means[i] - np.mean(thi_means)) > np.std(thi_means):
            short_name = cname.replace("[", "").replace("]", "").strip()
            ax.annotate(short_name, (thi_means[i], delta_means[i]),
                       fontsize=6, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    
    # Regression line
    if len(thi_means) > 2:
        z = np.polyfit(thi_means, delta_means, 1)
        x_line = np.linspace(min(thi_means), max(thi_means), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", lw=1, alpha=0.5)
    
    # Legend
    for group, color in GROUP_COLORS.items():
        ax.scatter([], [], c=color, label=group, s=40)
    ax.legend(fontsize=8, loc="best")
    
    ax.axhline(0, color="grey", ls=":", alpha=0.3)
    ax.set_xlabel("Tail-Heaviness Index (THI)", fontsize=12)
    ax.set_ylabel("Δ(WCA − Naive) at λ=" + str(args.lambda_val), fontsize=12)
    ax.set_title(f"Spectral Position vs. WCA Benefit\n"
                 f"r={r_concept:+.3f} (p={p_concept:.3e}), "
                 f"ρ={rho_concept:+.3f} (p={p_rho:.3e})", fontsize=11)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "thi_vs_delta.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # Plot 2: Per-group THI distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    group_thist = defaultdict(list)
    for cname, thi_val in zip(concept_names, thi_means):
        group = CONCEPT_TO_GROUP.get(cname, "Unknown")
        group_thist[group].append(thi_val)
    
    positions = []
    labels = []
    for i, (group, vals) in enumerate(sorted(group_thist.items())):
        color = GROUP_COLORS.get(group, "grey")
        bp = ax.boxplot([vals], positions=[i], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color="black"))
        positions.append(i)
        labels.append(group)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Tail-Heaviness Index", fontsize=11)
    ax.set_title("THI Distribution by Concept Group", fontsize=12)
    ax.grid(True, alpha=0.2, axis="y")
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "thi_by_group.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")
    
    # ── Step 5: Save results ──
    results = {
        "lambda": args.lambda_val,
        "thi_threshold": args.threshold,
        "concept_level": {
            "n": len(concept_names),
            "pearson_r": float(r_concept),
            "pearson_p": float(p_concept),
            "spearman_rho": float(rho_concept),
            "spearman_p": float(p_rho),
            "concepts": {
                c: {"thi": float(t), "delta": float(d), "group": CONCEPT_TO_GROUP.get(c, "Unknown")}
                for c, t, d in zip(concept_names, thi_means, delta_means)
            },
        },
    }
    
    if len(all_thi) > 10:
        results["instance_level"] = {
            "n": len(all_thi),
            "pearson_r": float(r_instance),
            "pearson_p": float(p_instance),
        }
    
    results_path = OUTPUT_DIR / "utility_correlation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Concept-level: r={r_concept:+.4f} (p={p_concept:.3e}) — "
          f"{'SIGNIFICANT' if p_concept < 0.05 else 'not significant'}")
    if len(all_thi) > 10:
        print(f"  Instance-level: r={r_instance:+.4f} (p={p_instance:.3e}) — "
              f"{'SIGNIFICANT' if p_instance < 0.05 else 'not significant'}")
    
    if p_concept < 0.05 and r_concept > 0:
        print("\n  ✅ POSITIVE: Tail-heavy concepts benefit more from WCA.")
        print("     This supports the spectral mechanism hypothesis.")
    elif p_concept < 0.05 and r_concept < 0:
        print("\n  ⚠️  NEGATIVE CORRELATION: Tail-heavy concepts benefit LESS from WCA.")
        print("     This contradicts the spectral mechanism hypothesis.")
    else:
        print("\n  ❌ NO SIGNIFICANT CORRELATION found.")
        print("     THI does not predict WCA benefit.")


if __name__ == "__main__":
    main()
