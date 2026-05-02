#!/usr/bin/env python3
"""
M1: POS-Tag Probing Experiment — "Does syntax shout?"
======================================================

For each model, we:
  1. Load a POS-tagged corpus (Universal Dependencies English-EWT).
  2. Extract residual-stream activations at the last hidden layer.
  3. Eigendecompose the saved covariance matrix Σ of the unembedding.
  4. Project activations onto:
       (a) top-k eigenvectors of Σ   (high-variance "shouting" subspace)
       (b) bottom-k eigenvectors     (low-variance "whispering" subspace)
       (c) random-k eigenvectors     (baseline)
  5. Train a logistic-regression POS classifier on each projection.
  6. Compare accuracy: if top-k >> bottom-k, then syntactic information
     concentrates in high-variance directions → "syntax shouts".

This validates the empirical claim that high-variance eigenvectors carry
more syntactic information than low-variance ones.

Usage:
    uv run python scripts/pos_probe_experiment.py --model google/gemma-2-2b
    uv run python scripts/pos_probe_experiment.py --model Qwen/Qwen2.5-3B --k-frac 0.1
    uv run python scripts/pos_probe_experiment.py --models google/gemma-2-2b Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B
"""

import argparse
import json
import os
import sys
import gc
import re
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Use PyTorch GPU logistic regression when CUDA is available (no cuML needed)
TORCH_GPU = torch.cuda.is_available()

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

RESULTS_DIR = Path("results")
SPECTRAL_DIR = Path("results/_spectral")
OUTPUT_DIR = Path("results/_spectral/pos_probe")

# Universal POS tags (UPOS) used in Universal Dependencies
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]

# Minimum samples per POS tag to include in classification
MIN_SAMPLES_PER_TAG = 30

# Number of sentences to use from UD corpus (None = all)
MAX_SENTENCES = 500


# ─── CoNLL-U Parser ──────────────────────────────────────────────────────────

def parse_conllu(path: str) -> list:
    """
    Parse a CoNLL-U file into a list of sentences.
    Each sentence is a list of (word, upos) tuples.
    """
    sentences = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#"):
                continue
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            tok_id = fields[0]
            # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
            if "-" in tok_id or "." in tok_id:
                continue
            word = fields[1]
            upos = fields[3]
            current.append((word, upos))
    if current:
        sentences.append(current)
    return sentences


def download_ud_english(data_dir: str = "data/ud") -> str:
    """
    Download UD English-EWT if not already present.
    Returns path to the dev set CoNLL-U file.
    """
    ud_dir = Path(data_dir)
    conllu_file = ud_dir / "en_ewt-ud-dev.conllu"

    if conllu_file.exists():
        return str(conllu_file)

    ud_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading UD English-EWT dev set...")

    import urllib.request
    url = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu"
    urllib.request.urlretrieve(url, conllu_file)
    print(f"  Saved to {conllu_file}")
    return str(conllu_file)


def download_ud_chinese(data_dir: str = "data/ud") -> str:
    """
    Download UD Chinese-GSD if not already present.
    Returns path to the dev set CoNLL-U file.
    """
    ud_dir = Path(data_dir)
    conllu_file = ud_dir / "zh_gsd-ud-dev.conllu"

    if conllu_file.exists():
        return str(conllu_file)

    ud_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading UD Chinese-GSD dev set...")

    import urllib.request
    url = "https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master/zh_gsd-ud-dev.conllu"
    urllib.request.urlretrieve(url, conllu_file)
    print(f"  Saved to {conllu_file}")
    return str(conllu_file)


# ─── Activation Extraction ───────────────────────────────────────────────────

def extract_token_activations(model, tokenizer, handler, sentences, device,
                               max_sentences=MAX_SENTENCES, max_length=256):
    """
    Extract residual-stream activations for each POS-tagged token.

    Returns:
        activations: (N, d_model) tensor of token activations
        pos_labels:  list of N POS-tag strings
    """
    all_activations = []
    all_labels = []

    if max_sentences and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    for sent in tqdm(sentences, desc="  Extracting activations"):
        words = [w for w, _ in sent]
        tags = [t for _, t in sent]
        text = " ".join(words)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()  # (seq_len, 2)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        # Last hidden layer, apply final norm
        h = outputs.hidden_states[-1]
        h = handler.apply_final_norm(model, h)
        h = h[0].cpu().float()  # (seq_len, d_model)

        # Map each word in the sentence to token indices via character offsets
        # Build word -> char_start, char_end
        word_char_spans = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            word_char_spans.append((start, end))
            pos = end

        for word_idx, (w_start, w_end) in enumerate(word_char_spans):
            tag = tags[word_idx]
            # Skip punctuation and symbols for cleaner signal
            if tag in ("PUNCT", "SYM", "X"):
                continue

            # Find token indices that overlap this word's character span
            token_indices = []
            for tok_idx, (ts, te) in enumerate(offset_mapping):
                if ts == te == 0:
                    continue  # special token
                if ts < w_end and te > w_start:
                    token_indices.append(tok_idx)

            if not token_indices:
                continue

            # Mean-pool over the word's subword tokens
            word_act = h[token_indices].mean(dim=0)
            all_activations.append(word_act)
            all_labels.append(tag)

    if not all_activations:
        return torch.zeros(0), []

    return torch.stack(all_activations), all_labels


# ─── Projection & Classification ─────────────────────────────────────────────

def project_activations(activations: torch.Tensor, eigenvectors: torch.Tensor,
                         indices: list) -> np.ndarray:
    """
    Project activations onto a subset of eigenvectors.
    activations: (N, d)
    eigenvectors: (d, d) — columns are eigenvectors, sorted descending by eigenvalue
    indices: list of eigenvector column indices to use
    Returns: (N, len(indices)) numpy array
    """
    U_sub = eigenvectors[:, indices].detach().float()  # (d, k)
    proj = (activations.detach().float() @ U_sub).numpy()  # (N, k)
    return proj


def _torch_logistic_regression_gpu(X_train, y_train, X_test, n_classes,
                                     max_iter=2000, C=1.0, lr=1.0):
    """
    GPU-accelerated logistic regression via PyTorch.
    Uses LBFGS (same algorithm as sklearn's default) on CUDA.
    """
    device = torch.device("cuda")
    n_features = X_train.shape[1]

    # Move data to GPU as float32
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).long().to(device)
    Xte = torch.from_numpy(X_test).float().to(device)

    # Linear model (logistic regression = linear + softmax)
    model = nn.Linear(n_features, n_classes).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    # L2 regularisation: sklearn C = 1/lambda, so weight_decay = 1/C
    weight_decay = 1.0 / C

    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20,
                            line_search_fn="strong_wolfe")
    loss_fn = nn.CrossEntropyLoss()

    # LBFGS needs closure
    prev_loss = float('inf')
    for _ in range(max_iter // 20):  # outer loops × 20 inner steps
        def closure():
            optimizer.zero_grad()
            logits = model(Xt)
            loss = loss_fn(logits, yt)
            # L2 penalty (matches sklearn's formulation)
            l2 = 0.5 * weight_decay * (model.weight ** 2).sum()
            total = loss + l2
            total.backward()
            return total

        loss_tensor = optimizer.step(closure)
        curr_loss = loss_tensor.item() if loss_tensor is not None else prev_loss
        # Converged
        if abs(prev_loss - curr_loss) < 1e-6:
            break
        prev_loss = curr_loss

    # Predict
    with torch.inference_mode():
        logits = model(Xte)
        preds = logits.argmax(dim=1).cpu().numpy()

    del Xt, yt, Xte, model
    torch.cuda.empty_cache()
    return preds


def train_pos_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                     seed: int = 42) -> dict:
    """
    Train a logistic regression POS probe with stratified k-fold CV.
    Uses PyTorch on GPU when available, falls back to sklearn on CPU.
    Returns dict with mean accuracy, std, and per-fold accuracies.
    """
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs = []
    all_preds = []
    all_trues = []
    all_indices = []  # track which test indices belong to which fold

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if TORCH_GPU:
            y_pred = _torch_logistic_regression_gpu(
                X_train, y_train, X_test, n_classes,
                max_iter=2000, C=1.0,
            )
        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=seed,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        fold_accs.append(accuracy_score(y_test, y_pred))
        all_preds.extend(y_pred.tolist())
        all_trues.extend(y_test.tolist())
        all_indices.extend(test_idx.tolist())

    # Per-token correctness for bootstrap CI
    all_correct = [int(p == t) for p, t in zip(all_preds, all_trues)]

    return {
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": [float(a) for a in fold_accs],
        "per_token_correct": all_correct,
    }



# ─── Main Experiment ─────────────────────────────────────────────────────────

def run_pos_probe(model_path: str, k_frac: float = 0.1,
                   lang_pair: str = "en-fr", ud_path: str = None,
                   max_sentences: int = MAX_SENTENCES,
                   n_random_trials: int = 5,
                   ud_lang: str = "english") -> dict:
    """
    Run the full POS probing experiment for one model.
    ud_lang: 'english' or 'chinese' — which UD corpus to use for POS tags.
    """
    model_name = model_path.split("/")[-1]
    pair_dir = RESULTS_DIR / model_name / lang_pair

    print(f"\n{'=' * 65}")
    print(f"  POS Probe Experiment: {model_name}")
    print(f"{'=' * 65}")
    
    if TORCH_GPU:
        print("  [INFO] Accelerated by PyTorch Logistic Regression (GPU)")
    else:
        print("  [INFO] Using scikit-learn Logistic Regression (CPU)")

    # ── Step 0: Load UD corpus ──
    if ud_lang == "chinese":
        print("\n[0] Loading UD Chinese-GSD corpus...")
        if ud_path is None:
            ud_path = download_ud_chinese()
    else:
        print("\n[0] Loading UD English-EWT corpus...")
        if ud_path is None:
            ud_path = download_ud_english()
    sentences = parse_conllu(ud_path)
    total_tokens = sum(len(s) for s in sentences)
    print(f"    {len(sentences)} sentences, {total_tokens} tokens")

    # ── Step 1: Load eigendecomposition ──
    print("\n[1] Loading eigendecomposition...")
    cache_path = SPECTRAL_DIR / "eigen_cache" / model_name / lang_pair / "eigen.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        eigenvalues, eigenvectors = data["eigenvalues"], data["eigenvectors"]
        print(f"    Loaded from cache: {cache_path}")
    else:
        cov_path = pair_dir / "cov_src.pt"
        if not cov_path.exists():
            cov_path = RESULTS_DIR / model_name / "cov_en.pt"
        if not cov_path.exists():
            print(f"    ERROR: No covariance matrix found for {model_name}")
            return {}
        cov = torch.load(cov_path, map_location="cpu", weights_only=True).float()
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        except RuntimeError:
            # MKL/LAPACK bug on CPU for some matrix sizes — use GPU instead
            print("    [WARN] CPU eigh failed (MKL bug), retrying on GPU...")
            cov_gpu = cov.to("cuda")
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_gpu)
            eigenvalues = eigenvalues.cpu()
            eigenvectors = eigenvectors.cpu()
            del cov_gpu
            torch.cuda.empty_cache()
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        print(f"    Computed from covariance: {cov_path}")

    d = len(eigenvalues)
    k = max(1, int(d * k_frac))
    cumvar = np.cumsum(eigenvalues.numpy()) / eigenvalues.numpy().sum()
    print(f"    d={d}, k={k} ({k_frac:.0%} of spectrum)")
    print(f"    Top-{k} captures {cumvar[k-1]:.1%} variance")
    print(f"    Bottom-{k} captures {1.0 - cumvar[d-k]:.1%} variance")

    # ── Step 2: Load model and extract activations ──
    print("\n[2] Loading model and extracting activations...")
    model, tokenizer, handler = lrg.load_model(model_path)
    device = next(model.parameters()).device
    print(f"    Device: {device}")

    activations, pos_labels = extract_token_activations(
        model, tokenizer, handler, sentences, device,
        max_sentences=max_sentences
    )

    # Free model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    Extracted {len(pos_labels)} token activations")

    if len(pos_labels) == 0:
        print("    ERROR: No activations extracted!")
        return {}

    # ── Step 3: Filter rare tags and encode labels ──
    tag_counts = Counter(pos_labels)
    print(f"\n    POS tag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"      {tag:>8}: {count}")

    # Filter out rare tags
    valid_tags = {tag for tag, count in tag_counts.items()
                  if count >= MIN_SAMPLES_PER_TAG}
    mask = [t in valid_tags for t in pos_labels]
    activations = activations[mask]
    pos_labels = [t for t, m in zip(pos_labels, mask) if m]
    print(f"\n    After filtering (min {MIN_SAMPLES_PER_TAG} samples): "
          f"{len(pos_labels)} tokens, {len(valid_tags)} tags")

    le = LabelEncoder()
    y = le.fit_transform(pos_labels)
    tag_names = list(le.classes_)
    n_classes = len(tag_names)
    print(f"    Classes: {tag_names}")

    # ── Step 4: Run probes on different projections ──
    print(f"\n[3] Training POS probes (5-fold CV)...")

    # Define subspace selections
    top_k_indices = list(range(k))
    bottom_k_indices = list(range(d - k, d))

    # Generate random baselines
    rng = np.random.default_rng(42)
    random_index_sets = [
        sorted(rng.choice(d, size=k, replace=False).tolist())
        for _ in range(n_random_trials)
    ]

    results = {}

    # Top-k probe
    print(f"    Top-{k} eigenvectors (high-variance)...")
    X_top = project_activations(activations, eigenvectors, top_k_indices)
    probe_top = train_pos_probe(X_top, y)
    results["top_k"] = probe_top
    print(f"      Accuracy: {probe_top['mean_accuracy']:.4f} "
          f"± {probe_top['std_accuracy']:.4f}")

    # Bottom-k probe
    print(f"    Bottom-{k} eigenvectors (low-variance)...")
    X_bot = project_activations(activations, eigenvectors, bottom_k_indices)
    probe_bot = train_pos_probe(X_bot, y)
    results["bottom_k"] = probe_bot
    print(f"      Accuracy: {probe_bot['mean_accuracy']:.4f} "
          f"± {probe_bot['std_accuracy']:.4f}")

    # Random-k probes (multiple trials)
    random_accs = []
    for trial_idx, rand_indices in enumerate(random_index_sets):
        X_rand = project_activations(activations, eigenvectors, rand_indices)
        probe_rand = train_pos_probe(X_rand, y)
        random_accs.append(probe_rand['mean_accuracy'])
    results["random_k"] = {
        "mean_accuracy": float(np.mean(random_accs)),
        "std_accuracy": float(np.std(random_accs)),
        "trial_accuracies": [float(a) for a in random_accs],
    }
    print(f"    Random-{k} eigenvectors ({n_random_trials} trials)...")
    print(f"      Accuracy: {results['random_k']['mean_accuracy']:.4f} "
          f"± {results['random_k']['std_accuracy']:.4f}")

    # Full-dimensional probe (upper bound)
    print(f"    Full-{d} dimensions (upper bound)...")
    X_full = activations.detach().cpu().numpy()
    probe_full = train_pos_probe(X_full, y)
    results["full"] = probe_full
    print(f"      Accuracy: {probe_full['mean_accuracy']:.4f} "
          f"± {probe_full['std_accuracy']:.4f}")

    # ── Step 5: Statistical test ──
    print(f"\n[4] Statistical comparison...")
    from scipy import stats

    # --- Test 1: Paired t-test on fold accuracies (conservative, N=5) ---
    top_accs = probe_top['fold_accuracies']
    bot_accs = probe_bot['fold_accuracies']
    t_stat, p_val = stats.ttest_rel(top_accs, bot_accs)
    gap = probe_top['mean_accuracy'] - probe_bot['mean_accuracy']

    print(f"    Top-k vs Bottom-k accuracy gap: {gap:+.4f}")
    print(f"    Paired t-test (N={len(top_accs)} folds): t={t_stat:.3f}, p={p_val:.4f}")
    print(f"    {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} "
          f"(α=0.05)")

    # --- Test 2: Bootstrap CI on per-token accuracy difference ---
    top_correct = np.array(probe_top['per_token_correct'])
    bot_correct = np.array(probe_bot['per_token_correct'])
    n_tokens = len(top_correct)
    observed_gap = top_correct.mean() - bot_correct.mean()

    rng_boot = np.random.default_rng(42)
    n_bootstrap = 10000
    boot_gaps = []
    for _ in range(n_bootstrap):
        idx = rng_boot.choice(n_tokens, size=n_tokens, replace=True)
        boot_gap = top_correct[idx].mean() - bot_correct[idx].mean()
        boot_gaps.append(boot_gap)
    boot_gaps = np.array(boot_gaps)
    ci_lo = np.percentile(boot_gaps, 2.5)
    ci_hi = np.percentile(boot_gaps, 97.5)
    boot_significant = ci_lo > 0  # entire 95% CI above zero

    print(f"\n    Bootstrap CI (N={n_tokens} tokens, 10k resamples):")
    print(f"      Gap: {observed_gap:+.4f}  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"      {'SIGNIFICANT' if boot_significant else 'NOT SIGNIFICANT'} "
          f"(CI {'excludes' if boot_significant else 'includes'} zero)")

    results["test"] = {
        "accuracy_gap": float(gap),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_ttest": bool(p_val < 0.05),
        "bootstrap_gap": float(observed_gap),
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "significant_bootstrap": bool(boot_significant),
        "significant": bool(p_val < 0.05 or boot_significant),
    }

    # ── Step 6: Verdict ──
    # Use combined significance: either t-test (conservative) or bootstrap (higher power)
    is_significant = (p_val < 0.05) or boot_significant
    print(f"\n{'=' * 65}")
    if gap > 0 and is_significant:
        print(f"  ✓ SYNTAX SHOUTS — Top-{k} eigenvectors carry significantly")
        print(f"    more POS information than bottom-{k} ({gap:+.4f}).")
        print(f"    t-test: p={p_val:.4f}  |  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    High-variance subspace encodes syntactic structure.")
        verdict = "CONFIRMED"
    elif gap > 0:
        print(f"  ~ TREND — Top-{k} > Bottom-{k} ({gap:+.4f}) but not significant")
        print(f"    t-test: p={p_val:.4f}  |  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    More data or larger k may help.")
        verdict = "TREND"
    else:
        print(f"  ✗ NOT CONFIRMED — Bottom-{k} ≥ Top-{k}")
        print(f"    ({gap:+.4f}, t-test p={p_val:.4f}, Bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}])")
        verdict = "NOT_CONFIRMED"
    print(f"{'=' * 65}")

    results["verdict"] = verdict

    # ── Save results ──
    # Strip per_token_correct arrays before saving (only needed at runtime for bootstrap)
    serializable_probes = {}
    for probe_key, probe_val in results.items():
        if isinstance(probe_val, dict) and 'per_token_correct' in probe_val:
            serializable_probes[probe_key] = {k: v for k, v in probe_val.items()
                                               if k != 'per_token_correct'}
        else:
            serializable_probes[probe_key] = probe_val

    lang_suffix = f"_{ud_lang}" if ud_lang != "english" else ""

    full_results = {
        "model": model_path,
        "model_name": model_name,
        "ud_lang": ud_lang,
        "k": k,
        "k_frac": k_frac,
        "d": d,
        "top_k_variance_captured": float(cumvar[k - 1]),
        "bottom_k_variance_captured": float(1.0 - cumvar[d - k]),
        "n_tokens": len(pos_labels),
        "n_sentences": min(len(sentences), max_sentences) if max_sentences else len(sentences),
        "n_classes": n_classes,
        "tag_names": tag_names,
        "tag_counts": dict(Counter(pos_labels)),
        "probes": serializable_probes,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f"pos_probe_{model_name}{lang_suffix}.json"
    with open(save_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Results saved: {save_path}")

    return full_results


# ─── Multi-model comparison plot ──────────────────────────────────────────────

def plot_comparison(all_results: list, save_path: Path):
    """
    Bar chart comparing top-k, bottom-k, random-k, and full accuracy
    across multiple models.
    """
    n_models = len(all_results)
    if n_models == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n_models * 3), 6))

    model_names = [r["model_name"] for r in all_results]
    x = np.arange(n_models)
    width = 0.18

    probes = ["top_k", "bottom_k", "random_k", "full"]
    labels = ["Top-k (high-var)", "Bottom-k (low-var)", "Random-k", "Full dim"]
    colors = ["#E63946", "#457B9D", "#95a5a6", "#2A9D8F"]
    hatches = [None, None, "///", None]

    for i, (probe, label, color, hatch) in enumerate(zip(probes, labels, colors, hatches)):
        means = [r["probes"][probe]["mean_accuracy"] for r in all_results]
        stds = [r["probes"][probe]["std_accuracy"] for r in all_results]
        bars = ax.bar(x + i * width, means, width, yerr=stds,
                      label=label, color=color, alpha=0.85, capsize=4,
                      edgecolor="white", linewidth=0.5,
                      hatch=hatch if hatch else "")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("POS Classification Accuracy (5-fold CV)", fontsize=12)
    ax.set_title("M1: POS Probing — Does Syntax Shout?", fontsize=14)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_ylim(0, 1.0)

    # Annotate significance
    for j, r in enumerate(all_results):
        test = r["probes"]["test"]
        gap = test["accuracy_gap"]
        p = test["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        top_acc = r["probes"]["top_k"]["mean_accuracy"]
        ax.annotate(f"Δ={gap:+.3f} ({sig})",
                    xy=(j + 0.5 * width, top_acc + 0.02),
                    fontsize=8, ha="center", color="#E63946", fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved: {save_path}")


# ─── Sweep over k values ─────────────────────────────────────────────────────

def plot_k_sweep(all_results: list, save_path: Path):
    """
    For each model, vary k from 5% to 50% and plot POS accuracy curves.
    Uses already-extracted activations to avoid re-loading the model.
    (This function is a placeholder for the k-sweep analysis.)
    """
    # This would require caching activations. For now, we produce the
    # main top-k vs bottom-k comparison.
    pass


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M1: POS-Tag Probing Experiment")
    parser.add_argument("--models", nargs="+",
                       default=["google/gemma-2-2b"],
                       help="Model paths to test (default: gemma-2-2b)")
    parser.add_argument("--model", type=str, default=None,
                       help="Single model path (shorthand for --models)")
    parser.add_argument("--k-frac", type=float, default=0.1,
                       help="Fraction of spectrum for top/bottom k (default: 0.1)")
    parser.add_argument("--pair", default="en-fr", help="Language pair for covariance")
    parser.add_argument("--ud-path", default=None,
                       help="Path to UD CoNLL-U file (auto-downloads if absent)")
    parser.add_argument("--ud-lang", default="english", choices=["english", "chinese"],
                       help="UD corpus language: 'english' (UD_English-EWT) or 'chinese' (UD_Chinese-GSD)")
    parser.add_argument("--max-sentences", type=int, default=MAX_SENTENCES,
                       help="Max sentences to use (default: 500)")
    parser.add_argument("--n-random-trials", type=int, default=5,
                       help="Number of random-k baseline trials (default: 5)")
    parser.add_argument("--out-suffix", type=str, default="",
                       help="Optional suffix to append to output files (e.g. 'expanded')")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.k_frac != 0.1:
        OUTPUT_DIR = Path(f"results/_spectral/pos_probe_k{args.k_frac}")
    else:
        OUTPUT_DIR = Path("results/_spectral/pos_probe")

    models = [args.model] if args.model else args.models

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("#" * 65)
    print("  M1: POS-Tag Probing Experiment")
    print(f"  Models: {models}")
    print(f"  k fraction: {args.k_frac:.0%}")
    print(f"  Max sentences: {args.max_sentences}")
    print("#" * 65)

    all_results = []

    for model_path in models:
        try:
            result = run_pos_probe(
                model_path,
                k_frac=args.k_frac,
                lang_pair=args.pair,
                ud_path=args.ud_path,
                max_sentences=args.max_sentences,
                n_random_trials=args.n_random_trials,
                ud_lang=args.ud_lang,
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR on {model_path}: {e}")
            import traceback
            traceback.print_exc()
    suffix = ""
    if args.ud_lang != "english":
        suffix += f"_{args.ud_lang}"
    if args.out_suffix:
        suffix += f"_{args.out_suffix}"

    # Multi-model comparison
    if len(all_results) > 0:
        plot_comparison(all_results, OUTPUT_DIR / f"pos_probe_comparison{suffix}.png")

    # Save combined results
    combined_path = OUTPUT_DIR / f"pos_probe_all_models{suffix}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined results: {combined_path}")

    # Final summary
    summary_lines = []
    summary_lines.append(f"\n{'#' * 65}")
    summary_lines.append("  SUMMARY")
    summary_lines.append(f"{'#' * 65}")
    summary_lines.append(f"  {'Model':<22} {'Top-k':>8} {'Bot-k':>8} {'Rand-k':>8} {'Full':>8} {'Gap':>8} {'p(t)':>8} {'Boot CI':>18} {'Sig?'}")
    summary_lines.append(f"  {'-' * 106}")
    for r in all_results:
        p = r["probes"]
        t = p["test"]
        ci_str = f"[{t['bootstrap_ci_lo']:+.4f}, {t['bootstrap_ci_hi']:+.4f}]"
        summary_lines.append(f"  {r['model_name']:<22} "
              f"{p['top_k']['mean_accuracy']:>8.4f} "
              f"{p['bottom_k']['mean_accuracy']:>8.4f} "
              f"{p['random_k']['mean_accuracy']:>8.4f} "
              f"{p['full']['mean_accuracy']:>8.4f} "
              f"{t['accuracy_gap']:>+8.4f} "
              f"{t['p_value']:>8.4f} "
              f"{ci_str:>18} "
              f"{'YES' if t['significant'] else 'no'}")
    summary_lines.append(f"{'#' * 65}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save to markdown
    md_path = OUTPUT_DIR / f"pos_probe_summary{suffix}.md"
    with open(md_path, "w") as f:
        f.write("```text\n")
        f.write(summary_text.strip() + "\n")
        f.write("```\n")
    print(f"\n  Summary saved to markdown: {md_path}")


if __name__ == "__main__":
    main()
