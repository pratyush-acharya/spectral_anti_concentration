#!/usr/bin/env python3
"""
Vector RAID Pipeline — Multi-language experiment runner.

For each model in MODEL_LIST, for each language pair in data/paired_contexts/:
  1. Extract concept vectors (WCA + Naive Procrustes)
  2. Run analysis (analyze_raid.py)
  3. Run ablation (ablate_whitening.py)

Results are saved to results/{model_name}/{lang_pair}/ (e.g., results/Qwen2.5-0.5B/en-fr/).
Existing legacy flat results (from old en-fr-only runs) are auto-migrated.
Skips completed language pairs unless FORCE_RERUN=1.
Handles OOM gracefully and continues to next model.
"""
import torch
import numpy as np
import json
import sys
import os
import gc
import glob
import time
import shutil
import subprocess
import traceback
from tqdm import tqdm
from dotenv import load_dotenv

# Set memory and backend optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

load_dotenv()

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────
# Concept grouping (for sorted output)
# ──────────────────────────────────────────────────────────
CONCEPT_GROUPS = {
    'verb_morphology': ['[verb - Ved]', '[verb - Ving]', '[verb - 3pSg]', '[verb - V + er]',
                        '[verb - V + able]', '[verb - V + ment]', '[verb - V + tion]',
                        '[Ving - Ved]', '[Ving - 3pSg]', '[3pSg - Ved]'],
    'adj_morphology':  ['[adj - comparative]', '[adj - superlative]', '[adj - un + adj]', '[adj - adj + ly]'],
    'noun_morphology': ['[noun - plural]', '[pronoun - possessive]'],
    'language':        ['[English - French]', '[French - German]', '[French - Spanish]', '[German - Spanish]'],
    'semantic':        ['[male - female]', '[country - capital]', '[thing - color]', '[thing - part]',
                        '[small - big]', '[lower - upper]', '[frequent - infrequent]'],
}


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


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────
def load_paired_data(filename):
    pairs = []
    with open(filename, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def get_vocab_subset(pairs, lang_key, tokenizer, handler, src_lang=None, tgt_lang=None):
    token_set = set()
    for p in pairs:
        contexts = p[lang_key]
        for text in contexts:
            formatted_text = handler.format_input(text, source_lang=src_lang, target_lang_code=tgt_lang, tokenizer=tokenizer)
            tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            token_set.update(tokens)
    return list(token_set)


# ──────────────────────────────────────────────────────────
# Result checking & migration
# ──────────────────────────────────────────────────────────
def has_existing_results(pair_dir):
    """Check if a language pair directory has complete results."""
    # New naming convention
    new_files = ["stats.txt", "X_src.pt", "X_tgt.pt", "cov_src.pt", "cov_tgt.pt"]
    if all(os.path.exists(os.path.join(pair_dir, f)) for f in new_files):
        return True
    # Legacy naming (from old en-fr only runs)
    old_files = ["stats.txt", "X_en.pt", "X_fr.pt", "cov_en.pt", "cov_fr.pt"]
    if all(os.path.exists(os.path.join(pair_dir, f)) for f in old_files):
        return True
    return False


def migrate_legacy_results(base_dir):
    """Move old flat en-fr results into en-fr/ subfolder with new naming."""
    legacy_stats = os.path.join(base_dir, "stats.txt")
    en_fr_dir = os.path.join(base_dir, "en-fr")
    
    if not os.path.exists(legacy_stats):
        return  # No legacy results
    if os.path.exists(en_fr_dir) and os.path.exists(os.path.join(en_fr_dir, "stats.txt")):
        return  # Already migrated
    
    print(f"  📦 Migrating legacy flat results → en-fr/")
    os.makedirs(en_fr_dir, exist_ok=True)
    
    # File rename map (old → new)
    rename_map = {
        "X_en.pt": "X_src.pt",
        "X_fr.pt": "X_tgt.pt",
        "cov_en.pt": "cov_src.pt",
        "cov_fr.pt": "cov_tgt.pt",
        "Q_en_fr.pt": "Q.pt",
        "Q_en_fr_naive.pt": "Q_naive.pt",
    }
    
    moved = 0
    for f in os.listdir(base_dir):
        src_path = os.path.join(base_dir, f)
        if os.path.isfile(src_path):
            # Skip non-result files
            if not (f.endswith('.pt') or f.endswith('.txt') or f.endswith('.png')):
                continue
            dest_name = rename_map.get(f, f)
            dest_path = os.path.join(en_fr_dir, dest_name)
            shutil.move(src_path, dest_path)
            moved += 1
    
    # Also move ablation folder if it exists
    abl_dir = os.path.join(base_dir, "ablation")
    if os.path.isdir(abl_dir):
        shutil.move(abl_dir, os.path.join(en_fr_dir, "ablation"))
        moved += 1
    
    print(f"    Moved {moved} files to en-fr/")


# ──────────────────────────────────────────────────────────
# Post-analysis
# ──────────────────────────────────────────────────────────
def run_post_analysis(result_dir, label):
    """Run analyze_raid.py and ablate_whitening.py on completed results."""
    analyze_script = os.path.join(SCRIPTS_DIR, "analyze_raid.py")
    ablate_script = os.path.join(SCRIPTS_DIR, "ablate_whitening.py")
    
    if os.path.exists(analyze_script):
        print(f"\n  [Post] Running analysis on {label}...")
        try:
            subprocess.run([sys.executable, analyze_script, result_dir], timeout=300, check=False)
        except Exception as e:
            print(f"  [Post] Analysis failed: {e}")
    
    if os.path.exists(ablate_script):
        cov_path = os.path.join(result_dir, "cov_src.pt")
        if not os.path.exists(cov_path):
            cov_path = os.path.join(result_dir, "cov_en.pt")  # legacy fallback
        if os.path.exists(cov_path):
            print(f"  [Post] Running ablation on {label}...")
            try:
                subprocess.run([sys.executable, ablate_script, result_dir], timeout=600, check=False)
            except Exception as e:
                print(f"  [Post] Ablation failed: {e}")
        else:
            print(f"  [Post] Skipping ablation — no covariances found")


# ──────────────────────────────────────────────────────────
# GPU helpers
# ──────────────────────────────────────────────────────────
def log_gpu_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {alloc:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ──────────────────────────────────────────────────────────
# Process a single language pair (model already loaded)
# ──────────────────────────────────────────────────────────
def process_language_pair(pair_name, pair_file, pair_dir, model, tokenizer, handler,
                          device, W_U, concepts_raw, concept_names):
    """Run RAID for one language pair. Model is already loaded."""
    os.makedirs(pair_dir, exist_ok=True)
    
    print(f"\n  ── {pair_name} ──")
    
    # Load paired data
    pairs = load_paired_data(pair_file)
    first = pairs[0]
    lang0, lang1 = first.get("lang0", "src"), first.get("lang1", "tgt")
    print(f"    Languages: {lang0} → {lang1}, {len(pairs)} word pairs")
    
    # Collect vocabularies
    print(f"    Collecting vocabularies...")
    vocab_src = get_vocab_subset(pairs, "contexts0", tokenizer, handler, src_lang=lang0, tgt_lang=lang1)
    vocab_tgt = get_vocab_subset(pairs, "contexts1", tokenizer, handler, src_lang=lang1, tgt_lang=lang0)
    print(f"    Vocab: {lang0}={len(vocab_src)}, {lang1}={len(vocab_tgt)}")
    
    # Compute covariances (on GPU with W_U, then move to CPU)
    print(f"    Computing covariances...")
    cov_src = lrg.compute_covariance_matrix(W_U, token_indices=vocab_src).cpu()
    cov_tgt = lrg.compute_covariance_matrix(W_U, token_indices=vocab_tgt).cpu()
    torch.save(cov_src, os.path.join(pair_dir, "cov_src.pt"))
    torch.save(cov_tgt, os.path.join(pair_dir, "cov_tgt.pt"))
    
    # Compute whitening (all on CPU now)
    print(f"    Computing whitening transforms...")
    psi_src, sqrt_src = lrg.get_whitening_transform(cov_src, reg_lambda=1e-3)
    psi_tgt, sqrt_tgt = lrg.get_whitening_transform(cov_tgt, reg_lambda=1e-3)
    
    # Get activations
    print(f"    Gathering activations ({len(pairs)} pairs)...")
    X_src_list, X_tgt_list = [], []
    
    for p in tqdm(pairs, desc=f"    {pair_name}"):
        batch_src = [handler.format_input(t, lang0, lang1, tokenizer) for t in p["contexts0"]]
        batch_tgt = [handler.format_input(t, lang1, lang0, tokenizer) for t in p["contexts1"]]
        
        embs_src = lrg.get_embeddings(batch_src, model, tokenizer, handler, device,
                                       batch_size=1, target_words=[p["word0"]] * len(batch_src))
        X_src_list.append(torch.mean(embs_src, dim=0).cpu().float())
        
        embs_tgt = lrg.get_embeddings(batch_tgt, model, tokenizer, handler, device,
                                       batch_size=1, target_words=[p["word1"]] * len(batch_tgt))
        X_tgt_list.append(torch.mean(embs_tgt, dim=0).cpu().float())
    
    X_src = torch.stack(X_src_list)
    X_tgt = torch.stack(X_tgt_list)
    torch.save(X_src, os.path.join(pair_dir, "X_src.pt"))
    torch.save(X_tgt, os.path.join(pair_dir, "X_tgt.pt"))
    
    # Solve Procrustes (both methods)
    print(f"    Solving Procrustes...")
    Q = lrg.solve_whitened_procrustes(X_src, X_tgt, psi_src, psi_tgt)
    Q_naive = lrg.solve_naive_procrustes(X_src, X_tgt)
    torch.save(Q.cpu(), os.path.join(pair_dir, "Q.pt"))
    torch.save(Q_naive.cpu(), os.path.join(pair_dir, "Q_naive.pt"))
    
    # Transport concepts
    print(f"    Transporting {len(concept_names)} concepts...")
    concepts_raid = {}
    concepts_naive = {}
    
    for name in concept_names:
        v_raw = concepts_raw[name].float()
        v_raid = lrg.transport_concept_vector(v_raw, psi_src, sqrt_tgt, Q)
        v_naive = lrg.transport_naive(v_raw, Q_naive)
        concepts_raid[name] = v_raid.cpu()
        concepts_naive[name] = v_naive.cpu()
        
        torch.save(v_raw.cpu(), os.path.join(pair_dir, f"v_raw_{name}.pt"))
        torch.save(v_raid.cpu(), os.path.join(pair_dir, f"v_raid_{name}.pt"))
        torch.save(v_naive.cpu(), os.path.join(pair_dir, f"v_naive_{name}.pt"))
    
    # Visualizations
    raw_vectors = [concepts_raw[n] for n in concept_names]
    raid_vectors = [concepts_raid[n] for n in concept_names]
    
    lrg.plot_similarity_matrix(raw_vectors, concept_names,
                                f"Concept Similarity ({lang0}/Raw)",
                                os.path.join(pair_dir, "sim_matrix_raw.png"))
    lrg.plot_similarity_matrix(raid_vectors, concept_names,
                                f"Concept Similarity ({lang1}/Transported)",
                                os.path.join(pair_dir, "sim_matrix_raid.png"))
    
    raw_stack = torch.stack(raw_vectors)
    raid_stack = torch.stack(raid_vectors)
    lrg.hist_measurement(X_src.cpu(), X_tgt.cpu(), raw_stack,
                          concept_names, base=lang0.capitalize(), target=lang1.capitalize(),
                          save_path=os.path.join(pair_dir, "proj_raw_all.png"))
    lrg.hist_measurement(X_src.cpu(), X_tgt.cpu(), raid_stack,
                          concept_names, base=lang0.capitalize(), target=lang1.capitalize(),
                          save_path=os.path.join(pair_dir, "proj_raid_all.png"))
    
    # Write stats
    with open(os.path.join(pair_dir, "stats.txt"), "w") as f:
        f.write(f"Language Pair: {lang0} → {lang1}\n")
        f.write(f"Hidden Dim: {W_U.shape[1]}\n")
        f.write(f"Vocab ({lang0}): {len(vocab_src)}\n")
        f.write(f"Vocab ({lang1}): {len(vocab_tgt)}\n")
        f.write(f"Paired Contexts: {len(pairs)}\n")
        f.write(f"Concepts: {len(concept_names)}\n\n")
        
        for name in concept_names:
            v_raw = concepts_raw[name].float()
            v_raid = concepts_raid[name].float()
            v_naive = concepts_naive[name].float()
            
            sim_wca = torch.dot(v_raw, v_raid) / (torch.norm(v_raw) * torch.norm(v_raid)).clamp(min=1e-10)
            sim_naive = torch.dot(v_raw, v_naive) / (torch.norm(v_raw) * torch.norm(v_naive)).clamp(min=1e-10)
            
            f.write(f"Concept: {name}\n")
            f.write(f"  Raw Norm: {torch.norm(v_raw):.4f}\n")
            f.write(f"  WCA Norm: {torch.norm(v_raid):.4f}\n")
            f.write(f"  Naive Norm: {torch.norm(v_naive):.4f}\n")
            f.write(f"  Cosine Sim (Raw vs WCA): {sim_wca.item():.4f}\n")
            f.write(f"  Cosine Sim (Raw vs Naive): {sim_naive.item():.4f}\n\n")
    
    print(f"    ✓ {pair_name} done")
    return True


# ──────────────────────────────────────────────────────────
# Process all language pairs for a single model
# ──────────────────────────────────────────────────────────
def run_raid_on_model(model_path):
    """Run the full RAID pipeline on a single model across all language pairs."""
    cleanup_gpu()
    
    model_name = model_path.split("/")[-1]
    base_dir = f"results/{model_name}"
    os.makedirs(base_dir, exist_ok=True)
    
    force = os.environ.get("FORCE_RERUN", "0") == "1"
    
    # Migrate old flat results to en-fr/ subfolder
    migrate_legacy_results(base_dir)
    
    # Discover language pairs
    pair_files = sorted(glob.glob("data/paired_contexts/*.jsonl"))
    if not pair_files:
        print(f"  ERROR: No language pair data found in data/paired_contexts/")
        return "FAILED"
    
    # Determine which pairs need running
    pairs_to_run = []
    for pf in pair_files:
        pair_name = os.path.basename(pf).replace(".jsonl", "")
        pair_dir = os.path.join(base_dir, pair_name)
        
        if has_existing_results(pair_dir) and not force:
            print(f"  SKIP: {pair_name} (results exist)")
            # Still run post-analysis (scripts may have been updated)
            run_post_analysis(pair_dir, f"{model_name}/{pair_name}")
            continue
        
        pairs_to_run.append((pair_name, pf, pair_dir))
    
    if not pairs_to_run:
        print(f"  All {len(pair_files)} language pairs complete for {model_name}")
        return "SKIP"
    
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  PROCESSING: {model_name}")
    print(f"  Model: {model_path}")
    print(f"  Pairs to run: {', '.join(p[0] for p in pairs_to_run)}")
    print(f"{'='*60}")
    
    # ── Load model (once for all pairs) ──
    print("\n[1] Loading model...")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    
    model, tokenizer, handler = None, None, None
    for attempt in range(3):
        try:
            model, tokenizer, handler = lrg.load_model(model_path)
            break
        except Exception as e:
            err_str = str(e)
            is_transient = any(kw in err_str.lower() for kw in ["timeout", "timed out", "connection"])
            if is_transient and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"  Load attempt {attempt+1}/3 failed (transient): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
                cleanup_gpu()
            else:
                print(f"  FAILED to load {model_name}: {e}")
                traceback.print_exc()
                for _, _, pair_dir in pairs_to_run:
                    os.makedirs(pair_dir, exist_ok=True)
                    with open(os.path.join(pair_dir, "error.txt"), "w") as f:
                        f.write(f"Load failed: {e}\n{traceback.format_exc()}")
                return "FAILED"
    
    log_gpu_memory()
    
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    W_U = lrg.get_lm_head(model).weight.detach().to(dtype=torch.float32)
    print(f"  LM Head shape: {W_U.shape}")
    
    # ── Extract concept directions (once — these are model-level, not pair-level) ──
    print(f"\n[2] Extracting concept directions...")
    concept_files = sorted(glob.glob("data/word_pairs/*.txt"))
    concepts_raw = {}
    concept_names = []
    
    for concept_file in tqdm(concept_files, desc="  Concepts"):
        name = os.path.basename(concept_file).replace('.txt', '')
        base_words, target_words = lrg.get_counterfactual_pairs(concept_file, tokenizer)
        if not base_words:
            continue
        v_raw, _ = lrg.concept_direction(base_words, target_words, model, tokenizer, handler, device)
        concepts_raw[name] = v_raw.cpu()
        concept_names.append(name)
    
    concept_names = sort_concept_names(concept_names)
    print(f"  Found {len(concept_names)} concepts")
    
    # ── Process each language pair (model stays loaded) ──
    print(f"\n[3] Processing {len(pairs_to_run)} language pairs...")
    pair_results = []
    
    for pair_name, pair_file, pair_dir in pairs_to_run:
        try:
            process_language_pair(pair_name, pair_file, pair_dir, model, tokenizer,
                                  handler, device, W_U, concepts_raw, concept_names)
            pair_results.append((pair_name, "SUCCESS"))
        except torch.cuda.OutOfMemoryError:
            print(f"\n    ✗ OOM on {pair_name}")
            cleanup_gpu()
            pair_results.append((pair_name, "OOM"))
            os.makedirs(pair_dir, exist_ok=True)
            with open(os.path.join(pair_dir, "error.txt"), "w") as ef:
                ef.write(f"OutOfMemoryError\n{traceback.format_exc()}")
        except Exception as e:
            print(f"\n    ✗ FAILED {pair_name}: {e}")
            traceback.print_exc()
            pair_results.append((pair_name, f"FAILED: {e}"))
            os.makedirs(pair_dir, exist_ok=True)
            with open(os.path.join(pair_dir, "error.txt"), "w") as ef:
                ef.write(f"{e}\n{traceback.format_exc()}")
    
    # ── Cleanup ──
    del model, W_U
    cleanup_gpu()
    
    elapsed = time.time() - t_start
    print(f"\n  ✓ {model_name}: {len([r for r in pair_results if r[1]=='SUCCESS'])}/{len(pair_results)} pairs in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    
    # ── Post-analysis for each completed pair ──
    for pair_name, status in pair_results:
        if status == "SUCCESS":
            pair_dir = os.path.join(base_dir, pair_name)
            run_post_analysis(pair_dir, f"{model_name}/{pair_name}")
    
    return "SUCCESS" if all(s == "SUCCESS" for _, s in pair_results) else "PARTIAL"


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    models_env = os.environ.get("MODEL_LIST", "")
    models = [m.strip() for m in models_env.split(",") if m.strip()]
    
    if not models:
        print("No models found in MODEL_LIST env var.")
        return
    
    # Discover language pairs
    pair_files = sorted(glob.glob("data/paired_contexts/*.jsonl"))
    pair_names = [os.path.basename(f).replace(".jsonl", "") for f in pair_files]
    
    total = len(models)
    print(f"\n{'#'*60}")
    print(f"  Vector RAID Pipeline (Multi-Language)")
    print(f"  Models: {total}")
    print(f"  Language pairs: {', '.join(pair_names)} ({len(pair_names)} pairs)")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'#'*60}")
    
    results_summary = []
    
    for i, model_path in enumerate(models, 1):
        model_name = model_path.split("/")[-1]
        print(f"\n\n{'━'*60}")
        print(f"  [{i}/{total}] {model_name}")
        print(f"{'━'*60}")
        
        try:
            status = run_raid_on_model(model_path)
            results_summary.append((model_name, status))
        except torch.cuda.OutOfMemoryError:
            print(f"\n  ✗ OOM: {model_name}")
            cleanup_gpu()
            results_summary.append((model_name, "OOM"))
            os.makedirs(f"results/{model_name}", exist_ok=True)
            with open(f"results/{model_name}/error.txt", "w") as f:
                f.write(f"OutOfMemoryError\n{traceback.format_exc()}")
        except Exception as e:
            print(f"\n  ✗ FAILED: {model_name} — {e}")
            traceback.print_exc()
            cleanup_gpu()
            results_summary.append((model_name, f"FAILED: {e}"))
            os.makedirs(f"results/{model_name}", exist_ok=True)
            with open(f"results/{model_name}/error.txt", "w") as f:
                f.write(f"{e}\n{traceback.format_exc()}")
    
    # Final summary
    print(f"\n\n{'#'*60}")
    print(f"  RAID COMPLETE — Summary")
    print(f"{'#'*60}")
    for name, status in results_summary:
        icon = "✓" if status in ("SUCCESS", "SKIP") else "◐" if status == "PARTIAL" else "✗"
        print(f"  {icon} {name}: {status}")
    
    successes = sum(1 for _, s in results_summary if s in ("SUCCESS", "SKIP"))
    print(f"\n  {successes}/{total} models completed successfully.")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
