# Spectral Anti-Concentration in Language Model Representations

> **TL;DR** — Concept directions in LLM residual streams preferentially encode in the *low-variance eigenspectral tail* of language-filtered covariance matrices, while static unembedding vectors concentrate in high-variance directions.
> This dual geometry reflects a functional rotation: syntax preferentially encodes in the high-variance subspace, forcing semantic concepts to "whisper" in the spectral tail to avoid grammatical interference. We validate these findings across 17 models using three independent extraction methods, and also definitively show that Whitened Causal Alignment (WCA) provides no geometric advantage for cross-lingual transport.

---

## Core Claims

This repository contains the code to reproduce the setup and experiments from our paper, **"Concepts Whisper While Syntax Shouts: Spectral Anti-Concentration and the Dual Geometry of Transformer Representations."**

Our work establishes four main claims:

1. **The Causal Geometry Does Not Aid Cross-Lingual Transport**: Through a matched-spectrum randomization test, we show that Whitened Causal Alignment (WCA) operates purely via spectral regularization, with the specific causal directions carrying no information for cross-lingual alignment.
2. **Spectral Anti-Concentration**: Concept representations systematically anti-concentrate in the eigenspectrum of the unembedding covariance matrix Σ, preferentially encoding in low-variance directions. This is validated via three independent extraction methods (Difference-of-means, SAE features, and Linear probes) across five architecture families.
3. **The Dual Geometry**: While contextualized representations anti-concentrate in the spectral tail, static unembedding vectors *concentrate* in high-variance directions, revealing a dual geometry between the vocabulary space and the reasoning space.
4. **Functional Basis (Concepts Whisper While Syntax Shouts)**: Syntactic information (POS tags) is preferentially encoded in the high-variance subspace. Injecting concept vectors along these high-variance eigendirections causes significantly more grammatical interference than injection along low-variance directions.

*Note: This repository focuses on providing the code to reproduce our experimental pipeline. For detailed quantitative results, statistical tests, and plots, please refer to the full paper.*

---

## Project Structure

```
.
├── data/
│   ├── paired_contexts/        # Parallel corpora (en-fr.jsonl, es-de.jsonl, ...)
│   ├── word_pairs/             # 27 counterfactual pair files ([male - female].txt, ...)
│   └── heldout_ppl_passages.txt  # Held-out passages for PPL evaluation
├── scripts/
│   ├── perform_raid.py         # RAID pipeline: Extract → Align → Transport → Visualize
│   ├── spectral_analysis.py    # Exp 1/4/5: Stratification CDFs, scaling, architecture
│   ├── utility_correlation.py  # Exp 3: THI vs WCA benefit correlation
│   ├── matched_spectrum_randomization.py  # Exp 2: Fake-covariance WCA control
│   ├── artifact_check_sae.py   # Exp 0A: Gemma Scope SAE feature extraction
│   ├── artifact_check_probe.py # Exp 0C: Linear probe concept directions
│   ├── artifact_check_unembed.py # Exp 0B: Unembedding-derived directions
│   ├── pos_probe_experiment.py # §7.2: POS-tag probing on spectral subspaces
│   ├── fix_hole4_v2_split_injection.py  # §7.1: Split-injection interference experiment
│   ├── fix_hole1_scm_ceiling.py  # App B: Corrected random SCM baselines
│   ├── fix_hole3_sae_collapse.py # App C: SAE feature collapse analysis
│   ├── fix_hole5_unembed_null.py # App D: Unembedding null model
│   ├── fix_hole6_overlay_figure.py # App E: Cross-method overlay figure
│   ├── ablate_whitening.py     # λ/truncation ablation grid
│   ├── analyze_raid.py         # Post-RAID analysis and visualization
│   ├── run_remote.sh           # Remote GPU deployment orchestration
│   ├── run_artifact_checks.sh  # Remote artifact check deployment
│   └── setup_remote.sh         # Remote environment bootstrap
├── src/
│   └── spectral_anti_concentration/
│       ├── core.py             # Core API: covariance, whitening, Procrustes, transport
│       ├── models.py           # Model handler factory and routing
│       ├── handlers/           # Per-architecture model handlers
│       └── __init__.py         # Package exports
├── pyproject.toml
└── LICENSE                     # CC BY 4.0
```

---

## Installation

```bash
# Clone and install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Environment

Create `.env` for model configuration:
```bash
MODEL_LIST="Qwen/Qwen2.5-0.5B,meta-llama/Llama-3.2-1B"
HF_TOKEN="hf_..."  # Required for gated models (Llama, Gemma)
```

For remote GPU deployment, copy `.env.deploy.example` to `.env.deploy` and configure SSH settings.

---

## Reproducing the Experiments

### Phase 1: RAID Pipeline (requires GPU)

Extract concept vectors and compute covariance matrices across 17 models:

```bash
uv run python scripts/perform_raid.py
```

### Phase 2: Spectral Analysis (CPU-only, uses saved .pt files)

```bash
# Exp 1 + 4 + 5: Stratification CDFs, Qwen scaling, Gemma vs Llama
uv run python scripts/spectral_analysis.py --experiment all

# Exp 2: Matched-spectrum randomization
uv run python scripts/matched_spectrum_randomization.py

# Exp 3: Utility correlation (requires Exp 1 output)
uv run python scripts/utility_correlation.py
```

### Phase 3: Artifact Checks (requires GPU)

```bash
# SAE-based (strongest test — Gemma Scope)
uv run python scripts/artifact_check_sae.py --model gemma-2-2b

# Linear probes (opposing L2 bias)
uv run python scripts/artifact_check_probe.py --models gemma-2-9b Llama-3.2-3B Qwen2.5-3B

# Unembedding-derived (control comparison)
uv run python scripts/artifact_check_unembed.py
```

### Phase 4: Functional Experiments (requires GPU)

```bash
# §7.1: Split-injection interference asymmetry
uv run python scripts/fix_hole4_v2_split_injection.py --model gemma-2-2b

# §7.2: POS-tag probing on spectral subspaces
uv run python scripts/pos_probe_experiment.py --model gemma-2-2b
```

### Phase 5: Supplementary Analyses (CPU-only, uses Phase 1 output)

```bash
# Corrected random SCM baselines (App B)
uv run python scripts/fix_hole1_scm_ceiling.py

# SAE feature collapse analysis (App C)
uv run python scripts/fix_hole3_sae_collapse.py

# Unembedding null model (App D)
uv run python scripts/fix_hole5_unembed_null.py

# Cross-method overlay figure (App E)
uv run python scripts/fix_hole6_overlay_figure.py
```

---

## Models Tested

| Model | Params | Hidden Dim | Mean Gini | Cond(Σ) |
|-------|--------|-----------|-----------|---------| 
| Qwen2.5-0.5B | 0.5B | 896 | −0.356 | 4.2×10³ |
| Qwen2.5-1.5B | 1.5B | 1536 | −0.282 | 5.8×10³ |
| Qwen2.5-3B | 3B | 2048 | −0.272 | 5.7×10³ |
| Qwen2.5-7B | 7B | 3584 | −0.396 | 2.9×10⁴ |
| Qwen2.5-14B | 14B | 5120 | −0.346 | 1.6×10⁴ |
| Qwen3-0.6B | 0.6B | 1024 | −0.216 | 2.6×10³ |
| Qwen3-1.7B | 1.7B | 2048 | −0.248 | 4.3×10³ |
| Qwen3-4B | 4B | 2560 | −0.312 | 1.0×10⁴ |
| Qwen3-8B | 8B | 4096 | −0.359 | 1.6×10⁴ |
| Llama-3.2-1B | 1B | 2048 | −0.140 | 1.4×10⁴ |
| Llama-3.2-3B | 3B | 3072 | −0.159 | 2.5×10⁴ |
| Gemma-2-2B | 2B | 2304 | −0.389 | 1.6×10⁵ |
| Gemma-2-9B | 9B | 3584 | −0.364 | 1.6×10⁵ |
| Mistral-7B-v0.3 | 7B | 4096 | −0.199 | 2.8×10⁵ |
| JetMoE-8B | 8B | 2048 | −0.242 | 3.7×10³ |
| OLMoE-1B-7B | 7B | 2048 | −0.076 | 7.7×10³ |
| SmolLM2-1.7B | 1.7B | 2048 | −0.436 | 2.7×10⁴ |

All 17 models show statistically significant anti-concentration (p < 10⁻⁶ individually).

---

## Concepts Tested (27)

**Verb Morphology** (10): verb-Ved, verb-Ving, verb-3pSg, V+er, V+able, V+ment, V+tion, Ving-Ved, Ving-3pSg, 3pSg-Ved  
**Semantic** (7): male-female, country-capital, thing-color, thing-part, small-big, lower-upper, frequent-infrequent  
**Grammatical** (6): comparative, superlative, un+adj, adj+ly, noun-plural, pronoun-possessive  
**Language Pairs** (4): English-French, French-German, French-Spanish, German-Spanish  

---

## Key Definitions

| Term | Definition |
|------|-----------|
| **Spectral Center of Mass (SCM)** | The cumulative variance V(k) at which 50% of concept energy is captured. SCM → 1.0 means the concept lives deep in the tail. |
| **Gini Deviation** | Signed area between the concept energy CDF and the uniform diagonal. Negative = anti-concentrated (tail-heavy). |
| **Tail-Heaviness Index (THI)** | 1 − C(k*), where C(k*) is energy captured when V(k) = 0.5. High THI = concept uses low-variance directions. |
| **WCA** | Whitened Causal Alignment — Procrustes solved in whitened space (Σ⁻¹/² transform). |

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{acharya2026concepts,
  title={Concepts Whisper While Syntax Shouts: Spectral Anti-Concentration and the Dual Geometry of Transformer Representations},
  author={Acharya, Pratyush},
  year={2026}
}
```

---

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## References

- **Park et al.** — *The Linear Representation Hypothesis and the Geometry of Large Language Models*
- **Zada et al.** — *Sparse Autoencoders Reveal Universal Feature Spaces*
- **Lim et al.** — *Language-Specific Latent Process Hinders Cross-Lingual Performance*
- **Lieberum et al.** — *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*
