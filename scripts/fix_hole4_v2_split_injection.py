#!/usr/bin/env python3
"""
Hole 4 Fix v2: Redesigned Split-Injection Causal Intervention
==============================================================

Redesign of the original fix_hole4_split_injection.py with stronger statistical
power to test the "interference avoidance" hypothesis.

Key changes from v1:
  1. Held-out PPL evaluation: paragraph-length passages from data/heldout_ppl_passages.txt
     (~200-500 tokens each) instead of 8 short sentences.
  2. Multi-layer steering: hooks a band of 4 layers (50%, 62%, 75%, 87% depth)
     simultaneously, amplifying the intervention signal.
  3. Higher alpha values: [1, 5, 10, 20, 50, 100] — low alpha produces no signal.
  4. Separate concept-flip vs PPL prompts: concept flip uses short prompts + logit shift;
     PPL uses held-out passages only (no confound).
  5. Better statistics: bootstrap CIs on shout-vs-whisper PPL difference, Cohen's d
     effect sizes, all available concepts (not just 4 hardcoded).

For a concept vector v (difference-of-means), we decompose it into:
  v_shout  = projection onto top-k eigenvectors of Sigma (high-variance subspace)
  v_whisper = projection onto bottom-k eigenvectors       (low-variance subspace)

Prediction:
  - v_whisper should flip the concept with LOW perplexity increase
  - v_shout  should cause HIGH perplexity increase (grammar disruption)

Usage (on remote GPU):
    uv run python scripts/fix_hole4_v2_split_injection.py --model Qwen/Qwen2.5-3B
    uv run python scripts/fix_hole4_v2_split_injection.py --model google/gemma-2-2b --alphas 5 10 20 50
"""

import argparse
import json
import os
import sys
import gc
import glob
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import spectral_anti_concentration as lrg

RESULTS_DIR = Path("results")
SPECTRAL_DIR = Path("results/_spectral")
OUTPUT_DIR = Path("results/_spectral/hole_fixes")


# ── Concept test configs (all 27 concepts) ──
# Each concept has:
#   test_prompts: sentences where steering should shift output
#   pos_keywords: tokens expected when steering toward the SECOND word in the pair
#   neg_keywords: tokens expected when steering toward the FIRST word in the pair
# Convention: concept "[A - B]" means pos_keywords relate to B, neg_keywords to A.

CONCEPT_TESTS = {
    # ── Verb Morphology (10 concepts) ──
    '[verb - Ved]': {
        'test_prompts': [
            "Yesterday the team",
            "Last week she",
            "Before the meeting they",
            "In the past he",
            "Earlier that day we",
            "Once upon a time they",
            "She suddenly",
            "The company recently",
        ],
        'pos_keywords': ['accepted', 'achieved', 'created', 'decided', 'discovered',
                         'enjoyed', 'followed', 'improved', 'received', 'asked', 'agreed'],
        'neg_keywords': ['accept', 'achieve', 'create', 'decide', 'discover',
                         'enjoy', 'follow', 'improve', 'receive', 'ask', 'agree'],
    },
    '[verb - Ving]': {
        'test_prompts': [
            "She is currently",
            "They are busy",
            "He keeps on",
            "We have been",
            "The children started",
            "I noticed them",
            "The machine is",
            "They continued",
        ],
        'pos_keywords': ['achieving', 'creating', 'developing', 'following', 'improving',
                         'learning', 'providing', 'running', 'working', 'including'],
        'neg_keywords': ['achieve', 'create', 'develop', 'follow', 'improve',
                         'learn', 'provide', 'run', 'work', 'include'],
    },
    '[verb - 3pSg]': {
        'test_prompts': [
            "He usually",
            "She always",
            "The system",
            "Every morning he",
            "The professor",
            "This method",
            "The company",
            "The algorithm",
        ],
        'pos_keywords': ['accepts', 'achieves', 'allows', 'appears', 'creates',
                         'includes', 'provides', 'requires', 'seems', 'tells'],
        'neg_keywords': ['accept', 'achieve', 'allow', 'appear', 'create',
                         'include', 'provide', 'require', 'seem', 'tell'],
    },
    '[verb - V + er]': {
        'test_prompts': [
            "The person who performs is called a",
            "Someone who teaches is a",
            "A person who writes is a",
            "One who manages others is a",
            "The one who speaks is the",
            "A person who listens is a",
            "Someone who explores is an",
            "The one who leads is a",
        ],
        'pos_keywords': ['teacher', 'writer', 'speaker', 'manager', 'performer',
                         'listener', 'explorer', 'leader', 'developer', 'publisher'],
        'neg_keywords': ['teach', 'write', 'speak', 'manage', 'perform',
                         'listen', 'explore', 'lead', 'develop', 'publish'],
    },
    '[verb - V + able]': {
        'test_prompts': [
            "This task is completely",
            "The results are",
            "The problem is quite",
            "This product is very",
            "The plan was deemed",
            "These conditions are",
            "The outcome is",
            "The solution seems",
        ],
        'pos_keywords': ['acceptable', 'achievable', 'affordable', 'believable', 'enjoyable',
                         'manageable', 'predictable', 'sustainable', 'understandable', 'reliable'],
        'neg_keywords': ['accept', 'achieve', 'afford', 'believe', 'enjoy',
                         'manage', 'predict', 'sustain', 'understand', 'rely'],
    },
    '[verb - V + ment]': {
        'test_prompts': [
            "The company announced a new",
            "This represents a major",
            "They celebrated the",
            "The government issued a formal",
            "The board approved the",
            "We witnessed a significant",
            "The report highlighted the",
            "The final outcome was the",
        ],
        'pos_keywords': ['achievement', 'agreement', 'announcement', 'development',
                         'improvement', 'investment', 'management', 'requirement',
                         'establishment', 'commitment'],
        'neg_keywords': ['achieve', 'agree', 'announce', 'develop', 'improve',
                         'invest', 'manage', 'require', 'establish', 'commit'],
    },
    '[verb - V + tion]': {
        'test_prompts': [
            "The official document required",
            "This process involves",
            "The committee reviewed the",
            "We need a thorough",
            "The research led to a new",
            "The result of the study was the",
            "They proposed an important",
            "The government announced the",
        ],
        'pos_keywords': ['examination', 'exploration', 'imagination', 'installation',
                         'observation', 'organization', 'preparation', 'realization',
                         'determination', 'declaration'],
        'neg_keywords': ['examine', 'explore', 'imagine', 'install', 'observe',
                         'organize', 'prepare', 'realize', 'determine', 'declare'],
    },
    '[Ving - Ved]': {
        'test_prompts': [
            "The project was",
            "After the process the result was",
            "The report stated it was",
            "By the end it had been",
            "The task was successfully",
            "The experiment was",
            "The building was",
            "The document was",
        ],
        'pos_keywords': ['added', 'agreed', 'created', 'decided', 'developed',
                         'established', 'included', 'received', 'sent', 'told'],
        'neg_keywords': ['adding', 'agreeing', 'creating', 'deciding', 'developing',
                         'establishing', 'including', 'receiving', 'sending', 'telling'],
    },
    '[Ving - 3pSg]': {
        'test_prompts': [
            "He typically",
            "The system regularly",
            "She often",
            "The machine",
            "This approach",
            "The company",
            "The model",
            "It frequently",
        ],
        'pos_keywords': ['adds', 'allows', 'appears', 'includes', 'involves',
                         'provides', 'receives', 'remains', 'requires', 'seems'],
        'neg_keywords': ['adding', 'allowing', 'appearing', 'including', 'involving',
                         'providing', 'receiving', 'remaining', 'requiring', 'seeming'],
    },
    '[3pSg - Ved]': {
        'test_prompts': [
            "Last year the company",
            "Previously the team",
            "Before that it",
            "In the old days the leader",
            "Once the system",
            "Back then the committee",
            "A decade ago the organization",
            "The experiment previously",
        ],
        'pos_keywords': ['added', 'agreed', 'allowed', 'appeared', 'created',
                         'decided', 'developed', 'included', 'received', 'told'],
        'neg_keywords': ['adds', 'agrees', 'allows', 'appears', 'creates',
                         'decides', 'develops', 'includes', 'receives', 'tells'],
    },

    # ── Adjective Morphology (4 concepts) ──
    '[adj - comparative]': {
        'test_prompts': [
            "This one is much",
            "Compared to the other it is",
            "The second option is even",
            "After the upgrade it became",
            "Than the previous version it is",
            "The new model is significantly",
            "In comparison this feels",
            "She became even",
        ],
        'pos_keywords': ['bigger', 'better', 'faster', 'stronger', 'higher',
                         'cheaper', 'easier', 'longer', 'smaller', 'harder'],
        'neg_keywords': ['big', 'good', 'fast', 'strong', 'high',
                         'cheap', 'easy', 'long', 'small', 'hard'],
    },
    '[adj - superlative]': {
        'test_prompts': [
            "This is the most",
            "It was the",
            "Of all options this is the",
            "Among all candidates this is the",
            "In the entire group the",
            "Without a doubt the",
            "The record for the",
            "This was by far the",
        ],
        'pos_keywords': ['biggest', 'best', 'fastest', 'strongest', 'highest',
                         'cheapest', 'easiest', 'longest', 'smallest', 'hardest'],
        'neg_keywords': ['big', 'good', 'fast', 'strong', 'high',
                         'cheap', 'easy', 'long', 'small', 'hard'],
    },
    '[adj - un + adj]': {
        'test_prompts': [
            "The situation was completely",
            "This outcome was",
            "The result was rather",
            "The response seemed",
            "Many found it",
            "The proposal was deemed",
            "The decision was considered",
            "The behavior was",
        ],
        'pos_keywords': ['unable', 'unacceptable', 'unavailable', 'unaware', 'uncertain',
                         'uncomfortable', 'unexpected', 'unfortunate', 'unhappy', 'unknown',
                         'unpleasant', 'unpopular', 'unreasonable', 'unusual'],
        'neg_keywords': ['able', 'acceptable', 'available', 'aware', 'certain',
                         'comfortable', 'expected', 'fortunate', 'happy', 'known',
                         'pleasant', 'popular', 'reasonable', 'usual'],
    },
    '[adj - adj + ly]': {
        'test_prompts': [
            "The team performed",
            "She spoke about it",
            "The results improved",
            "He handled the situation",
            "The policy was",
            "The market shifted",
            "The technology advanced",
            "They responded",
        ],
        'pos_keywords': ['actually', 'apparently', 'critically', 'effectively', 'globally',
                         'immediately', 'obviously', 'previously', 'seriously', 'significantly',
                         'similarly', 'strongly', 'successfully', 'typically'],
        'neg_keywords': ['actual', 'apparent', 'critical', 'effective', 'global',
                         'immediate', 'obvious', 'previous', 'serious', 'significant',
                         'similar', 'strong', 'successful', 'typical'],
    },

    # ── Noun Morphology (2 concepts) ──
    '[noun - plural]': {
        'test_prompts': [
            "I saw one",
            "There was a single",
            "He bought a",
            "She found the",
            "They noticed a",
            "We observed a",
            "The report mentioned a",
            "I need another",
        ],
        'pos_keywords': ['cars', 'days', 'years', 'ideas', 'problems', 'students',
                         'things', 'people', 'children', 'many', 'several',
                         'items', 'groups', 'systems', 'members'],
        'neg_keywords': ['car', 'day', 'year', 'idea', 'problem', 'student',
                         'thing', 'person', 'child', 'single', 'one', 'a '],
    },
    '[pronoun - possessive]': {
        'test_prompts': [
            "The student forgot to bring",
            "Everyone should do",
            "The team completed",
            "Each person submitted",
            "The traveler packed",
            "The author wrote about",
            "The worker finished",
            "The child lost",
        ],
        'pos_keywords': ['his', 'their', 'our', 'your', 'her', 'its', 'my'],
        'neg_keywords': ['he', 'they', 'we', 'you', 'she', 'it', 'I'],
    },

    # ── Language Pairs (4 concepts) ──
    '[English - French]': {
        'test_prompts': [
            "The word for house is",
            "In another language cat is",
            "The translation of water is",
            "The foreign word for book is",
            "In that language dog means",
            "The word for bread is",
            "Mother translates to",
            "The foreign term for car is",
        ],
        'pos_keywords': ['maison', 'chat', 'eau', 'livre', 'chien', 'pain', 'mère',
                         'voiture', 'le', 'la', 'les', 'des', 'un', 'une', 'du'],
        'neg_keywords': ['house', 'cat', 'water', 'book', 'dog', 'bread', 'mother',
                         'car', 'the', 'a', 'an', 'of'],
    },
    '[French - German]': {
        'test_prompts': [
            "The word for house is",
            "In another language cat is",
            "The translation of water is",
            "The foreign word for book is",
            "In that language dog means",
            "The word for bread is",
            "Mother translates to",
            "The foreign term for car is",
        ],
        'pos_keywords': ['Haus', 'Katze', 'Wasser', 'Buch', 'Hund', 'Brot', 'Mutter',
                         'Auto', 'der', 'die', 'das', 'ein', 'eine', 'und', 'ist'],
        'neg_keywords': ['maison', 'chat', 'eau', 'livre', 'chien', 'pain', 'mère',
                         'voiture', 'le', 'la', 'les', 'des', 'un', 'une'],
    },
    '[French - Spanish]': {
        'test_prompts': [
            "The word for house is",
            "In another language cat is",
            "The translation of water is",
            "The foreign word for book is",
            "In that language dog means",
            "The word for bread is",
            "Mother translates to",
            "The foreign term for car is",
        ],
        'pos_keywords': ['casa', 'gato', 'agua', 'libro', 'perro', 'pan', 'madre',
                         'coche', 'el', 'la', 'los', 'las', 'un', 'una', 'del', 'es'],
        'neg_keywords': ['maison', 'chat', 'eau', 'livre', 'chien', 'pain', 'mère',
                         'voiture', 'le', 'la', 'les', 'des'],
    },
    '[German - Spanish]': {
        'test_prompts': [
            "The word for house is",
            "In another language cat is",
            "The translation of water is",
            "The foreign word for book is",
            "In that language dog means",
            "The word for bread is",
            "Mother translates to",
            "The foreign term for car is",
        ],
        'pos_keywords': ['casa', 'gato', 'agua', 'libro', 'perro', 'pan', 'madre',
                         'coche', 'el', 'la', 'los', 'las', 'un', 'una', 'es'],
        'neg_keywords': ['Haus', 'Katze', 'Wasser', 'Buch', 'Hund', 'Brot', 'Mutter',
                         'Auto', 'der', 'die', 'das', 'ein', 'eine'],
    },

    # ── Semantic (7 concepts) ──
    '[male - female]': {
        'test_prompts': [
            "The doctor walked into the room and",
            "The teacher told the students that",
            "The CEO announced that",
            "The nurse carefully checked the",
            "My friend said that",
            "The engineer explained how",
            "The professor asked the class to",
            "The athlete finished the race and",
        ],
        'pos_keywords': ['he', 'his', 'him', 'man', 'boy', 'father', 'sir', 'mr', 'himself'],
        'neg_keywords': ['she', 'her', 'woman', 'girl', 'mother', 'madam', 'mrs', 'ms', 'herself'],
    },
    '[country - capital]': {
        'test_prompts': [
            "The most famous city in France is",
            "People travel to",
            "The government headquarters is located in",
            "The capital of Germany is",
            "Tourists often visit",
            "The main city of Italy is",
            "The political center of Spain is",
            "The largest city of the country is",
        ],
        'pos_keywords': ['paris', 'london', 'berlin', 'rome', 'madrid', 'capital', 'city',
                         'tokyo', 'washington', 'moscow', 'beijing'],
        'neg_keywords': ['france', 'germany', 'italy', 'spain', 'england', 'country', 'nation',
                         'japan', 'america', 'russia', 'china'],
    },
    '[thing - color]': {
        'test_prompts': [
            "The apple is typically",
            "The sky appears",
            "Snow is usually",
            "Grass is naturally",
            "Coal is",
            "The tomato turned",
            "The ocean looks",
            "Milk is always",
        ],
        'pos_keywords': ['red', 'blue', 'green', 'white', 'black', 'yellow', 'brown',
                         'orange', 'pink', 'gray', 'grey', 'purple'],
        'neg_keywords': ['apple', 'sky', 'snow', 'grass', 'coal', 'tomato', 'ocean',
                         'milk', 'cherry', 'cloud', 'emerald', 'sugar'],
    },
    '[thing - part]': {
        'test_prompts': [
            "The most important part of a car is the",
            "A bird is covered in",
            "The main component of a guitar is the",
            "A flower has a delicate",
            "A door swings on its",
            "The key part of a gun is the",
            "A shirt is fastened with a",
            "The sharp part of a sword is the",
        ],
        'pos_keywords': ['engine', 'feathers', 'string', 'petal', 'hinge', 'trigger',
                         'button', 'blade', 'seat', 'keyboard', 'wheel', 'sleeve'],
        'neg_keywords': ['car', 'bird', 'guitar', 'flower', 'door', 'gun',
                         'shirt', 'sword', 'chair', 'piano', 'bus', 'dress'],
    },
    '[small - big]': {
        'test_prompts': [
            "The animal was very",
            "I picked up the",
            "The size of the building was",
            "She described the room as",
            "The object appeared to be",
            "They said the portion was",
            "The crowd was surprisingly",
            "The difference in size was",
        ],
        'pos_keywords': ['tiny', 'small', 'little', 'miniature', 'compact', 'minor', 'slight'],
        'neg_keywords': ['large', 'big', 'huge', 'enormous', 'massive', 'giant', 'great', 'vast'],
    },
    '[lower - upper]': {
        'test_prompts': [
            "The letter was written in",
            "Please type your name in",
            "The title appeared in",
            "The word was displayed in",
            "The sign read in",
            "The text was formatted in",
            "The label was printed in",
            "The heading should be in",
        ],
        'pos_keywords': ['uppercase', 'upper', 'capital', 'CAPS', 'capitalized', 'majuscule'],
        'neg_keywords': ['lowercase', 'lower', 'small letter', 'minuscule'],
    },
    '[frequent - infrequent]': {
        'test_prompts': [
            "A more formal word for angry is",
            "A sophisticated synonym for big is",
            "The elevated term for happy is",
            "A literary word for cold is",
            "An uncommon way to say fast is",
            "A rare synonym for bad is",
            "The formal version of buy is",
            "A fancy word for eat is",
        ],
        'pos_keywords': ['irate', 'gorgeous', 'ecstatic', 'frigid', 'swift', 'terrible',
                         'purchase', 'consume', 'affluent', 'melancholy', 'perceive',
                         'robust', 'stroll', 'utilize', 'adversary'],
        'neg_keywords': ['angry', 'beautiful', 'happy', 'cold', 'fast', 'bad',
                         'buy', 'eat', 'rich', 'sad', 'see',
                         'strong', 'walk', 'use', 'enemy'],
    },
}


# ── Steering Infrastructure ──

class SteeringHook:
    """Adds a steering vector to residual stream at a specific layer."""

    def __init__(self, vector, alpha=1.0, position="last"):
        self.vector = vector.detach().float()
        self.alpha = alpha
        self.position = position  # "last", "all", or int
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        v = self.vector.to(h.device, dtype=h.dtype)

        if self.position == "all":
            h = h + self.alpha * v.unsqueeze(0).unsqueeze(0)
        elif self.position == "last":
            h = h.clone()
            h[:, -1, :] += self.alpha * v
        else:
            h = h.clone()
            pos = min(self.position, h.shape[1] - 1)
            h[:, pos, :] += self.alpha * v

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def attach(self, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


class MultiLayerSteeringContext:
    """
    Context manager that attaches steering hooks to multiple layers simultaneously.
    Usage:
        with MultiLayerSteeringContext(vector, alpha, layers, layer_indices, position) as ctx:
            # model forward pass here; all layers are steered
    """
    def __init__(self, vector, alpha, layers, layer_indices, position="all"):
        self.hooks = []
        for idx in layer_indices:
            hook = SteeringHook(vector, alpha=alpha, position=position)
            hook.attach(layers[idx])
            self.hooks.append(hook)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return False


def get_layers(model):
    """Get transformer layers for hook injection."""
    for attr_chain in [
        ['model', 'layers'],
        ['model', 'decoder', 'layers'],
        ['transformer', 'h'],
        ['gpt_neox', 'layers'],
    ]:
        obj = model
        for attr in attr_chain:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                obj = None
                break
        if obj is not None:
            return obj
    raise RuntimeError("Cannot find transformer layers")


# ── Held-out PPL Infrastructure ──

def load_heldout_passages(path="data/heldout_ppl_passages.txt"):
    """
    Load paragraph-length held-out passages for perplexity measurement.
    Passages are separated by double newlines in the file.
    """
    with open(path) as f:
        text = f.read()
    passages = [p.strip() for p in text.split("\n\n") if p.strip()]
    return passages


def compute_perplexity_heldout(model, tokenizer, passages, layers=None,
                                vector=None, alpha=0.0, layer_indices=None,
                                max_length=512):
    """
    Compute mean perplexity over held-out passages.
    If vector is provided, steers at ALL positions across multiple layers.
    """
    ppls = []
    for passage in passages:
        inputs = tokenizer(passage, return_tensors='pt', truncation=True,
                           max_length=max_length)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        ctx = None
        if vector is not None and alpha != 0.0 and layers is not None and layer_indices:
            ctx = MultiLayerSteeringContext(vector, alpha, layers, layer_indices,
                                            position="all")
            ctx.__enter__()

        try:
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
            ppls.append(float(torch.exp(outputs.loss)))
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    return ppls


# ── Logit shift measurement ──

def compute_logit_shift(model, tokenizer, prompt, layers, vector, alpha,
                        layer_indices, pos_keywords, neg_keywords):
    """
    Measure next-token logit mass shift toward concept keywords
    when steering is applied at multiple layers (last-token position).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors='pt', padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Baseline logits
    with torch.no_grad():
        base_logits = model(**inputs).logits[:, -1, :]
        base_probs = torch.softmax(base_logits, dim=-1)

    # Steered logits (multi-layer, last token)
    with MultiLayerSteeringContext(vector, alpha, layers, layer_indices, position="last"):
        with torch.no_grad():
            steer_logits = model(**inputs).logits[:, -1, :]
            steer_probs = torch.softmax(steer_logits, dim=-1)

    # KL divergence D_KL(steered || baseline)
    kl = torch.sum(steer_probs * (torch.log(steer_probs.clamp(min=1e-10)) -
                                   torch.log(base_probs.clamp(min=1e-10)))).item()

    def keyword_prob_mass(probs, keywords):
        total = 0.0
        for kw in keywords:
            toks = tokenizer.encode(kw, add_special_tokens=False)
            if len(toks) == 1:
                total += probs[0, toks[0]].item()
            toks_sp = tokenizer.encode(" " + kw, add_special_tokens=False)
            if len(toks_sp) == 1:
                total += probs[0, toks_sp[0]].item()
        return total

    base_pos_mass = keyword_prob_mass(base_probs, pos_keywords)
    base_neg_mass = keyword_prob_mass(base_probs, neg_keywords)
    steer_pos_mass = keyword_prob_mass(steer_probs, pos_keywords)
    steer_neg_mass = keyword_prob_mass(steer_probs, neg_keywords)

    concept_shift = (steer_pos_mass - base_pos_mass) - (steer_neg_mass - base_neg_mass)

    return {
        "kl_divergence": kl,
        "concept_shift": concept_shift,
        "pos_mass_base": base_pos_mass,
        "pos_mass_steer": steer_pos_mass,
        "neg_mass_base": base_neg_mass,
        "neg_mass_steer": steer_neg_mass,
    }


# ── Vector decomposition ──

def decompose_vector(v, eigenvectors, eigenvalues, top_k):
    """
    Decompose v into "shouting" (top-k eigenvectors) and "whispering" (bottom-k) components.
    Both are re-normalized to have the same norm as the original v.
    """
    v = v.detach().float()
    U = eigenvectors.detach().float()
    d = len(eigenvalues)
    projections = U.T @ v

    # Shouting = top-k eigenvectors (highest eigenvalues)
    mask_shout = torch.zeros(d)
    mask_shout[:top_k] = 1.0
    shout_coeffs = projections * mask_shout
    v_shout = U @ shout_coeffs

    # Whispering = bottom-k eigenvectors (lowest eigenvalues)
    mask_whisper = torch.zeros(d)
    mask_whisper[d - top_k:] = 1.0
    whisper_coeffs = projections * mask_whisper
    v_whisper = U @ whisper_coeffs

    # Middle = everything in between
    mask_mid = 1.0 - mask_shout - mask_whisper
    mid_coeffs = projections * mask_mid
    v_mid = U @ mid_coeffs

    # Energy fractions (before normalization)
    total_energy = (projections ** 2).sum().item()
    shout_energy = (shout_coeffs ** 2).sum().item() / max(total_energy, 1e-12)
    whisper_energy = (whisper_coeffs ** 2).sum().item() / max(total_energy, 1e-12)
    mid_energy = (mid_coeffs ** 2).sum().item() / max(total_energy, 1e-12)

    # Normalize each to have same norm as original
    orig_norm = v.norm().item()
    for vec in [v_shout, v_whisper, v_mid]:
        n = vec.norm().item()
        if n > 1e-12:
            vec.mul_(orig_norm / n)

    return {
        "v_shout": v_shout.detach(),
        "v_whisper": v_whisper.detach(),
        "v_mid": v_mid.detach(),
        "shout_energy_frac": shout_energy,
        "whisper_energy_frac": whisper_energy,
        "mid_energy_frac": mid_energy,
    }


# ── Statistics helpers ──

def bootstrap_ci(data, n_boot=5000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([rng.choice(data, size=n, replace=True).mean()
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(np.mean(data)), float(lo), float(hi)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    g1, g2 = np.array(group1), np.array(group2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * g1.std(ddof=1)**2 + (n2 - 1) * g2.std(ddof=1)**2) /
                         (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)


# ── Main experiment runner ──

def run_split_injection_v2(model, tokenizer, concept_name, test_config,
                            v_raw, eigenvectors, eigenvalues, layers,
                            layer_indices_steer, top_k, alphas,
                            heldout_passages):
    """
    Run redesigned split-injection experiment for one concept.
    """
    d = len(eigenvalues)

    # Decompose the concept vector
    decomp = decompose_vector(v_raw, eigenvectors, eigenvalues, top_k)

    print(f"\n    Energy distribution: "
          f"shout={decomp['shout_energy_frac']:.1%}, "
          f"mid={decomp['mid_energy_frac']:.1%}, "
          f"whisper={decomp['whisper_energy_frac']:.1%}")

    components = [
        ("full", v_raw, "Full vector"),
        ("shout", decomp["v_shout"], f"Top-{top_k} (shouting)"),
        ("whisper", decomp["v_whisper"], f"Bottom-{top_k} (whispering)"),
        ("mid", decomp["v_mid"], f"Middle (neither)"),
    ]

    all_results = []

    # Baseline PPL (no steering) on held-out passages
    ppls_baseline = compute_perplexity_heldout(model, tokenizer, heldout_passages)
    mean_ppl_base = np.mean(ppls_baseline)
    print(f"    Baseline PPL (held-out, N={len(heldout_passages)}): {mean_ppl_base:.1f}")

    for alpha in alphas:
        print(f"\n    alpha = {alpha}:")

        for comp_key, vec, comp_label in components:
            # 1. PPL on held-out passages with multi-layer steering
            ppls_steered = compute_perplexity_heldout(
                model, tokenizer, heldout_passages,
                layers=layers, vector=vec, alpha=alpha,
                layer_indices=layer_indices_steer
            )
            mean_ppl_steer = np.mean(ppls_steered)
            ppl_ratio = mean_ppl_steer / mean_ppl_base if mean_ppl_base > 0 else float('inf')

            # Per-passage PPL increases for bootstrap
            ppl_increases = [(s / b - 1) * 100 if b > 0 else 0
                             for s, b in zip(ppls_steered, ppls_baseline)]
            ppl_mean, ppl_ci_lo, ppl_ci_hi = bootstrap_ci(ppl_increases)

            # 2. Concept shift via logit measurement (last-token, multi-layer)
            concept_shifts = []
            kl_divs = []
            has_logit_data = False

            if concept_name in CONCEPT_TESTS:
                has_logit_data = True
                test_cfg = CONCEPT_TESTS[concept_name]
                for prompt in test_cfg['test_prompts']:
                    logit_data = compute_logit_shift(
                        model, tokenizer, prompt, layers, vec, alpha,
                        layer_indices_steer,
                        test_cfg['pos_keywords'], test_cfg['neg_keywords']
                    )
                    concept_shifts.append(logit_data['concept_shift'])
                    kl_divs.append(logit_data['kl_divergence'])
            else:
                print(f"        WARNING: No CONCEPT_TESTS entry for '{concept_name}' "
                      f"— PPL-only (no logit shift measurement)")

            mean_concept_shift = float(np.mean(concept_shifts)) if concept_shifts else 0.0
            mean_kl = float(np.mean(kl_divs)) if kl_divs else 0.0

            result = {
                'concept': concept_name,
                'component': comp_key,
                'component_label': comp_label,
                'alpha': alpha,
                'ppl_baseline': float(mean_ppl_base),
                'ppl_steered': float(mean_ppl_steer),
                'ppl_ratio': float(ppl_ratio),
                'ppl_increase_pct': float(ppl_mean),
                'ppl_increase_ci_lo': float(ppl_ci_lo),
                'ppl_increase_ci_hi': float(ppl_ci_hi),
                'ppl_increases_per_passage': ppl_increases,
                'mean_kl_divergence': mean_kl,
                'mean_concept_shift': mean_concept_shift,
                'has_logit_data': has_logit_data,
                'n_layers_steered': len(layer_indices_steer),
                'n_heldout_passages': len(heldout_passages),
            }
            all_results.append(result)

            ci_str = f"[{ppl_ci_lo:+.0f}%, {ppl_ci_hi:+.0f}%]"
            print(f"      {comp_label:<28} PPL: {mean_ppl_base:.1f} -> {mean_ppl_steer:.1f} "
                  f"({ppl_mean:+.0f}% {ci_str})  KL={mean_kl:.4f}  "
                  f"ConceptDelta={mean_concept_shift:+.4f}")

    return all_results, decomp


def main():
    parser = argparse.ArgumentParser(
        description="Hole 4 v2: Redesigned Split-Injection Causal Intervention")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B",
                       help="Model path (default: Qwen/Qwen2.5-3B)")
    parser.add_argument("--pair", default="en-fr", help="Language pair")
    parser.add_argument("--alphas", nargs='+', type=float,
                       default=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                       help="Steering strengths to test")
    parser.add_argument("--top-k-frac", type=float, default=0.1,
                       help="Fraction of eigenspectrum for top/bottom-k (default: 0.1 = 10%%)")
    parser.add_argument("--concepts", nargs='+', default=None,
                       help="Specific concepts to test (default: all available)")
    parser.add_argument("--heldout-path", default="data/heldout_ppl_passages.txt",
                       help="Path to held-out PPL passages file")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    pair_dir = RESULTS_DIR / model_name / args.pair

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  HOLE 4 v2: Redesigned Split-Injection Causal Intervention")
    print(f"  Model: {args.model}")
    print(f"  Alphas: {args.alphas}")
    print(f"  Top/Bottom k fraction: {args.top_k_frac:.0%}")
    print(f"  Multi-layer steering: 4 layers (50%, 62%, 75%, 87% depth)")
    print("=" * 70)

    # Load held-out passages
    print(f"\n[0] Loading held-out PPL passages from {args.heldout_path}...")
    heldout_passages = load_heldout_passages(args.heldout_path)
    print(f"    Loaded {len(heldout_passages)} passages")

    # Load model
    print("\n[1] Loading model...")
    model, tokenizer, handler = lrg.load_model(args.model)
    device = next(model.parameters()).device
    layers = get_layers(model)
    n_layers = len(layers)

    # Multi-layer steering: band at 50%, 62%, 75%, 87% depth
    layer_indices_steer = [
        int(n_layers * 0.50),
        int(n_layers * 0.62),
        int(n_layers * 0.75),
        int(n_layers * 0.87),
    ]
    # Deduplicate and clamp
    layer_indices_steer = sorted(set(min(idx, n_layers - 1) for idx in layer_indices_steer))
    print(f"    Layers: {n_layers}, steering at layers {layer_indices_steer}")

    # Load eigendecomposition
    print("\n[2] Loading eigendecomposition...")
    cache_path = SPECTRAL_DIR / "eigen_cache" / model_name / args.pair / "eigen.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        eigenvalues, eigenvectors = data["eigenvalues"], data["eigenvectors"]
    else:
        cov_path = pair_dir / "cov_src.pt"
        if not cov_path.exists():
            cov_path = RESULTS_DIR / model_name / "cov_en.pt"
        cov = torch.load(cov_path, map_location="cpu", weights_only=True).float()
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    d = len(eigenvalues)
    top_k = max(1, int(d * args.top_k_frac))
    cumvar = np.cumsum(eigenvalues.numpy()) / eigenvalues.numpy().sum()
    print(f"    d={d}, top_k={top_k} ({args.top_k_frac:.0%})")
    print(f"    Top-{top_k} eigenvalues capture {cumvar[top_k-1]:.1%} of total variance")
    print(f"    Bottom-{top_k} eigenvalues capture {1.0 - cumvar[d-top_k]:.1%} of total variance")

    # Find available concepts
    if args.concepts:
        concepts_to_test = args.concepts
    else:
        # Use all concepts that have v_raw files AND are in CONCEPT_TESTS (for logit shift)
        # Plus any extra v_raw files (PPL-only, no logit shift)
        concepts_to_test = []
        for v_path in sorted(pair_dir.glob("v_raw_*.pt")):
            cname = v_path.stem.replace("v_raw_", "")
            concepts_to_test.append(cname)

    available = []
    for cname in concepts_to_test:
        v_path = pair_dir / f"v_raw_{cname}.pt"
        if v_path.exists():
            available.append(cname)
        else:
            print(f"    SKIP {cname}: no v_raw file at {v_path}")

    print(f"\n[3] Testing {len(available)} concepts: {available}")

    # Run experiments
    all_results = []
    all_decomps = {}

    for concept_name in available:
        v_raw = torch.load(pair_dir / f"v_raw_{concept_name}.pt",
                          map_location="cpu", weights_only=True).float()

        print(f"\n  == {concept_name} ==")
        test_config = CONCEPT_TESTS.get(concept_name, None)
        results, decomp = run_split_injection_v2(
            model, tokenizer, concept_name, test_config,
            v_raw, eigenvectors, eigenvalues, layers,
            layer_indices_steer, top_k, args.alphas,
            heldout_passages
        )
        all_results.extend(results)
        all_decomps[concept_name] = {
            "shout_energy_frac": decomp["shout_energy_frac"],
            "whisper_energy_frac": decomp["whisper_energy_frac"],
            "mid_energy_frac": decomp["mid_energy_frac"],
        }

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Summary with improved statistics ──
    summary_lines = []
    def log_print(msg=""):
        print(msg)
        summary_lines.append(str(msg))

    log_print(f"\n\n{'=' * 70}")
    log_print("  SPLIT-INJECTION v2 SUMMARY")
    log_print(f"{'=' * 70}")

    from scipy import stats

    for alpha in args.alphas:
        shout_ppls = [r['ppl_increase_pct'] for r in all_results
                      if r['component'] == 'shout' and r['alpha'] == alpha]
        whisper_ppls = [r['ppl_increase_pct'] for r in all_results
                        if r['component'] == 'whisper' and r['alpha'] == alpha]
        shout_kls = [r['mean_kl_divergence'] for r in all_results
                     if r['component'] == 'shout' and r['alpha'] == alpha]
        whisper_kls = [r['mean_kl_divergence'] for r in all_results
                       if r['component'] == 'whisper' and r['alpha'] == alpha]

        if not shout_ppls or not whisper_ppls:
            continue

        d_ppl = cohens_d(shout_ppls, whisper_ppls)
        d_kl = cohens_d(shout_kls, whisper_kls)

        # Bootstrap CI on the difference (shout - whisper PPL increase)
        diffs = [s - w for s, w in zip(shout_ppls, whisper_ppls)]
        diff_mean, diff_ci_lo, diff_ci_hi = bootstrap_ci(diffs)

        log_print(f"\n  alpha = {alpha} (N={len(shout_ppls)} concepts):")
        log_print(f"    Shouting PPL increase:   {np.mean(shout_ppls):+.1f}%")
        log_print(f"    Whispering PPL increase: {np.mean(whisper_ppls):+.1f}%")
        log_print(f"    Difference (shout-whisper): {diff_mean:+.1f}% "
              f"[95% CI: {diff_ci_lo:+.1f}%, {diff_ci_hi:+.1f}%]")
        log_print(f"    Cohen's d (PPL): {d_ppl:+.3f}")
        log_print(f"    Cohen's d (KL):  {d_kl:+.3f}")

        if len(shout_ppls) > 2:
            t_ppl, p_ppl = stats.ttest_rel(shout_ppls, whisper_ppls)
            t_kl, p_kl = stats.ttest_rel(shout_kls, whisper_kls)
            log_print(f"    PPL paired t-test: t={t_ppl:.3f}, p={p_ppl:.4f}")
            log_print(f"    KL  paired t-test: t={t_kl:.3f}, p={p_kl:.4f}")

    # ── Verdict ──
    log_print(f"\n\n{'=' * 70}")
    log_print("  HOLE 4 v2 VERDICT")
    log_print(f"{'=' * 70}")

    # Use the middle alpha for verdict
    mid_alpha = args.alphas[len(args.alphas) // 2]
    shout_ppls = [r['ppl_increase_pct'] for r in all_results
                  if r['component'] == 'shout' and r['alpha'] == mid_alpha]
    whisper_ppls = [r['ppl_increase_pct'] for r in all_results
                    if r['component'] == 'whisper' and r['alpha'] == mid_alpha]
    shout_kls = [r['mean_kl_divergence'] for r in all_results
                 if r['component'] == 'shout' and r['alpha'] == mid_alpha]
    whisper_kls = [r['mean_kl_divergence'] for r in all_results
                   if r['component'] == 'whisper' and r['alpha'] == mid_alpha]
    shout_shifts = [r['mean_concept_shift'] for r in all_results
                    if r['component'] == 'shout' and r['alpha'] == mid_alpha
                    and r.get('has_logit_data', False)]
    whisper_shifts = [r['mean_concept_shift'] for r in all_results
                      if r['component'] == 'whisper' and r['alpha'] == mid_alpha
                      and r.get('has_logit_data', False)]

    if shout_ppls and whisper_ppls:
        d_ppl = cohens_d(shout_ppls, whisper_ppls)
        diffs = [s - w for s, w in zip(shout_ppls, whisper_ppls)]
        diff_mean, diff_ci_lo, diff_ci_hi = bootstrap_ci(diffs)

        log_print(f"\n  At alpha={mid_alpha} (N={len(shout_ppls)} concepts):")
        log_print(f"    Shouting (top-{top_k}):   PPL +{np.mean(shout_ppls):.0f}%  "
              f"KL={np.mean(shout_kls):.4f}")
        log_print(f"    Whispering (bot-{top_k}): PPL +{np.mean(whisper_ppls):.0f}%  "
              f"KL={np.mean(whisper_kls):.4f}")
        log_print(f"    Difference: {diff_mean:+.1f}% [95% CI: {diff_ci_lo:+.1f}%, {diff_ci_hi:+.1f}%]")
        log_print(f"    Cohen's d: {d_ppl:+.3f}")

        if len(shout_ppls) > 2:
            t_ppl, p_ppl = stats.ttest_rel(shout_ppls, whisper_ppls)
            t_kl, p_kl = stats.ttest_rel(shout_kls, whisper_kls)
            log_print(f"    PPL p={p_ppl:.4f}, KL p={p_kl:.4f}")

            if p_ppl < 0.05 and np.mean(shout_ppls) > np.mean(whisper_ppls):
                log_print(f"\n  INTERFERENCE ASYMMETRY CONFIRMED (p={p_ppl:.4f}, d={d_ppl:+.2f}).")
                log_print(f"     High-variance injection causes significantly more PPL disruption.")
                if whisper_shifts and np.mean(whisper_shifts) > 0:
                    log_print(f"     AND the whispering component flips the concept (shift={np.mean(whisper_shifts):+.4f}).")
                    log_print(f"     -> Strong functional evidence for why concepts anti-concentrate.")
                else:
                    log_print(f"     Concept flip via whispering is weak/absent.")
            elif diff_ci_lo > 0:
                log_print(f"\n  PARTIAL ASYMMETRY: 95% CI for difference excludes zero.")
            else:
                log_print(f"\n  WARNING: No clear asymmetry at alpha={mid_alpha}. "
                      f"Check higher alpha values.")

    md_path = OUTPUT_DIR / f"hole4_v2_split_injection_{model_name}_summary.md"
    with open(md_path, "w") as f:
        f.write("```text\n")
        f.write("\n".join(summary_lines).strip() + "\n")
        f.write("```\n")
    print(f"\n  Summary saved to markdown: {md_path}")

    # ── Plot ──
    if all_results:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        comp_colors = {'full': '#333333', 'shout': '#E63946',
                      'whisper': '#457B9D', 'mid': '#2A9D8F'}
        comp_labels = {'full': 'Full vector', 'shout': f'Top-{top_k} (shouting)',
                      'whisper': f'Bottom-{top_k} (whispering)', 'mid': 'Middle'}

        # Panel 1: PPL increase with CIs
        ax = axes[0]
        for comp in ['full', 'shout', 'whisper', 'mid']:
            alphas_plot = []
            ppls_plot = []
            ci_los = []
            ci_his = []
            for alpha in sorted(args.alphas):
                vals = [r['ppl_increase_pct'] for r in all_results
                       if r['component'] == comp and r['alpha'] == alpha]
                if vals:
                    mean, lo, hi = bootstrap_ci(vals)
                    alphas_plot.append(alpha)
                    ppls_plot.append(mean)
                    ci_los.append(lo)
                    ci_his.append(hi)
            if alphas_plot:
                ax.plot(alphas_plot, ppls_plot, 'o-', color=comp_colors[comp], lw=2,
                       ms=8, label=comp_labels[comp])
                ax.fill_between(alphas_plot, ci_los, ci_his,
                               color=comp_colors[comp], alpha=0.12)

        ax.set_xlabel("Steering Strength (alpha)", fontsize=12)
        ax.set_ylabel("PPL Increase (%) on Held-Out Text", fontsize=12)
        ax.set_title(f"{model_name} — Grammatical Interference\n"
                     f"(Multi-layer steering, {len(heldout_passages)} held-out passages)",
                     fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='black', ls='-', lw=0.5)
        ax.set_xscale('log')

        # Panel 2: KL divergence
        ax = axes[1]
        for comp in ['full', 'shout', 'whisper', 'mid']:
            alphas_plot = []
            kls_plot = []
            for alpha in sorted(args.alphas):
                vals = [r['mean_kl_divergence'] for r in all_results
                       if r['component'] == comp and r['alpha'] == alpha]
                if vals:
                    alphas_plot.append(alpha)
                    kls_plot.append(np.mean(vals))
            if alphas_plot:
                ax.plot(alphas_plot, kls_plot, 'o-', color=comp_colors[comp], lw=2,
                       ms=8, label=comp_labels[comp])

        ax.set_xlabel("Steering Strength (alpha)", fontsize=12)
        ax.set_ylabel("Mean KL Divergence", fontsize=12)
        ax.set_title(f"{model_name} — KL Divergence\nby Spectral Component", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xscale('log')

        # Panel 3: Concept shift (for concepts with logit data)
        ax = axes[2]
        for comp in ['full', 'shout', 'whisper', 'mid']:
            alphas_plot = []
            shifts_plot = []
            for alpha in sorted(args.alphas):
                vals = [r['mean_concept_shift'] for r in all_results
                       if r['component'] == comp and r['alpha'] == alpha
                       and r.get('has_logit_data', False)]
                if vals:
                    alphas_plot.append(alpha)
                    shifts_plot.append(np.mean(vals))
            if alphas_plot:
                ax.plot(alphas_plot, shifts_plot, 'o-', color=comp_colors[comp], lw=2,
                       ms=8, label=comp_labels[comp])

        ax.set_xlabel("Steering Strength (alpha)", fontsize=12)
        ax.set_ylabel("Concept Shift (logit mass delta)", fontsize=12)
        ax.set_title(f"{model_name} — Concept Flip Effectiveness\nby Spectral Component",
                     fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='black', ls='-', lw=0.5)
        ax.set_xscale('log')

        plt.tight_layout()
        save_path = OUTPUT_DIR / f"hole4_v2_split_injection_{model_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved: {save_path}")

    # Save JSON (strip per-passage PPL lists for size)
    serializable = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k != 'ppl_increases_per_passage'}
        serializable.append(s)

    save_path = OUTPUT_DIR / f"hole4_v2_split_injection_{model_name}.json"
    with open(save_path, 'w') as f:
        json.dump({
            "model": args.model,
            "model_name": model_name,
            "top_k": top_k,
            "top_k_frac": args.top_k_frac,
            "alphas": args.alphas,
            "n_concepts": len(available),
            "concepts": available,
            "layer_indices_steer": layer_indices_steer,
            "n_layers": n_layers,
            "n_heldout_passages": len(heldout_passages),
            "decompositions": all_decomps,
            "results": serializable,
        }, f, indent=2, default=str)
    print(f"  Results saved: {save_path}")


if __name__ == "__main__":
    main()
