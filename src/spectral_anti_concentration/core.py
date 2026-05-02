import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import json
from tqdm import tqdm
import os
import transformers
from . import models

sns.set_theme(
    context="paper",
    style="white",
    palette="colorblind",
    font="sans-serif",
    font_scale=1.75,
)

def load_model(model_path, device_map="auto"):
    """
    Loads the model and tokenizer using model-specific handlers.
    Returns: model, tokenizer, handler
    """
    handler = models.get_handler(model_path)
    model, tokenizer = handler.load(device_map=device_map)
    return model, tokenizer, handler

def get_lm_head(model):
    """
    Safely retrieves the language model head (unembedding matrix) from the model.
    """
    if hasattr(model, "lm_head"):
        return model.lm_head
    elif hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    elif hasattr(model, "text_model") and hasattr(model.text_model, "lm_head"):
        return model.text_model.lm_head
    
    raise AttributeError(f"Could not find lm_head in model type: {type(model)}")

## get indices of counterfactual pairs
def get_counterfactual_pairs(filename, tokenizer):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # word1 \t word2
    words_pairs = [line.strip().split('\t') for line in lines if line.strip()]

    base_words = []
    target_words = []

    for i in range(len(words_pairs)):
        if len(words_pairs[i]) >= 2:
            base_words.append(words_pairs[i][0])
            target_words.append(words_pairs[i][1])
            
    return base_words, target_words

## get concept direction
def concept_direction(base_words, target_words, model, tokenizer, handler, device, batch_size=4):
    """
    Extracts the concept direction by mean-pooling over word embeddings.
    """
    # Use context-free extraction (the word itself) for simple direction
    base_data = get_embeddings(base_words, model, tokenizer, handler, device, batch_size=batch_size, target_words=base_words)
    target_data = get_embeddings(target_words, model, tokenizer, handler, device, batch_size=batch_size, target_words=target_words)

    diff_data = target_data - base_data
    mean_diff_data = torch.mean(diff_data, dim=0)
    norm = torch.norm(mean_diff_data)
    if norm > 0:
        mean_diff_data = mean_diff_data / norm

    return mean_diff_data, diff_data

## get embeddings of each text
def get_embeddings(text_batch, model, tokenizer, handler, device, batch_size=4, target_words=None):
    """
    Wrapper around handler's pooled embeddings to ensure consistency.
    """
    return handler.get_pooled_embeddings(
        model, tokenizer, text_batch, 
        target_words=target_words, 
        device=device, 
        batch_size=batch_size
    )

####### Experiment 1: subspace #######
def inner_product_loo(base_ind, target_ind, data):
    base_data = data[base_ind,]
    target_data = data[target_ind,]

    diff_data = target_data - base_data
    products = []
    for i in range(diff_data.shape[0]):
        mask = torch.ones(diff_data.shape[0], dtype=bool)
        mask[i] = False
        loo_diff = diff_data[mask]
        mean_diff_data = torch.mean(loo_diff, dim=0)
        norm = torch.norm(mean_diff_data)
        if norm > 0:
            loo_mean = mean_diff_data / norm
        else:
            loo_mean = mean_diff_data
        products.append(loo_mean @ diff_data[i])
    return torch.stack(products), diff_data

def show_histogram_LOO(inner_product_with_counterfactual_pairs_LOO,
                        random_pairs, concept, concept_names, save_path=None):
    fig, axs = plt.subplots(7, 4, figsize=(16, 20))
    axs = axs.flatten()

    for i in range(min(concept.shape[0], len(axs))):
        target = inner_product_with_counterfactual_pairs_LOO[i]
        baseline = random_pairs @ concept[i]

        axs[i].hist(baseline.detach().cpu().numpy(), bins=50, alpha=0.6, color='blue', label='random pairs', density=True)
        axs[i].hist(target.detach().cpu().numpy(), alpha=0.7, color='red', label='counterfactual pairs', density=True)
        axs[i].set_yticks([])
        axs[i].set_title(concept_names[i])

    handles, labels = axs[0].get_legend_handles_labels()
    if len(axs) > concept.shape[0]:
        axs[concept.shape[0]].legend(handles, labels, loc='center')
        axs[concept.shape[0]].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show() # Return figure/axis or save, don't show in CLI

####### Experiment 2: heatmap #######
def draw_heatmaps(data_matrices, concept_labels, cmap='PiYG', save_path=None):
    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 3, wspace=0.2)
    
    ticks = list(range(2, 27, 3))
    labels = [str(i+1) for i in ticks]
    ytick = list(range(len(concept_labels)))
    ims = []

    ax_left = plt.subplot(gs[0:2, 0:2])
    im = ax_left.imshow(data_matrices[0], cmap=cmap)
    ims.append(im)
    ax_left.set_xticks(ticks)
    ax_left.set_xticklabels(labels)
    ax_left.set_yticks(ytick)
    ax_left.set_yticklabels(concept_labels)
    ax_left.set_title(r'$M = \mathrm{Cov}(\gamma)^{-1}$')

    ax_top_right = plt.subplot(gs[0, 2])
    im = ax_top_right.imshow(data_matrices[1], cmap=cmap)
    ims.append(im)
    ax_top_right.set_xticks([])
    ax_top_right.set_yticks([])
    ax_top_right.set_title(r'$M = I_d$')

    ax_bottom_right = plt.subplot(gs[1, 2])
    im = ax_bottom_right.imshow(data_matrices[2], cmap=cmap)
    ims.append(im)
    ax_bottom_right.set_xticks([])
    ax_bottom_right.set_yticks([])
    ax_bottom_right.set_title(r'Random $M$')
    
    divider = make_axes_locatable(ax_left)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(ims[-1], cax=cax, orientation='vertical')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

####### Experiment 3: measurement #######
def get_lambda_pairs(filename, model, tokenizer, handler, device, num_eg=20):
    lambdas_0 = []
    lambdas_1 = []

    count = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading lambda pairs"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            if count >= num_eg:
                break

            text_0 = [s.strip(" " + data['word0']) for s in data['contexts0']]
            lambdas_0.append(get_embeddings(text_0, model, tokenizer, handler, device, target_words=[data['word0']]*len(text_0)))

            text_1 = [s.strip(" " + data['word1']) for s in data['contexts1']]
            lambdas_1.append(get_embeddings(text_1, model, tokenizer, handler, device, target_words=[data['word1']]*len(text_1)))
            
            count += 1

    if not lambdas_0:
        return torch.tensor([]), torch.tensor([])

    return torch.cat(lambdas_0), torch.cat(lambdas_1)

def hist_measurement(lambda_0, lambda_1, concept, concept_names,
                    base="English", target="French", alpha=0.5, save_path=None):
    n_concepts = concept.shape[0]
    n_cols = 4
    n_rows = (n_concepts + n_cols) // n_cols  # +n_cols for legend slot
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axs = axs.flatten()

    # Ensure concept is float32 for projection
    concept = concept.to(dtype=torch.float32)

    for i in range(min(n_concepts, len(axs))):
        # Cast activations to float32 to match concept vector
        W0 = lambda_0.to(dtype=torch.float32) @ concept[i]
        W1 = lambda_1.to(dtype=torch.float32) @ concept[i]

        axs[i].hist(W0.detach().cpu().numpy(), bins=25, alpha=alpha, label=base, density=True)
        axs[i].hist(W1.detach().cpu().numpy(), bins=25, alpha=alpha, label=target, density=True)
        axs[i].set_yticks([])
        axs[i].set_title(f'{concept_names[i]}', fontsize=10)

    handles, labels = axs[0].get_legend_handles_labels()
    if len(axs) > n_concepts:
        axs[n_concepts].legend(handles, labels, loc='center', fontsize=12)
        axs[n_concepts].axis('off')

    # Hide any remaining unused subplots
    for j in range(n_concepts + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

####### Experiment 4: intervention #######
def get_logit(embedding, unembedding, tokenizer, base="king", W="queen", Z="King"):
    num = embedding.shape[0]
    logit = torch.zeros(num, 2)
    
    # Encode words to get indices
    # Warning: simple encoding might assume words are single tokens
    def get_idx(w):
        ids = tokenizer.encode(w, add_special_tokens=False)
        return ids[0] if ids else 0 # Fallback

    index_base = get_idx(base)
    index_W = get_idx(W)
    index_Z = get_idx(Z)

    for i in range(num):
        value = unembedding @ embedding[i]
        logit[i, 0] = value[index_W] - value[index_base]
        logit[i, 1] = value[index_Z] - value[index_base]
    return logit

def show_arrows(logit_original, logit_intervened_l, concept_names,
                 base="king", W="queen", Z="King",
                 xlim=[-15, 5], ylim=[-15, 7], save_path=None):
    fig, axs = plt.subplots(7, 4, figsize=(16, 20))
    axs = axs.flatten()

    for i in range(min(len(concept_names), len(axs))):
        origin = logit_original.detach().numpy()
        vectors_A = logit_intervened_l[i].detach().numpy() - logit_original.detach().numpy()
        
        axs[i].quiver(*origin.T, vectors_A[:, 0], vectors_A[:, 1], color='b', angles='xy', scale_units='xy', scale=1, label='intervened lambda', alpha=1)
    
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].set_title(f'{concept_names[i]}')

    handles, labels = axs[0].get_legend_handles_labels()
    if len(axs) > len(concept_names):
        axs[len(concept_names)].legend(handles, labels, loc='center')
        axs[len(concept_names)].set_yticklabels([])
        axs[len(concept_names)].set_xticklabels([])

    plt.xlabel(rf"$\log\frac{{\mathbb{{P}}({W}\mid\lambda)}}{{\mathbb{{P}}({base}\mid \lambda)}}$")
    plt.ylabel(rf"$\log\frac{{\mathbb{{P}}({Z}\mid \lambda)}}{{\mathbb{{P}}({base}\mid \lambda)}}$")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

####### Metrics & Visualization #######

def plot_similarity_matrix(vectors, labels, title, save_path=None):
    """
    Plots a heatmap of cosine similarities between a list of vectors.
    """
    # Normalize vectors and ensure float32 for stable matmul
    vecs_stack = torch.stack(vectors).to(dtype=torch.float32)
    vecs_norm = torch.nn.functional.normalize(vecs_stack, p=2, dim=1)
    # Compute cosine similarity matrix
    sim_matrix = torch.mm(vecs_norm, vecs_norm.T).detach().cpu().numpy()
    
    n = len(labels)
    fig_size = max(12, n * 0.55)
    annot_size = max(6, 12 - n // 5)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=labels, yticklabels=labels, vmin=-1, vmax=1,
                annot_kws={"size": annot_size}, ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

####### Whitened Causal Alignment (WCA) #######

def compute_covariance_matrix(unembedding_matrix, token_indices=None):
    if token_indices is not None:
        subset = unembedding_matrix[token_indices]
    else:
        subset = unembedding_matrix

    # Sigma = E[x x^T] approx (X.T @ X) / N
    # Use float32 for stable computation
    second_moment = (subset.T @ subset.to(torch.float32)) / subset.shape[0]
    return second_moment

def get_whitening_transform(cov_matrix, reg_lambda=1e-4, truncate_k=None):
    """
    Computes whitening transform with regularization and stability checks.
    reg_lambda: Tikhonov regularization parameter.
    truncate_k: If set, only use top K eigenvalues.
    """
    # Ensure float32 for stable decomposition
    cov_matrix = cov_matrix.to(torch.float32)

    # Calculate condition number before regularization
    with torch.no_grad():
        vals_raw = torch.linalg.eigvalsh(cov_matrix)
        cond_number = vals_raw.max() / (vals_raw.min().clamp(min=1e-12))
        print(f"Condition Number of Covariance: {cond_number:.2e}")

    # Regularization
    cov_matrix = cov_matrix + reg_lambda * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
    vals, vecs = torch.linalg.eigh(cov_matrix)

    # Sort eigenvalues descending
    vals = torch.flip(vals, [0])
    vecs = torch.flip(vecs, [1])

    if truncate_k:
        vals = vals[:truncate_k]
        vecs = vecs[:, :truncate_k]
        print(f"Truncated to top {truncate_k} dimensions.")

    # inv_sqrt(Lambda)
    inv_sqrt_vals = 1.0 / torch.sqrt(torch.clamp(vals, min=1e-12))
    inv_sqrt_mat = vecs @ torch.diag(inv_sqrt_vals) @ vecs.T

    # sqrt(Lambda)
    sqrt_vals = torch.sqrt(torch.clamp(vals, min=0))
    sqrt_mat = vecs @ torch.diag(sqrt_vals) @ vecs.T

    return inv_sqrt_mat, sqrt_mat
def solve_whitened_procrustes(X_src, X_tgt, Psi_src, Psi_tgt):
    X_src_white = X_src @ Psi_src
    X_tgt_white = X_tgt @ Psi_tgt
    
    M = X_tgt_white.T @ X_src_white
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    Q = U @ Vh
    
    return Q

def transport_concept_vector(v_src, Psi_src, Sqrt_tgt, Q):
    # Ensure everything is in Float32 and on the same device for geometric transport
    dtype = torch.float32
    device = Psi_src.device
    
    v_src = v_src.to(device=device, dtype=dtype)
    Psi_src = Psi_src.to(dtype=dtype)
    Sqrt_tgt = Sqrt_tgt.to(dtype=dtype)
    Q = Q.to(dtype=dtype)
    
    # 1. Whiten Source
    v_platonic = Psi_src @ v_src
    
    # 2. Rotate to Target Space
    v_rot = Q @ v_platonic
    
    # 3. Color Target
    v_tgt = Sqrt_tgt @ v_rot
    
    return v_tgt

def solve_naive_procrustes(X_src, X_tgt):
    """
    Standard Procrustes alignment WITHOUT whitening.
    Baseline for comparison against Whitened Causal Alignment.
    """
    M = X_tgt.T @ X_src
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    Q = U @ Vh
    return Q

def transport_naive(v_src, Q):
    """
    Naive vector transport: simple rotation without whitening/recoloring.
    Baseline for comparison against WCA transport.
    """
    v_src = v_src.to(dtype=torch.float32)
    Q = Q.to(dtype=torch.float32)
    return Q @ v_src