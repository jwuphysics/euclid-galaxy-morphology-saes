"""
Generating figures for the Euclid Q1 SAE interpretability analysis.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Repository base path (parent of src/)
REPO_BASE = Path(__file__).parent.parent

# Set matplotlib parameters for consistent styling
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.grid'] = True


class MatryoshkaSAE(nn.Module):
    """
    Matryoshka Sparse Autoencoder that learns a sparse, hierarchical dictionary
    of features from input embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        group_sizes: List[int],
        top_k: int,
        l1_coeff: float = 1e-3,
        aux_penalty: float = 1e-2,
        n_batches_to_dead: int = 20,
        aux_k_multiplier: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.group_sizes = group_sizes
        self.total_dict_size = sum(group_sizes)
        self.top_k = top_k
        self.l1_coeff = l1_coeff
        self.aux_penalty = aux_penalty
        self.n_batches_to_dead = n_batches_to_dead
        self.aux_k_multiplier = aux_k_multiplier

        self.b_dec = nn.Parameter(torch.zeros(self.input_dim))
        self.b_enc = nn.Parameter(torch.zeros(self.total_dict_size))

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.input_dim, self.total_dict_size))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.total_dict_size, self.input_dim))
        )

        with torch.no_grad():
            self.W_dec.data = self.W_enc.t().clone()
            self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True)

        self.register_buffer(
            "num_batches_not_active", torch.zeros(self.total_dict_size, dtype=torch.long)
        )
        
        self.group_indices = [0] + list(np.cumsum(self.group_sizes))


def compute_sae_activations(model: MatryoshkaSAE, embeddings, batch_size=1024, device='cpu'):
    """Compute dense SAE activations for all embeddings using MatryoshkaSAE"""
    model = model.to(device)
    dense_activations = []
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size].to(device)
            
            # Compute activations using MatryoshkaSAE forward pass components
            x_cent = batch - model.b_dec
            pre_acts = x_cent @ model.W_enc + model.b_enc
            dense_acts = torch.nn.functional.relu(pre_acts)
            
            dense_activations.append(dense_acts.cpu())
    
    return torch.cat(dense_activations)


def get_top_sae_features(dense_activations, n_features=64, min_freq=0.01):
    """Get top SAE features by activation frequency"""
    frequencies = (dense_activations > 0).float().mean(dim=0)
    active_features = torch.where(frequencies > min_freq)[0]
    sorted_idxs = torch.argsort(frequencies[active_features], descending=True)
    return active_features[sorted_idxs][:n_features], frequencies


def compute_max_correlations(features, gz_features, feature_names):
    """Compute maximum Spearman correlation between each feature and any GZ feature"""
    max_correlations = []
    
    for i in range(features.shape[1]):
        feature_vals = features[:, i]
        correlations = []
        
        for j in range(gz_features.shape[1]):
            gz_vals = gz_features[:, j]
            # Skip if either feature has no variation
            if np.std(feature_vals) > 0 and np.std(gz_vals) > 0:
                corr, _ = spearmanr(feature_vals, gz_vals)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        max_correlations.append(max(correlations) if correlations else 0)
    
    return np.array(max_correlations)


def compute_novelty_via_regression(features, gz_features):
    """Compute novelty scores using linear regression R² (how much each feature is NOT explained by GZ features)"""
    # Handle NaNs in GZ features by imputing with median values
    imputer = SimpleImputer(strategy='median')
    gz_features_clean = imputer.fit_transform(gz_features)
    
    novelty_scores = []
    
    for i in range(features.shape[1]):
        try:
            feature_i = features[:, i]
            
            # Skip if feature has no variation
            if np.std(feature_i) < 1e-10:
                novelty_scores.append(1.0)  # Maximum novelty for constant features
                continue
            
            # Check for NaNs in the current feature
            if np.isnan(feature_i).any():
                novelty_scores.append(1.0)  # Default maximum novelty
                continue
            
            # Fit linear regression to predict this feature from GZ features
            reg = LinearRegression()
            reg.fit(gz_features_clean, feature_i)
            predictions = reg.predict(gz_features_clean)
            
            # Compute R²
            r2 = r2_score(feature_i, predictions)
            r2 = max(0, r2)  # Clip negative R² to 0
            
            # Novelty is 1 - R² (how much is NOT explained)
            novelty = 1 - r2
            novelty_scores.append(novelty)
            
        except Exception as e:
            novelty_scores.append(1.0)  # Default to maximum novelty on error
    
    return np.array(novelty_scores)


def cache_correlation_computations(cache_path: Path, sae_activations, valid_indices, gz_features, top_sae_features_, group_size_bin_edges, mode='supervised'):
    """Cache expensive correlation computations for SAE groups"""
    
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    cached_correlations = []
    
    for g1, g2 in zip(group_size_bin_edges[:-1], group_size_bin_edges[1:]):
        sae_features_g = sae_activations[valid_indices][:, top_sae_features_[g1:g2]].numpy()
        sae_max_corrs_g = compute_max_correlations(
            sae_features_g, gz_features, 
            [f"SAE_{i}" for i in top_sae_features_[g1:g2]]
        )
        cached_correlations.append(sae_max_corrs_g)
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_correlations, f)
    
    return cached_correlations


def cache_novelty_computations(cache_path: Path, sae_activations, valid_indices, gz_features, top_sae_features_, group_size_bin_edges):
    """Cache expensive novelty computations for SAE groups"""
    
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    cached_novelty = []
    
    for g1, g2 in zip(group_size_bin_edges[:-1], group_size_bin_edges[1:]):
        sae_features_g = sae_activations[valid_indices][:, top_sae_features_[g1:g2]].numpy()
        sae_novelty_g = compute_novelty_via_regression(sae_features_g, gz_features)
        cached_novelty.append(sae_novelty_g)
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_novelty, f)
    
    return cached_novelty


def load_supervised_data():
    """Load data for supervised analysis"""
    
    # Load supervised embeddings and IDs
    test_embeddings = torch.from_numpy(np.load(REPO_BASE / 'results/euclid_test_embeddings.npy').astype(np.float32))
    test_ids = np.load(REPO_BASE / 'results/euclid_test_ids.npy', allow_pickle=True)
    
    # Load MatryoshkaSAE model
    model_config = {
        'input_dim': 640,
        'group_sizes': [64, 64, 128, 256, 512, 1024],
        'top_k': 64,
        'l1_coeff': 0.001,
        'aux_penalty': 0.03,
        'n_batches_to_dead': 256,
        'aux_k_multiplier': 16,
    }
    
    sae_model = MatryoshkaSAE(**model_config)
    
    # Load the trained weights
    checkpoint_path = REPO_BASE / 'results/matryoshka_sae/weights/matryoshka_sae_final.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sae_model.load_state_dict(checkpoint)
    sae_model.eval()
    
    # Load PCA components
    pca_transformed = np.load(REPO_BASE / 'results/pca/pca_transformed_data.npy')
    
    # Load Galaxy Zoo dataset
    gz_dataset = load_dataset("mwalmsley/gz_euclid", split="test")
    
    return test_embeddings, test_ids, sae_model, pca_transformed, gz_dataset


def load_ssl_data():
    """Load data for SSL analysis"""
    
    # Load MAE test embeddings and IDs
    test_embeddings = torch.from_numpy(np.load(REPO_BASE / 'results/mae_analysis/mae_test_embeddings.npy').astype(np.float32))
    test_ids = np.load(REPO_BASE / 'results/mae_analysis/mae_test_ids.npy', allow_pickle=True)
    
    # Load MAE MatryoshkaSAE model
    model_config = {
        'input_dim': 384,  # MAE embeddings are 384-dimensional  
        'group_sizes': [64, 64, 128, 256, 512, 1024],
        'top_k': 64,
        'l1_coeff': 0.001,
        'aux_penalty': 0.03,
        'n_batches_to_dead': 256,
        'aux_k_multiplier': 16,
    }
    
    sae_model = MatryoshkaSAE(**model_config)
    
    # Load the trained weights for MAE SAE
    checkpoint_path = REPO_BASE / 'results/mae_analysis/matryoshka_sae/weights/matryoshka_sae_final.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sae_model.load_state_dict(checkpoint)
    sae_model.eval()
    
    # Load MAE PCA components
    pca_transformed = np.load(REPO_BASE / 'results/mae_analysis/pca/pca_transformed_data.npy')
    
    # Load local morphology catalog
    morphology_df = pd.read_parquet(REPO_BASE / 'data/morphology_catalogue.parquet')
    # Apply ID string correction: replace '-' with 'NEG' for crossmatching with MAE embeddings
    morphology_df['id_str'] = morphology_df['id_str'].apply(lambda x: x.replace('-', 'NEG'))
    
    # Load Huggingface dataset for images
    gz_dataset = load_dataset("mwalmsley/euclid_q1", "v1-gz_arcsinh_vis_y", split="test")
    
    return test_embeddings, test_ids, sae_model, pca_transformed, morphology_df, gz_dataset


def extract_supervised_gz_features(gz_dataset, test_ids):
    """Extract Galaxy Zoo features for supervised test samples"""
    # Create ID mapping
    id_to_idx = {item['id_str']: idx for idx, item in enumerate(gz_dataset)}
    
    # Get only the _fraction Galaxy Zoo feature columns
    sample_item = gz_dataset[0]
    gz_feature_cols = [col for col in sample_item.keys() if col.endswith('_fraction')]
    
    # Extract features for test samples
    gz_features = []
    valid_indices = []
    
    for i, id_str in enumerate(test_ids):
        if id_str in id_to_idx:
            dataset_idx = id_to_idx[id_str]
            item = gz_dataset[dataset_idx]
            feature_vals = [item[col] for col in gz_feature_cols]
            gz_features.append(feature_vals)
            valid_indices.append(i)
    
    return np.array(gz_features), valid_indices, gz_feature_cols


def extract_ssl_gz_features(morphology_df, test_ids):
    """Extract morphology features for SSL test samples from local catalog"""
    # Create ID mapping
    morph_id_to_idx = {row['id_str']: idx for idx, row in morphology_df.iterrows()}
    
    # Get all morphology feature columns (those ending with '_fraction')
    gz_feature_cols = [col for col in morphology_df.columns if col.endswith('_fraction')]
    
    # Extract features for test samples
    gz_features = []
    valid_indices = []
    
    for i, id_str in enumerate(test_ids):
        if id_str in morph_id_to_idx:
            df_idx = morph_id_to_idx[id_str]
            row = morphology_df.iloc[df_idx]
            feature_vals = [row[col] for col in gz_feature_cols]
            gz_features.append(feature_vals)
            valid_indices.append(i)
    
    return np.array(gz_features), valid_indices, gz_feature_cols


def make_supervised_correlation_figure():
    """Generate supervised_feature_comparison_corr.pdf"""
    
    # Load data
    test_embeddings, test_ids, sae_model, pca_transformed, gz_dataset = load_supervised_data()
    
    # Compute SAE activations and get top features
    sae_activations = compute_sae_activations(sae_model, test_embeddings)
    top_sae_features, _ = get_top_sae_features(sae_activations)
    
    # Extract Galaxy Zoo features
    gz_features, valid_indices, _ = extract_supervised_gz_features(gz_dataset, test_ids)
    
    # Compute correlations for top 64 features
    sae_features_subset = sae_activations[valid_indices][:, top_sae_features[:64]].numpy()
    sae_max_corrs = compute_max_correlations(sae_features_subset, gz_features, 
                                           [f"SAE_{i}" for i in top_sae_features[:64]])
    
    pca_features_subset = pca_transformed[valid_indices][:, :64]
    pca_max_corrs = compute_max_correlations(pca_features_subset, gz_features, 
                                           [f"PC_{i}" for i in range(64)])
    
    # Create figure
    plt.style.use("default")
    sns.set_style("white")
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=300)
    
    colors = {"PCA": "#003f5c", "SAE": "#ff6361"}
    
    sns.kdeplot(pca_max_corrs, ax=ax, color=colors["PCA"], lw=2, bw_adjust=0.7, label="PCA")
    sns.kdeplot(sae_max_corrs, ax=ax, color=colors["SAE"], lw=2, bw_adjust=0.7, label="SAE")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Max Spearman Correlation with GZ")
    ax.set_ylabel("Density")
    
    ax.annotate(
        "More aligned with GZ",
        xy=(0.95, 0.9),
        xytext=(0.15, 0.9),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left", va="center",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )
    
    ax.text(0.25, 0.8, "PCA", color=colors["PCA"], fontsize=16, ha="right",
             path_effects=[pe.withStroke(linewidth=10, foreground="white")])
    ax.text(0.50, 0.8, "SAE", color=colors["SAE"], fontsize=16, ha="right",
             path_effects=[pe.withStroke(linewidth=10, foreground="white")])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    
    fig.suptitle("Top 64 features (supervised)", fontsize=16)
    
    plt.tight_layout()
    output_path = REPO_BASE / 'results/figures/supervised_feature_comparison_corr.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def make_supervised_correlation_all_figure():
    """Generate supervised_feature_comparison_corr_all.pdf"""
    
    # Load data
    test_embeddings, test_ids, sae_model, pca_transformed, gz_dataset = load_supervised_data()
    
    # Compute SAE activations and get all top features
    sae_activations = compute_sae_activations(sae_model, test_embeddings)
    top_sae_features_, _ = get_top_sae_features(sae_activations, n_features=sae_activations.shape[1])
    
    # Extract Galaxy Zoo features
    gz_features, valid_indices, _ = extract_supervised_gz_features(gz_dataset, test_ids)
    
    # Use caching for expensive correlation computations
    group_size_bin_edges = [0, 64, 128, 256, 512, 1024, 2048]
    cache_path = REPO_BASE / 'results/.cache/supervised_correlations_all.pkl'
    cached_correlations = cache_correlation_computations(
        cache_path, sae_activations, valid_indices, gz_features, 
        top_sae_features_, group_size_bin_edges, 'supervised'
    )
    
    # Compute PCA correlations for first group (0-64)
    pca_features_g = pca_transformed[valid_indices][:, 0:64]
    pca_max_corrs_g = compute_max_correlations(pca_features_g, gz_features, 
                                              [f"PC_{i}" for i in range(0, 64)])
    
    # Create figure
    plt.style.use("default")
    sns.set_style("white")
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=300)
    
    colors = {"PCA": "#003f5c", "SAE": "#ff6361"}
    cs = ["#ed5c5a", "#da5453", "#c54c4b", "#ad4341", "#903836", "#692828"]
    
    # Plot PCA
    sns.kdeplot(pca_max_corrs_g, ax=ax, color="white", lw=2, bw_adjust=0.7)
    sns.kdeplot(pca_max_corrs_g, ax=ax, color=colors["PCA"], ls="--", lw=0.5, bw_adjust=0.7)
    
    # Plot SAE groups
    for z, (c, sae_max_corrs_g, g1, g2) in enumerate(zip(cs, cached_correlations, group_size_bin_edges[:-1], group_size_bin_edges[1:])):
        sns.kdeplot(sae_max_corrs_g, ax=ax, color=c, lw=2, bw_adjust=0.7)
        ax.text(0.42, 6.2-0.8*z, f"{g1}$-${g2}", color=c, fontsize=12, ha="left")
    
    ax.set_xlim(0, 1)
    ax.set_xlabel("Max Spearman Correlation with GZ")
    ax.set_ylabel("Density")
    
    ax.annotate(
        "More aligned with GZ",
        xy=(0.95, 0.9),
        xytext=(0.15, 0.9),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left", va="center",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )
    
    ax.text(0.42, 7, "SAE Activation Groups", color=colors["SAE"], fontsize=12, ha="left")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    
    fig.suptitle("All features (supervised)", fontsize=16)
    
    plt.tight_layout()
    output_path = REPO_BASE / 'results/figures/supervised_feature_comparison_corr_all.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def make_supervised_novelty_all_figure():
    """Generate supervised_feature_comparison_novelty_all.pdf"""
    
    # Load data
    test_embeddings, test_ids, sae_model, pca_transformed, gz_dataset = load_supervised_data()
    
    # Compute SAE activations and get all top features
    sae_activations = compute_sae_activations(sae_model, test_embeddings)
    top_sae_features_, _ = get_top_sae_features(sae_activations, n_features=sae_activations.shape[1])
    
    # Extract Galaxy Zoo features
    gz_features, valid_indices, _ = extract_supervised_gz_features(gz_dataset, test_ids)
    
    # Use caching for expensive novelty computations
    group_size_bin_edges = [0, 64, 128, 256, 512, 1024, 2048]
    cache_path = REPO_BASE / 'results/.cache/supervised_novelty_all.pkl'
    cached_novelty = cache_novelty_computations(
        cache_path, sae_activations, valid_indices, gz_features, 
        top_sae_features_, group_size_bin_edges
    )
    
    # Compute PCA novelty for first group (0-64)
    pca_features_g = pca_transformed[valid_indices][:, 0:64]
    pca_novelty_g = compute_novelty_via_regression(pca_features_g, gz_features)
    
    # Create figure
    plt.style.use("default")
    sns.set_style("white")
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=300)
    
    colors = {"PCA": "#003f5c", "SAE": "#ff6361"}
    cs = ["#ed5c5a", "#da5453", "#c54c4b", "#ad4341", "#903836", "#692828"]
    
    # Plot PCA
    sns.kdeplot(pca_novelty_g, ax=ax, color="white", lw=2, bw_adjust=0.7)
    sns.kdeplot(pca_novelty_g, ax=ax, color=colors["PCA"], ls="--", lw=0.5, bw_adjust=0.7)
    
    # Plot SAE groups
    for z, (c, sae_novelty_g, g1, g2) in enumerate(zip(cs, cached_novelty, group_size_bin_edges[:-1], group_size_bin_edges[1:])):
        sns.kdeplot(sae_novelty_g, ax=ax, color=c, lw=2, bw_adjust=0.7)
        ax.text(0.09, 10.2-1.4*z, f"{g1}$-${g2}", color=c, fontsize=12, ha="left")
    
    ax.set_xlim(0, 1)
    ax.set_xlabel("Unexplained GZ Variance ($1-R^2$)")
    ax.set_ylabel("Density")
    
    ax.annotate(
        "Less predictable features",
        xy=(0.9, 0.9),
        xytext=(0.05, 0.9),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left", va="center",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )
    
    ax.text(0.09, 11.6, "SAE Activation Groups", color=colors["SAE"], fontsize=12, ha="left")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    
    fig.suptitle("All features (supervised)", fontsize=16)
    
    plt.tight_layout()
    output_path = REPO_BASE / 'results/figures/supervised_feature_comparison_novelty_all.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def make_ssl_correlation_figure():
    """Generate ssl_feature_comparison_corr.pdf"""
    
    # Load data
    test_embeddings, test_ids, sae_model, pca_transformed, morphology_df, gz_dataset = load_ssl_data()
    
    # Compute SAE activations and get top features
    sae_activations = compute_sae_activations(sae_model, test_embeddings)
    top_sae_features, _ = get_top_sae_features(sae_activations)
    
    # Extract morphology features
    gz_features, valid_indices, _ = extract_ssl_gz_features(morphology_df, test_ids)
    
    # Compute correlations for top 64 features
    sae_features_subset = sae_activations[valid_indices][:, top_sae_features[:64]].numpy()
    sae_max_corrs = compute_max_correlations(sae_features_subset, gz_features, 
                                           [f"MAE_SAE_{i}" for i in top_sae_features[:64]])
    
    pca_features_subset = pca_transformed[valid_indices][:, :64]
    pca_max_corrs = compute_max_correlations(pca_features_subset, gz_features, 
                                           [f"MAE_PC_{i}" for i in range(64)])
    
    # Create figure
    plt.style.use("default")
    sns.set_style("white")
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=300)
    
    colors = {"PCA": "#003f5c", "SAE": "#ff6361"}
    
    sns.kdeplot(pca_max_corrs, ax=ax, color=colors["PCA"], lw=2, bw_adjust=0.7, label="PCA")
    sns.kdeplot(sae_max_corrs, ax=ax, color=colors["SAE"], lw=2, bw_adjust=0.7, label="SAE")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Max Spearman Correlation with GZ")
    ax.set_ylabel("Density")
    
    ax.text(0.3, 0.8, "PCA", color=colors["PCA"], fontsize=16, ha="right",
             path_effects=[pe.withStroke(linewidth=5, foreground="white")])
    ax.text(0.63, 0.8, "SAE", color=colors["SAE"], fontsize=16, ha="right",
             path_effects=[pe.withStroke(linewidth=10, foreground="white")])
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    
    fig.suptitle("(self-supervised)", fontsize=16)
    
    plt.tight_layout()
    output_path = REPO_BASE / 'results/figures/ssl_feature_comparison_corr.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def make_ssl_correlation_all_figure():
    """Generate ssl_feature_comparison_corr_all.pdf"""
    
    # Load data
    test_embeddings, test_ids, sae_model, pca_transformed, morphology_df, gz_dataset = load_ssl_data()
    
    # Compute SAE activations and get all top features
    sae_activations = compute_sae_activations(sae_model, test_embeddings)
    top_sae_features_, _ = get_top_sae_features(sae_activations, n_features=sae_activations.shape[1])
    
    # Extract morphology features
    gz_features, valid_indices, _ = extract_ssl_gz_features(morphology_df, test_ids)
    
    # Use caching for expensive correlation computations
    group_size_bin_edges = [0, 64, 128, 256, 512, 1024, 2048]
    cache_path = REPO_BASE / 'results/.cache/ssl_correlations_all.pkl'
    cached_correlations = cache_correlation_computations(
        cache_path, sae_activations, valid_indices, gz_features, 
        top_sae_features_, group_size_bin_edges, 'ssl'
    )
    
    # Compute PCA correlations for first group (0-64)
    pca_features_g = pca_transformed[valid_indices][:, 0:64]
    pca_max_corrs_g = compute_max_correlations(pca_features_g, gz_features, 
                                              [f"PC_{i}" for i in range(0, 64)])
    
    # Create figure
    plt.style.use("default")
    sns.set_style("white")
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=300)
    
    colors = {"PCA": "#003f5c", "SAE": "#ff6361"}
    cs = ["#ed5c5a", "#da5453", "#c54c4b", "#ad4341", "#903836", "#692828"]
    
    # Plot PCA
    sns.kdeplot(pca_max_corrs_g, ax=ax, color="white", lw=2, bw_adjust=0.7)
    sns.kdeplot(pca_max_corrs_g, ax=ax, color=colors["PCA"], ls="--", lw=0.5, bw_adjust=0.7)
    
    # Plot SAE groups
    for z, (c, sae_max_corrs_g) in enumerate(zip(cs, cached_correlations)):
        sns.kdeplot(sae_max_corrs_g, ax=ax, color=c, lw=2, bw_adjust=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel("Max Spearman Correlation with GZ")
    ax.set_ylabel("Density")
    
    ax.annotate(
        "More aligned with GZ",
        xy=(0.95, 0.9),
        xytext=(0.15, 0.9),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left", va="center",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    
    fig.suptitle("(self-supervised)", fontsize=16)
    
    plt.tight_layout()
    output_path = REPO_BASE / 'results/figures/ssl_feature_comparison_corr_all.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def make_ssl_novelty_all_figure():
    """Generate ssl_feature_comparison_novelty_all.pdf"""
    
    # Load data
    test_embeddings, test_ids, sae_model, pca_transformed, morphology_df, gz_dataset = load_ssl_data()
    
    # Compute SAE activations and get all top features
    sae_activations = compute_sae_activations(sae_model, test_embeddings)
    top_sae_features_, _ = get_top_sae_features(sae_activations, n_features=sae_activations.shape[1])
    
    # Extract morphology features
    gz_features, valid_indices, _ = extract_ssl_gz_features(morphology_df, test_ids)
    
    # Use caching for expensive novelty computations
    group_size_bin_edges = [0, 64, 128, 256, 512, 1024, 2048]
    cache_path = REPO_BASE / 'results/.cache/ssl_novelty_all.pkl'
    cached_novelty = cache_novelty_computations(
        cache_path, sae_activations, valid_indices, gz_features, 
        top_sae_features_, group_size_bin_edges
    )
    
    # Compute PCA novelty for first group (0-64)
    pca_features_g = pca_transformed[valid_indices][:, 0:64]
    pca_novelty_g = compute_novelty_via_regression(pca_features_g, gz_features)
    
    # Create figure
    plt.style.use("default")
    sns.set_style("white")
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=300)
    
    colors = {"PCA": "#003f5c", "SAE": "#ff6361"}
    cs = ["#ed5c5a", "#da5453", "#c54c4b", "#ad4341", "#903836", "#692828"]
    
    # Plot PCA
    sns.kdeplot(pca_novelty_g, ax=ax, color="white", lw=2, bw_adjust=0.7)
    sns.kdeplot(pca_novelty_g, ax=ax, color=colors["PCA"], ls="--", lw=0.5, bw_adjust=0.7)
    
    # Plot SAE groups
    for z, (c, sae_novelty_g) in enumerate(zip(cs, cached_novelty)):
        sns.kdeplot(sae_novelty_g, ax=ax, color=c, lw=2, bw_adjust=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel("Unexplained GZ Variance ($1-R^2$)")
    ax.set_ylabel("Density")
    
    ax.annotate(
        "Less predictable features",
        xy=(0.9, 0.9),
        xytext=(0.05, 0.9),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left", va="center",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    
    fig.suptitle("(self-supervised)", fontsize=16)
    
    plt.tight_layout()
    output_path = REPO_BASE / 'results/figures/ssl_feature_comparison_novelty_all.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def make_feature_gallery(feature_type, mode, feature_id, reverse=False, prefix=None):
    """
    Create a single-row gallery of 10 galaxy images for a specific feature.
    
    Args:
        feature_type: 'sae' or 'pca'
        mode: 'supervised' or 'ssl'  
        feature_id: The specific feature number/ID
        reverse: If True, show images with lowest activations (for PCA negative direction)
        prefix: Optional string prefix for the output filename (e.g., "00", "01")
    
    Saves to: results/figures/gallery/{prefix}-{mode}-{feature_type}-{feature_id}.pdf
    """
    
    if mode == 'supervised':
        # Load supervised data
        test_embeddings, test_ids, sae_model, pca_transformed, gz_dataset = load_supervised_data()
        
        # Create ID mapping for images
        id_to_idx = {item['id_str']: idx for idx, item in enumerate(gz_dataset)}
        
        if feature_type == 'sae':
            # Compute SAE activations
            sae_activations = compute_sae_activations(sae_model, test_embeddings)
            feature_activations = sae_activations[:, feature_id].numpy()
        elif feature_type == 'pca':
            feature_activations = pca_transformed[:, feature_id]
        else:
            raise ValueError("feature_type must be 'sae' or 'pca'")
            
    elif mode == 'ssl':
        # Load SSL data
        test_embeddings, test_ids, sae_model, pca_transformed, morphology_df, gz_dataset = load_ssl_data()
        
        # Create ID mapping for images
        gz_id_to_idx = {item['id_str']: idx for idx, item in enumerate(gz_dataset)}
        
        if feature_type == 'sae':
            # Compute SAE activations
            sae_activations = compute_sae_activations(sae_model, test_embeddings)
            feature_activations = sae_activations[:, feature_id].numpy()
        elif feature_type == 'pca':
            feature_activations = pca_transformed[:, feature_id]
        else:
            raise ValueError("feature_type must be 'sae' or 'pca'")
            
    else:
        raise ValueError("mode must be 'supervised' or 'ssl'")
    
    # Get activations for this feature
    if reverse:
        # Show most negative activations (lowest values) - first 10 from sorted array
        top_indices = np.argsort(feature_activations)[:10]
    else:
        # Show most positive activations (highest values) - last 10 from sorted array, reversed
        top_indices = np.argsort(feature_activations)[-10:][::-1]
    
    # Create figure with single row of images
    fig, axes = plt.subplots(1, 10, figsize=(20, 2), dpi=300)
    
    for i, ax in enumerate(axes):
        if i < len(top_indices):
            sample_idx = top_indices[i]
            id_str = test_ids[sample_idx]
            
            try:
                # Get image based on mode
                if mode == 'supervised':
                    if id_str in id_to_idx:
                        dataset_idx = id_to_idx[id_str]
                        img = gz_dataset[dataset_idx]['image']
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)
                        ax.imshow(img)
                    else:
                        # Show blank if image not found
                        ax.set_facecolor('lightgray')
                else:  # SSL mode
                    if id_str in gz_id_to_idx:
                        dataset_idx = gz_id_to_idx[id_str]
                        img = gz_dataset[dataset_idx]['image']
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img = img.resize((224, 224), Image.Resampling.LANCZOS)
                        ax.imshow(img)
                    else:
                        # Show blank if image not found
                        ax.set_facecolor('lightgray')
                        
            except Exception:
                # Show blank on any error
                ax.set_facecolor('lightgray')
        else:
            # Show blank for missing images
            ax.set_facecolor('lightgray')
            
        # Remove all axes, ticks, and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Set minimal spacing between images
    plt.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)
    
    # Create output directory and save
    output_dir = REPO_BASE / 'results/figures/gallery'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add suffix for reverse galleries
    suffix = "-rev" if reverse else ""
    
    # Add prefix to filename if provided
    if prefix:
        filename = f"{prefix}-{mode}-{feature_type}-{feature_id}{suffix}.pdf"
    else:
        filename = f"{mode}-{feature_type}-{feature_id}{suffix}.pdf"
    
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


if __name__ == "__main__":
    """Generate all figures"""
    
    print("Generating supervised figures...")
    make_supervised_correlation_figure()
    make_supervised_correlation_all_figure()
    make_supervised_novelty_all_figure()
    
    print("Generating SSL figures...")
    make_ssl_correlation_figure()
    make_ssl_correlation_all_figure()
    make_ssl_novelty_all_figure()
    
    print("All figures generated successfully!")