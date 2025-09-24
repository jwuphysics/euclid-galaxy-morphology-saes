"""
Visualization utilities for Euclid Q1 SAE analysis.

This module provides common visualization functions used across different
analysis scripts for plotting feature activations and galaxy images.
"""

import io
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def plot_feature_examples(
    feature_idx: int,
    activations: torch.Tensor,
    hf_dataset,
    embedding_ids: List[str],
    n_examples: int = 20,
    save_path: Optional[Path] = None,
    title_prefix: str = "Feature",
    id_to_idx_mapping: Optional[dict] = None
):
    """
    Plot galaxy images that most strongly activate a specific feature.
    
    Args:
        feature_idx: Index of the feature to visualize
        activations: Feature activations tensor [n_samples, n_features]
        hf_dataset: HuggingFace dataset containing galaxy images
        embedding_ids: List of embedding IDs corresponding to activation rows
        n_examples: Number of example images to show
        save_path: Path to save the figure (optional)
        title_prefix: Prefix for the plot title
        id_to_idx_mapping: Pre-computed mapping from ID strings to dataset indices
    """
    # Get activations for this feature across all samples
    feature_acts = activations[:, feature_idx]
    
    # Find top activating examples
    top_acts, top_indices = torch.topk(feature_acts, k=min(n_examples, len(feature_acts)))
    
    # Calculate rows and columns for subplot grid
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols  # Ceiling division
    
    # Create id_str to index mapping for quick lookup if not provided
    if id_to_idx_mapping is None:
        id_to_idx_mapping = {item['id_str']: idx for idx, item in enumerate(hf_dataset)}
    
    # Create figure
    fig = plt.figure(figsize=(20, 4 * n_rows), dpi=150)
    fig.suptitle(f'{title_prefix} {feature_idx}', fontsize=16)
    
    for i in range(len(top_acts)):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        act_val = top_acts[i].item()
        sample_idx = top_indices[i].item()
        id_str = embedding_ids[sample_idx]
        
        try:
            # Find the corresponding image in HF dataset
            if id_str in id_to_idx_mapping:
                dataset_idx = id_to_idx_mapping[id_str]
                img = hf_dataset[dataset_idx]['image']
                
                # Ensure image is RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Resize and display
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'val: {act_val:.2f}')
            else:
                ax.text(0.5, 0.5, f"ID not found:\n{id_str}", 
                       ha='center', va='center')
                ax.axis('off')
                
        except Exception as e:
            print(f"Error loading image {id_str}: {e}")
            ax.text(0.5, 0.5, f"Error loading\n{id_str}", 
                   ha='center', va='center')
            ax.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.close()


def plot_pca_component_examples(
    component_idx: int,
    top_indices: np.ndarray,
    top_values: np.ndarray,
    hf_dataset,
    embedding_ids: np.ndarray,
    id_to_idx_mapping: dict,
    n_examples: int = 30,
    save_path: Optional[Path] = None,
    explained_variance_ratio: Optional[float] = None
):
    """
    Plot galaxy images with highest absolute PCA component values.
    
    Args:
        component_idx: Index of the PCA component
        top_indices: Pre-computed indices of top activating samples
        top_values: Pre-computed activation values for top samples
        hf_dataset: HuggingFace dataset containing galaxy images
        embedding_ids: Array of embedding IDs
        id_to_idx_mapping: Mapping from ID strings to dataset indices
        n_examples: Number of example images to show
        save_path: Path to save the figure (optional)
        explained_variance_ratio: Explained variance ratio for this component
    """
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 4 * n_rows), dpi=150)
    
    title = f'PCA Component {component_idx}'
    if explained_variance_ratio is not None:
        title += f' (EV: {explained_variance_ratio:.3f})'
    fig.suptitle(title, fontsize=16)
    
    for i, (img_idx, val) in enumerate(zip(top_indices, top_values)):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        id_str = embedding_ids[img_idx]
        
        try:
            # Find the corresponding image in HF dataset
            if id_str in id_to_idx_mapping:
                dataset_idx = id_to_idx_mapping[id_str]
                img = hf_dataset[dataset_idx]['image']
                
                # Ensure image is RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Resize and display
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'val: {val:.2f}')
            else:
                ax.text(0.5, 0.5, f"ID not found:\n{id_str}", 
                       ha='center', va='center')
                ax.axis('off')
                
        except Exception as e:
            print(f"Error loading image {id_str}: {e}")
            ax.text(0.5, 0.5, f"Error loading\n{id_str}", 
                   ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_mae_feature_examples(
    feature_idx: int, 
    activations: torch.Tensor,
    test_dataset,  # MAEEmbeddingDataset
    n_examples: int = 30, 
    save_path: Optional[Path] = None
):
    """
    Plot galaxy images for MAE feature activations.
    
    Args:
        feature_idx: Index of the feature to visualize
        activations: Feature activations tensor
        test_dataset: MAEEmbeddingDataset containing crossmatched data
        n_examples: Number of example images to show
        save_path: Path to save the figure (optional)
    """
    feature_acts = activations[:, feature_idx]
    top_acts, top_indices = torch.topk(feature_acts, k=min(n_examples, len(feature_acts)))
    
    # Get corresponding GZ data for images
    gz_data = test_dataset.gz_data
    
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(20, 4 * n_rows), dpi=150)
    fig.suptitle(f'MAE Feature {feature_idx}', fontsize=16)
    
    for i in range(len(top_acts)):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        act_val = top_acts[i].item()
        test_idx = top_indices[i].item()
        
        try:
            # Get the corresponding image from GZ data
            row = gz_data.iloc[test_idx]
            id_str = row['id_str']
            
            # Extract image from HF dataset format
            if 'image' in row and row['image'] is not None:
                img_data = row['image']
                if hasattr(img_data, 'get'):
                    img = img_data  # Already a PIL image
                else:
                    # Handle bytes data if needed
                    img = Image.open(io.BytesIO(img_data['bytes']))
                
                # Ensure image is RGB
                if hasattr(img, 'mode') and img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Resize and display
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'val: {act_val:.2f}\nID: {id_str[:20]}')
            else:
                ax.text(0.5, 0.5, f"No image:\n{id_str[:20]}", 
                       ha='center', va='center')
                ax.axis('off')
                
        except Exception as e:
            print(f"Error loading image for index {test_idx}: {e}")
            ax.text(0.5, 0.5, f"Error loading\nindex {test_idx}", 
                   ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def analyze_features_by_frequency_and_magnitude(
    activations: torch.Tensor,
    min_freq: float = 0.05,
    sort_by: str = "frequency"
):
    """
    Analyze features by activation frequency and magnitude.
    
    Args:
        activations: Dense feature activations [n_samples, n_features]
        min_freq: Minimum activation frequency threshold
        sort_by: "frequency" or "magnitude" - how to rank features
        
    Returns:
        Tuple of (active_features, frequencies, mean_magnitudes, sorted_features)
    """
    n_features = activations.shape[1]
    
    # Compute frequencies and mean magnitudes efficiently
    frequencies = (activations > 0).float().mean(dim=0)
    mean_magnitudes = activations.sum(dim=0) / (frequencies * len(activations) + 1e-10)
    
    # Find active features
    active_features = torch.where(frequencies > min_freq)[0]
    print(f"Found {len(active_features)} features activated > {min_freq*100:.1f}% of the time")
    
    # Sort features by chosen metric
    if sort_by == "frequency":
        feature_scores = frequencies[active_features]
        sort_desc = "frequency"
    elif sort_by == "magnitude":
        feature_scores = mean_magnitudes[active_features]
        sort_desc = "mean activation strength"
    else:
        raise ValueError("sort_by must be 'frequency' or 'magnitude'")
    
    sorted_idxs = torch.argsort(feature_scores, descending=True)
    top_features = active_features[sorted_idxs]
    
    print(f"Ranked features by {sort_desc}")
    
    return active_features, frequencies, mean_magnitudes, top_features


def create_feature_gallery(
    feature_activations: Union[torch.Tensor, np.ndarray],
    feature_id: int,
    test_ids: List[str],
    hf_dataset,
    id_to_idx_mapping: dict,
    reverse: bool = False,
    n_images: int = 10,
    save_path: Optional[Path] = None
):
    """
    Create a single-row gallery of galaxy images for a specific feature.
    
    Args:
        feature_activations: Activations for one feature [n_samples]
        feature_id: Feature identifier for naming
        test_ids: List of test sample IDs
        hf_dataset: HuggingFace dataset with images
        id_to_idx_mapping: Mapping from ID strings to dataset indices
        reverse: If True, show images with lowest activations
        n_images: Number of images to show
        save_path: Path to save the gallery
    """
    # Get activations for this feature
    if reverse:
        # Show most negative activations (lowest values)
        top_indices = np.argsort(feature_activations)[:n_images]
    else:
        # Show most positive activations (highest values)  
        top_indices = np.argsort(feature_activations)[-n_images:][::-1]
    
    # Create figure with single row of images
    fig, axes = plt.subplots(1, n_images, figsize=(20, 2), dpi=300)
    if n_images == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < len(top_indices):
            sample_idx = top_indices[i]
            id_str = test_ids[sample_idx]
            
            try:
                if id_str in id_to_idx_mapping:
                    dataset_idx = id_to_idx_mapping[id_str]
                    img = hf_dataset[dataset_idx]['image']
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
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Set minimal spacing between images
    plt.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()