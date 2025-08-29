"""
This script performs PCA on Euclid Q1 embeddings and visualizes the top principal components
to provide a baseline comparison with SAE features.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
from datasets import load_dataset

# Import shared utilities
from .utils import set_seed, configure_torch_reproducibility


def compute_pca_components(embedding_path: str, n_components: int = 32, standardize: bool = True):
    print("Loading embeddings...")
    embeddings = np.load(embedding_path).astype(np.float32)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    if standardize:
        print("Standardizing features...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    print(f"Computing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(embeddings)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")  # Show first 5
    print(f"Cumulative explained variance (first 10): {np.cumsum(pca.explained_variance_ratio_[:10])}")
    
    return pca, pca.components_, transformed_data


def plot_pca_component_examples_optimized(
    component_idx: int,
    top_indices: np.ndarray,
    top_values: np.ndarray,
    hf_dataset,
    embedding_ids: np.ndarray,
    id_to_idx: dict,
    n_examples: int = 30,
    save_path: Path = None,
    explained_variance_ratio: float = None
):
    """Optimized version that takes pre-computed top indices and values"""
    
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 4 * n_rows), dpi=150)
    
    for i, (img_idx, val) in enumerate(zip(top_indices, top_values)):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        id_str = embedding_ids[img_idx]
        
        try:
            # Find the corresponding image in HF dataset
            if id_str in id_to_idx:
                dataset_idx = id_to_idx[id_str]
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
        # print(f"Saved figure to {save_path}")
    plt.close()


def analyze_pca_components(
    embedding_path: str,
    embedding_ids_path: str,
    output_dir: str,
    dataset_name: str = "mwalmsley/gz_euclid",
    split: str = "test",
    n_components: int = 16,
    n_examples: int = 30,
    components_to_plot: list = None
):
    pca, components, transformed_data = compute_pca_components(
        embedding_path, n_components=n_components
    )
    
    # Load embedding IDs and HF dataset ONCE
    print("Loading embedding IDs and HF dataset...")
    embedding_ids = np.load(embedding_ids_path, allow_pickle=True)
    print(f"Loading HF dataset: {dataset_name}, split: {split}")
    if "euclid_q1" in dataset_name:
        hf_dataset = load_dataset(dataset_name, "v1-gz_arcsinh_vis_y", split=split)
    else:
        hf_dataset = load_dataset(dataset_name, split=split)
    
    # Create id_str to index mapping ONCE for all components
    print("Creating ID lookup mapping...")
    id_to_idx = {item['id_str']: idx for idx, item in enumerate(hf_dataset)}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    np.save(output_dir / "pca_components.npy", components)
    np.save(output_dir / "pca_transformed_data.npy", transformed_data)
    np.save(output_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)
    
    plt.figure(figsize=(12, 5), dpi=150)
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.95, color='green', linestyle=':', label="95% Explained")
    plt.axhline(y=0.90, color='orange', linestyle=':', label="90% Explained")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "pca_explained_variance.png")
    plt.close()
    print(f"Saved explained variance plot to {output_dir / 'pca_explained_variance.png'}")
    
    if components_to_plot is None:
        components_to_plot = list(range(min(16, n_components)))
    
    # Pre-compute top indices for all components for better performance
    print("Pre-computing top activating examples for all components...")
    all_top_indices = {}
    for component_idx in tqdm(components_to_plot, desc="Computing top examples"):
        component_values = transformed_data[:, component_idx]
        abs_values = np.abs(component_values)
        top_indices = np.argsort(abs_values)[-n_examples:][::-1]
        all_top_indices[component_idx] = (top_indices, component_values[top_indices])
    
    print("Generating component visualization plots...")
    for component_idx in tqdm(components_to_plot, desc="Plotting components"):
        plot_path = output_dir / f"pca_component_{component_idx}.png"
        top_indices, top_values = all_top_indices[component_idx]
        
        plot_pca_component_examples_optimized(
            component_idx=component_idx,
            top_indices=top_indices,
            top_values=top_values,
            hf_dataset=hf_dataset,
            embedding_ids=embedding_ids,
            id_to_idx=id_to_idx,
            n_examples=n_examples,
            save_path=plot_path,
            explained_variance_ratio=pca.explained_variance_ratio_[component_idx]
        )
    
    print(f"Analysis complete! Results saved to {output_dir}")
    
    return pca, components, transformed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze PCA components of Euclid Q1 embeddings")
    parser.add_argument('--embedding_path', type=str, 
                      default='./results/euclid_test_embeddings.npy',
                      help='Path to numpy file containing embeddings')
    parser.add_argument('--embedding_ids_path', type=str,
                      default='./results/euclid_test_ids.npy',
                      help='Path to numpy file containing embedding IDs')
    parser.add_argument('--output_dir', type=str, 
                      default='./results/pca',
                      help='Directory to save analysis results')
    parser.add_argument('--dataset_name', type=str, default='mwalmsley/gz_euclid',
                      help='Huggingface dataset name')
    parser.add_argument('--split', type=str, default='test',
                      help='Dataset split to use for visualization')
    parser.add_argument('--n_components', type=int, default=64,
                      help='Number of PCA components to compute')
    parser.add_argument('--n_examples', type=int, default=30,
                      help='Number of example images to show per component')
    parser.add_argument('--max_components_to_plot', type=int, default=64,
                      help='Maximum number of components to plot')
    
    args = parser.parse_args()
    
    # Initialize reproducible random state
    set_seed(42)
    configure_torch_reproducibility()
    
    components_to_plot = list(range(min(args.max_components_to_plot, args.n_components)))
    
    analyze_pca_components(
        embedding_path=args.embedding_path,
        embedding_ids_path=args.embedding_ids_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        n_components=args.n_components,
        n_examples=args.n_examples,
        components_to_plot=components_to_plot
    )