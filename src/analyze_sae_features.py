import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

class KSparseAutoencoder(torch.nn.Module):
    def __init__(self, n_dirs: int, d_model: int, k: int):
        super().__init__()
        self.n_dirs = n_dirs
        self.d_model = d_model
        self.k = k
        
        self.encoder = torch.nn.Linear(d_model, n_dirs, bias=False)
        self.decoder = torch.nn.Linear(n_dirs, d_model, bias=False)
        self.pre_bias = torch.nn.Parameter(torch.zeros(d_model))
        self.latent_bias = torch.nn.Parameter(torch.zeros(n_dirs))

    def forward(self, x):
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias
        
        topk_values, topk_indices = torch.topk(latents_pre_act, k=self.k, dim=-1)
        topk_values = torch.nn.functional.relu(topk_values)
        
        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, topk_indices, topk_values)
        
        return latents, topk_values, topk_indices

def compute_activations(model_path, embedding_path, batch_size=1024, device='cuda'):
    embeddings = torch.from_numpy(np.load(embedding_path).astype(np.float32))
    
    checkpoint = torch.load(model_path, map_location='cpu')
    n_dirs = checkpoint['model_state_dict']['encoder.weight'].shape[0]
    d_model = checkpoint['model_state_dict']['encoder.weight'].shape[1]
    k = 8 
    
    model = KSparseAutoencoder(n_dirs, d_model, k)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                      if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()

    activations, indices, values = [], [], []
    dense_activations = []  # Store dense activations for efficient lookup
    
    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), batch_size)):
            batch = embeddings[i:i+batch_size].to(device)
            
            # Compute dense activations (before top-k selection)
            x = batch - model.pre_bias
            latents_pre_act = model.encoder(x) + model.latent_bias
            dense_acts = torch.nn.functional.relu(latents_pre_act)
            dense_activations.append(dense_acts.cpu())
            
            # Also compute sparse activations for backward compatibility
            latents, topk_values, topk_indices = model(batch)
            activations.append(latents.cpu())
            indices.append(topk_indices.cpu())
            values.append(topk_values.cpu())

    activations = torch.cat(activations)
    indices = torch.cat(indices) 
    values = torch.cat(values)
    dense_activations = torch.cat(dense_activations)
    
    return activations, indices, values, dense_activations

def plot_feature_examples(feature_idx, dense_activations, hf_dataset, embedding_ids, n_examples=20, save_path=None):
    """Efficient version using dense activations - much faster than sparse lookup"""
    # Get activations for this feature across all samples
    feature_acts = dense_activations[:, feature_idx]
    
    # Find top activating examples
    top_acts, top_indices = torch.topk(feature_acts, k=min(n_examples, len(feature_acts)))
    
    # Calculate rows and columns for subplot grid
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols  # Ceiling division
    
    # Create id_str to index mapping for quick lookup
    id_to_idx = {item['id_str']: idx for idx, item in enumerate(hf_dataset)}
    
    # Plot top examples
    fig = plt.figure(figsize=(20, 4 * n_rows), dpi=150)
    
    for i in range(len(top_acts)):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        act_val = top_acts[i].item()
        sample_idx = top_indices[i].item()
        id_str = embedding_ids[sample_idx]
        
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

def analyze_activations(model_path, embedding_path, embedding_ids_path, output_dir, 
                       dataset_name="mwalmsley/gz_euclid", split="test", min_freq=0.05, n_examples=30,
                       sort_by="frequency"):
    """
    Analyze SAE activations and visualize top features.
    
    Args:
        sort_by: "frequency" (default) or "magnitude" - how to rank features
    """
    print("Computing activations...")
    activations, indices, values, dense_activations = compute_activations(model_path, embedding_path)
    
    # Load embedding IDs
    embedding_ids = np.load(embedding_ids_path, allow_pickle=True)
    
    # Load HF dataset
    print(f"Loading HF dataset: {dataset_name}, split: {split}")
    hf_dataset = load_dataset(dataset_name, split=split)
    
    # Calculate feature statistics efficiently using dense activations
    n_features = dense_activations.shape[1]
    
    # Compute frequencies and mean magnitudes efficiently
    frequencies = (dense_activations > 0).float().mean(dim=0)
    mean_magnitudes = dense_activations.sum(dim=0) / (frequencies * len(dense_activations) + 1e-10)
    
    # Find active features
    active_features = torch.where(frequencies > min_freq)[0]
    print(f"\nFound {len(active_features)} features activated > {min_freq*100:.1f}% of the time")
    
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
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot top features using efficient method
    print(f"\nPlotting top feature examples by {sort_desc}...")
    for i, feat_idx in enumerate(top_features[:n_examples]):
        print(f"Feature {feat_idx}: {frequencies[feat_idx]*100:.1f}% activation rate, {mean_magnitudes[feat_idx]:.3f} mean magnitude")
        
        suffix = "_by_freq" if sort_by == "frequency" else "_by_mag"
        plot_feature_examples(
            feat_idx.item(), 
            dense_activations,
            hf_dataset,
            embedding_ids,
            n_examples=n_examples, 
            save_path=output_dir / f'feature_{feat_idx}{suffix}.png'
        )
        
    return frequencies, mean_magnitudes, indices, values

if __name__ == '__main__':
    MODEL_PATH = './results/sae/best_model.pt'
    EMBEDDING_PATH = './results/euclid_test_embeddings.npy'
    EMBEDDING_IDS_PATH = './results/euclid_test_ids.npy'
    OUTPUT_DIR = './results/sae/euclid-feature-analysis'
    DATASET_NAME = "mwalmsley/gz_euclid"
    SPLIT = "test"
    
    print(f"Using HF dataset: {DATASET_NAME}, split: {SPLIT}")
    
    print("\n" + "="*50)
    print("ANALYZING FEATURES BY ACTIVATION STRENGTH")
    print("="*50)
    analyze_activations(
        MODEL_PATH,
        EMBEDDING_PATH,
        EMBEDDING_IDS_PATH,
        OUTPUT_DIR,
        dataset_name=DATASET_NAME,
        split=SPLIT,
        sort_by="magnitude"
    )