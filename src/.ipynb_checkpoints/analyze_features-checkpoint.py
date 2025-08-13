import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image

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
    # Load embeddings from NumPy file
    embeddings = torch.from_numpy(np.load(embedding_path).astype(np.float32))
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    n_dirs = checkpoint['model_state_dict']['encoder.weight'].shape[0]
    d_model = checkpoint['model_state_dict']['encoder.weight'].shape[1]
    k = 8  # Using same k as in original code
    
    model = KSparseAutoencoder(n_dirs, d_model, k)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                      if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()

    activations, indices, values = [], [], []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), batch_size)):
            batch = embeddings[i:i+batch_size].to(device)
            latents, topk_values, topk_indices = model(batch)
            
            activations.append(latents.cpu())
            indices.append(topk_indices.cpu())
            values.append(topk_values.cpu())

    activations = torch.cat(activations)
    indices = torch.cat(indices) 
    values = torch.cat(values)
    
    return activations, indices, values

def plot_feature_examples(feature_idx, indices, values, image_paths, root_dir, n_examples=20, save_path=None):
    # Find examples where this feature is activated
    feature_activations = []
    for i in range(len(indices)):
        if feature_idx in indices[i]:
            feat_idx = (indices[i] == feature_idx).nonzero().item()
            feature_activations.append((i, values[i, feat_idx]))
            
    # Sort by activation strength
    feature_activations.sort(key=lambda x: x[1], reverse=True)
    top_examples = feature_activations[:n_examples]
    
    # Calculate rows and columns for subplot grid
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols  # Ceiling division
    
    # Plot top examples
    fig = plt.figure(figsize=(20, 4 * n_rows), dpi=150)
    
    for i, (idx, val) in enumerate(top_examples):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        # Load and display the image
        rel_path = image_paths[idx]
        # Combine with root directory to get absolute path
        img_path = Path(root_dir) / rel_path
        try:
            # Load image and resize to a consistent square size
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)  # Resize to 224x224 square
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'val: {val:.2f}')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            ax.text(0.5, 0.5, f"Error loading\n{Path(rel_path).name}", 
                   ha='center', va='center')
            ax.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    plt.close()

def analyze_activations(model_path, embedding_path, image_paths_file, output_dir, root_dir='../', min_freq=0.05, n_examples=30):
    print("Computing activations...")
    activations, indices, values = compute_activations(model_path, embedding_path)
    
    # Load image paths
    image_paths = np.load(image_paths_file, allow_pickle=True)
    
    # Calculate feature statistics
    n_features = activations.shape[1]
    frequencies = torch.zeros(n_features)
    mean_magnitudes = torch.zeros(n_features)
    
    for i in range(len(indices)):
        frequencies[indices[i]] += 1
        mean_magnitudes[indices[i]] += values[i]
    
    frequencies = frequencies / len(indices)
    mean_magnitudes = mean_magnitudes / (frequencies * len(indices) + 1e-10)  # Avoid division by zero
    
    # Find active features
    active_features = torch.where(frequencies > min_freq)[0]
    print(f"\nFound {len(active_features)} features activated > {min_freq*100:.1f}% of the time")
    
    # Sort by frequency of activation (instead of magnitude)
    feature_scores = frequencies[active_features]
    sorted_idxs = torch.argsort(feature_scores, descending=True)
    top_features = active_features[sorted_idxs]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot top features
    print("\nPlotting top feature examples by frequency...")
    for i, feat_idx in enumerate(top_features[:n_examples]):
        print(f"Feature {feat_idx}: {frequencies[feat_idx]*100:.1f}% activation rate, {mean_magnitudes[feat_idx]:.3f} mean magnitude")
        plot_feature_examples(
            feat_idx.item(), 
            indices,
            values,
            image_paths,
            root_dir,
            n_examples=n_examples, 
            save_path=output_dir / f'feature_{feat_idx}.png'
        )
        
    return frequencies, mean_magnitudes, indices, values

if __name__ == '__main__':
    MODEL_PATH = '../results/sae/best_model.pt'
    EMBEDDING_PATH = '../results/euclid_q1_embeddings.npy'
    IMAGE_PATHS_FILE = '../results/euclid_q1_image_paths.npy'
    OUTPUT_DIR = '../results/sae/euclid-feature-analysis'
    ROOT_DIR = '../'  # Root directory relative to the script location
    
    # Get absolute path for root directory
    ROOT_DIR = Path(ROOT_DIR).resolve()
    print(f"Using root directory: {ROOT_DIR}")
    
    frequencies, magnitudes, indices, values = analyze_activations(
        MODEL_PATH,
        EMBEDDING_PATH,
        IMAGE_PATHS_FILE,
        OUTPUT_DIR,
        ROOT_DIR
    )