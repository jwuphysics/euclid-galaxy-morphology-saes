import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

# Import shared utilities
try:
    from .models import MatryoshkaSAE
    from .utils import set_seed, configure_torch_reproducibility, load_config, plot_feature_examples
except ImportError:
    # When running as script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from models import MatryoshkaSAE
    from utils import set_seed, configure_torch_reproducibility, load_config, plot_feature_examples


def compute_activations(model_path, embedding_path, config_path, batch_size=1024, device='cuda'):
    """Compute activations using MatryoshkaSAE model."""
    embeddings = torch.from_numpy(np.load(embedding_path).astype(np.float32))
    
    # Load configuration
    config = load_config(config_path, train=False)
    
    # Initialize model with config parameters
    model = MatryoshkaSAE(
        input_dim=config["INPUT_DIM"],
        group_sizes=config["GROUP_SIZES"],
        top_k=config["TOP_K"],
        l1_coeff=config["L1_COEFF"],
        aux_penalty=config["AUX_PENALTY"],
        n_batches_to_dead=config["N_BATCHES_TO_DEAD"],
        aux_k_multiplier=config["AUX_K_MULTIPLIER"]
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    # Compute dense activations efficiently
    dense_activations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Computing activations"):
            batch = embeddings[i:i+batch_size].to(device)
            
            # Use the model's built-in method for dense activations
            dense_acts = model.get_feature_activations(batch)
            dense_activations.append(dense_acts.cpu())

    dense_activations = torch.cat(dense_activations)
    
    # Return format compatible with old interface (sparse activations not needed for analysis)
    return None, None, None, dense_activations


def analyze_activations(model_path, embedding_path, embedding_ids_path, config_path, output_dir, 
                       dataset_name="mwalmsley/gz_euclid", split="test", min_freq=0.05, n_examples=30,
                       sort_by="frequency"):
    """
    Analyze MatryoshkaSAE activations and visualize top features.
    
    Args:
        sort_by: "frequency" (default) or "magnitude" - how to rank features
    """
    print("Computing activations...")
    _, _, _, dense_activations = compute_activations(model_path, embedding_path, config_path)
    
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
    
    # Create ID mapping for visualization
    id_to_idx_mapping = {item['id_str']: idx for idx, item in enumerate(hf_dataset)}
    
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
            save_path=output_dir / f'feature_{feat_idx}{suffix}.png',
            id_to_idx_mapping=id_to_idx_mapping
        )
        
    return frequencies, mean_magnitudes, top_features

if __name__ == '__main__':
    # Initialize reproducible random state
    set_seed(42)
    configure_torch_reproducibility()
    
    MODEL_PATH = './results/sae/weights/matryoshka_sae_final.pth'
    CONFIG_PATH = './src/config/matryoshka_sae_config.toml'
    EMBEDDING_PATH = './results/euclid_test_embeddings.npy'
    EMBEDDING_IDS_PATH = './results/euclid_test_ids.npy'
    OUTPUT_DIR = './results/sae/feature-analysis'
    DATASET_NAME = "mwalmsley/gz_euclid"
    SPLIT = "test"
    
    print(f"Using HF dataset: {DATASET_NAME}, split: {SPLIT}")
    
    print("\n" + "="*50)
    print("ANALYZING MATRYOSHKA SAE FEATURES BY ACTIVATION STRENGTH")
    print("="*50)
    analyze_activations(
        MODEL_PATH,
        EMBEDDING_PATH,
        EMBEDDING_IDS_PATH,
        CONFIG_PATH,
        OUTPUT_DIR,
        dataset_name=DATASET_NAME,
        split=SPLIT,
        sort_by="magnitude"
    )