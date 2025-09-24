"""
Extract MAE embeddings from Hugging Face datasets for SSL analysis.

This script loads the MAE embeddings dataset from Hugging Face, crossmatches with
Galaxy Zoo labels, and saves the embeddings and IDs to disk for use in downstream
analysis scripts like make_figures.py.
"""

import logging
import numpy as np
from pathlib import Path
from datasets import load_dataset

# Import shared utilities
try:
    from .utils import set_seed, configure_torch_reproducibility, load_config, MAEEmbeddingDataset
except ImportError:
    # When running as script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from utils import set_seed, configure_torch_reproducibility, load_config, MAEEmbeddingDataset

# Repository base path (parent of src/)
REPO_BASE = Path(__file__).parent.parent

def extract_mae_embeddings():
    """Extract MAE embeddings and save to disk for analysis."""
    
    # Load configuration
    config_path = REPO_BASE / 'src/config/mae_matryoshka_sae_config.toml'
    config = load_config(config_path, train=False)
    
    logging.info("Loading GZ and MAE datasets from Hugging Face...")
    
    # Load datasets
    gz_test = load_dataset(config["GZ_DATASET_NAME"], config["GZ_CONFIG_NAME"], split="test")
    mae_test = load_dataset(config["MAE_DATASET_NAME"], split="test")
    
    # Create crossmatched dataset
    logging.info(f"Using embedding block: {config['EMBEDDING_BLOCK']}")
    test_dataset = MAEEmbeddingDataset(gz_test, mae_test, config["EMBEDDING_BLOCK"], normalize=False)
    
    # Extract embeddings and IDs
    embeddings = test_dataset.embeddings.numpy()
    ids = test_dataset.id_strs
    
    # Create output directory
    output_dir = REPO_BASE / 'results/mae_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings and IDs
    embeddings_path = output_dir / 'mae_test_embeddings.npy'
    ids_path = output_dir / 'mae_test_ids.npy'
    
    np.save(embeddings_path, embeddings)
    np.save(ids_path, ids)
    
    logging.info(f"Saved MAE test embeddings: {embeddings_path}")
    logging.info(f"Saved MAE test IDs: {ids_path}")
    logging.info(f"Embeddings shape: {embeddings.shape}")
    logging.info(f"Number of IDs: {len(ids)}")
    
    return embeddings_path, ids_path, embeddings.shape

if __name__ == "__main__":
    # Initialize reproducible random state
    set_seed(42)
    configure_torch_reproducibility()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    print("Extracting MAE embeddings from Hugging Face datasets...")
    embeddings_path, ids_path, shape = extract_mae_embeddings()
    print(f"✅ Successfully extracted MAE embeddings: {shape}")
    print(f"   Embeddings saved to: {embeddings_path}")
    print(f"   IDs saved to: {ids_path}")