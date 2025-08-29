"""
Data loading utilities for Euclid Q1 SAE analysis.

This module provides common data loading functionality used across different
analysis scripts, including embedding datasets and configuration loading.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tomli
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .reproducibility import get_worker_init_fn


class EmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings with optional normalization."""
    
    def __init__(self, npy_path: str, normalize: bool = True):
        """
        Initialize embedding dataset.
        
        Args:
            npy_path: Path to numpy file containing embeddings
            normalize: Whether to standardize embeddings (mean=0, std=1)
        """
        self.embeddings = torch.from_numpy(np.load(npy_path).astype(np.float32))
            
        if normalize:
            mean = self.embeddings.mean(0, keepdim=True)
            std = self.embeddings.std(0, keepdim=True)
            self.embeddings = (self.embeddings - mean) / (std + 1e-5)
            
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx]


class MAEEmbeddingDataset(Dataset):
    """Dataset for MAE embeddings from Hugging Face with GZ labels."""
    
    def __init__(self, gz_dataset, mae_dataset, embedding_block: str = 'pooled_features_block_11', normalize: bool = True):
        """
        Initialize MAE embedding dataset with crossmatched GZ labels.
        
        Args:
            gz_dataset: Galaxy Zoo dataset from Hugging Face
            mae_dataset: MAE embeddings dataset from Hugging Face
            embedding_block: Which embedding block to use from MAE dataset
            normalize: Whether to standardize embeddings
        """
        # Create crossmatch based on id_str
        gz_df = gz_dataset.to_pandas()
        mae_df = mae_dataset.to_pandas()
        
        # Inner join on id_str to get matched data
        merged_df = pd.merge(gz_df, mae_df, on='id_str', how='inner')
        
        logging.info(f"Original GZ dataset size: {len(gz_df)}")
        logging.info(f"Original MAE dataset size: {len(mae_df)}")
        logging.info(f"Crossmatched dataset size: {len(merged_df)}")
        
        # Extract embeddings
        embeddings_list = merged_df[embedding_block].tolist()
        self.embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
        
        # Store other data for potential use
        self.id_strs = merged_df['id_str'].tolist()
        self.gz_data = merged_df
        
        if normalize:
            mean = self.embeddings.mean(0, keepdim=True)
            std = self.embeddings.std(0, keepdim=True)
            self.embeddings = (self.embeddings - mean) / (std + 1e-5)
            
        logging.info(f"Embedding shape: {self.embeddings.shape}")
            
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx]
    
    def get_id_str(self, idx):
        return self.id_strs[idx]


def load_config(config_path: str, train: bool = False) -> Dict[str, Any]:
    """
    Load configuration from TOML file.
    
    Args:
        config_path: Path to TOML configuration file
        train: Whether training mode is enabled (affects epoch loading)
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, "rb") as f:
        toml_config = tomli.load(f)
    
    config = {
        "INPUT_DIM": toml_config["model"]["input_dim"],
        "GROUP_SIZES": toml_config["model"]["group_sizes"],
        "TOP_K": toml_config["model"]["top_k"],
        "L1_COEFF": toml_config["model"]["l1_coeff"],
        "AUX_PENALTY": toml_config["model"]["aux_penalty"],
        "N_BATCHES_TO_DEAD": toml_config["model"]["n_batches_to_dead"],
        "AUX_K_MULTIPLIER": toml_config["model"]["aux_k_multiplier"],
        "EPOCHS": toml_config["training"]["epochs"] if train else 0,
        "LEARNING_RATE": toml_config["training"]["learning_rate"],
        "SAE_BATCH_SIZE": toml_config["training"]["batch_size"],
        "RANDOM_SEED": toml_config["data"]["random_seed"],
        "BASE_RESULTS_DIR": Path(toml_config["output"]["results_dir"]),
        "MODEL_FILENAME": toml_config["output"]["model_filename"],
        "WANDB_PROJECT": toml_config["output"]["wandb_project"],
        "MAX_FEATURES_TO_VISUALIZE": toml_config["visualization"]["max_features_to_visualize"],
        "TOP_IMAGES_PER_FEATURE": toml_config["visualization"]["top_images_per_feature"],
    }
    
    # Handle different config formats for supervised vs SSL
    if "dataset_name" in toml_config["data"]:
        # Supervised config format
        config.update({
            "DATASET_NAME": toml_config["data"]["dataset_name"],
            "TRAIN_EMBEDDING_PATH": toml_config["data"]["train_embedding_path"],
            "TRAIN_IDS_PATH": toml_config["data"]["train_ids_path"],
            "TEST_EMBEDDING_PATH": toml_config["data"]["test_embedding_path"],
            "TEST_IDS_PATH": toml_config["data"]["test_ids_path"],
        })
    else:
        # SSL config format
        config.update({
            "GZ_DATASET_NAME": toml_config["data"]["gz_dataset_name"],
            "GZ_CONFIG_NAME": toml_config["data"]["gz_config_name"],
            "MAE_DATASET_NAME": toml_config["data"]["mae_dataset_name"],
            "EMBEDDING_BLOCK": toml_config["data"]["embedding_block"],
        })
    
    # Handle device configuration
    device_config = toml_config["training"]["device"]
    if device_config == "auto":
        config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config["DEVICE"] = device_config
        
    return config


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, 
                     num_workers: int = 4, random_seed: int = 42) -> DataLoader:
    """
    Create a DataLoader with reproducible random state.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=get_worker_init_fn(random_seed),
        generator=torch.Generator().manual_seed(random_seed)
    )


def get_supervised_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for supervised (Zoobot) embeddings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    logging.info("Loading supervised Zoobot embeddings...")
    
    train_dataset = EmbeddingDataset(config["TRAIN_EMBEDDING_PATH"], normalize=True)
    test_dataset = EmbeddingDataset(config["TEST_EMBEDDING_PATH"], normalize=True)

    train_loader = create_dataloader(
        train_dataset, 
        batch_size=config["SAE_BATCH_SIZE"], 
        shuffle=True,
        random_seed=config["RANDOM_SEED"]
    )
    test_loader = create_dataloader(
        test_dataset, 
        batch_size=config["SAE_BATCH_SIZE"] * 2, 
        shuffle=False,
        random_seed=config["RANDOM_SEED"]
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    logging.info("Supervised dataloaders created.")
    
    return train_loader, test_loader


def get_ssl_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, MAEEmbeddingDataset, MAEEmbeddingDataset]:
    """
    Create dataloaders for SSL (MAE) embeddings with GZ labels.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, test_loader, train_dataset, test_dataset)
    """
    logging.info("Loading GZ and MAE datasets from Hugging Face...")
    
    # Load GZ dataset with labels
    logging.info(f"Loading GZ dataset: {config['GZ_DATASET_NAME']} with config {config['GZ_CONFIG_NAME']}")
    gz_train = load_dataset(config["GZ_DATASET_NAME"], config["GZ_CONFIG_NAME"], split="train")
    gz_test = load_dataset(config["GZ_DATASET_NAME"], config["GZ_CONFIG_NAME"], split="test")
    
    # Load MAE embeddings dataset
    logging.info(f"Loading MAE embeddings dataset: {config['MAE_DATASET_NAME']}")
    mae_train = load_dataset(config["MAE_DATASET_NAME"], split="train")
    mae_test = load_dataset(config["MAE_DATASET_NAME"], split="test")
    
    # Create crossmatched datasets
    logging.info(f"Using embedding block: {config['EMBEDDING_BLOCK']}")
    train_dataset = MAEEmbeddingDataset(gz_train, mae_train, config["EMBEDDING_BLOCK"], normalize=True)
    test_dataset = MAEEmbeddingDataset(gz_test, mae_test, config["EMBEDDING_BLOCK"], normalize=True)

    train_loader = create_dataloader(
        train_dataset, 
        batch_size=config["SAE_BATCH_SIZE"], 
        shuffle=True,
        random_seed=config["RANDOM_SEED"]
    )
    test_loader = create_dataloader(
        test_dataset, 
        batch_size=config["SAE_BATCH_SIZE"] * 2, 
        shuffle=False,
        random_seed=config["RANDOM_SEED"]
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    logging.info("SSL dataloaders created.")
    
    return train_loader, test_loader, train_dataset, test_dataset