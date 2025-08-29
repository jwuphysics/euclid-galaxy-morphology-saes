"""
This script trains and evaluates a Matryoshka Sparse Autoencoder (SAE)
on MAE embeddings from Hugging Face datasets. It uses Galaxy Zoo (GZ) labels 
from mwalmsley/euclid_q1 and MAE embeddings from mwalmsley/euclid_q1_embeddings
for self-supervised learning (SSL). The datasets are crossmatched on id_str.
"""

import argparse
import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tomli
import wandb
from datasets import load_dataset

# Import shared utilities
from .models import MatryoshkaSAE
from .utils import (
    set_seed, configure_torch_reproducibility,
    load_config, get_ssl_dataloaders, MAEEmbeddingDataset
)


# MatryoshkaSAE class now imported from models module




def setup_run_directory(base_dir: Path) -> Dict[str, Path]:
    run_dir = base_dir 
    paths = {
        "run": run_dir,
        "weights": run_dir / "weights",
        "logs": run_dir / "logs",
        "figures": run_dir / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_file: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )




def train_sae(
    model: MatryoshkaSAE,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    train_loader: DataLoader,
    config: Dict[str, Any],
) -> MatryoshkaSAE:
    logging.info("Starting Matryoshka SAE training...")
    model.train()

    for epoch in range(config["EPOCHS"]):
        epoch_metrics = {
            "loss": 0.0, "l2_loss": 0.0, "l1_loss": 0.0,
            "aux_loss": 0.0, "l0_norm": 0.0
        }
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['EPOCHS']}", leave=False):
            optimizer.zero_grad()
            original_embeddings = batch.to(config["DEVICE"])
            
            sae_output = model(original_embeddings)
            loss = sae_output["loss"]
            loss.backward()
            
            model.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            model.update_inactive_features(sae_output["feature_acts"])
            
            for key in epoch_metrics:
                if key in sae_output:
                    epoch_metrics[key] += sae_output[key].item()

        # Log average metrics for the epoch
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        dead_features = (model.num_batches_not_active >= config["N_BATCHES_TO_DEAD"]).sum().item()
        dead_feature_fraction = dead_features / model.total_dict_size
        
        wandb_metrics = {
            'epoch': epoch + 1,
            'train_loss': epoch_metrics['loss'],
            'l2_loss': epoch_metrics['l2_loss'],
            'l1_loss': epoch_metrics['l1_loss'],
            'aux_loss': epoch_metrics['aux_loss'],
            'l0_norm': epoch_metrics['l0_norm'],
            'dead_features': dead_features,
            'dead_feature_fraction': dead_feature_fraction,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        wandb.log(wandb_metrics)
        
        logging.info(
            f"Epoch {epoch+1:>4d}/{config['EPOCHS']} | Loss: {epoch_metrics['loss']:.4f} | "
            f"L2: {epoch_metrics['l2_loss']:.4f} | L1: {epoch_metrics['l1_loss']:.4f} | "
            f"Aux: {epoch_metrics['aux_loss']:.4f} | L0: {epoch_metrics['l0_norm']:.2f} | "
            f"Dead: {dead_features}"
        )
        
        scheduler.step()

    logging.info("SAE Training Finished.")
    return model


def evaluate_sae(
    model: MatryoshkaSAE,
    test_loader: DataLoader,
    config: Dict[str, Any],
    figures_dir: Path,
):
    logging.info("Starting SAE evaluation...")
    model.eval()

    cumulative_features = np.cumsum(model.group_sizes)
    num_nesting_levels = len(cumulative_features)

    logging.info("Pass 1/2: Calculating dataset variance...")
    total_elements = 0
    sum_of_values = 0.0
    sum_of_squares = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calculating Variance"):
            embeddings = batch.to(config["DEVICE"])
            total_elements += embeddings.numel()
            sum_of_values += embeddings.sum()
            sum_of_squares += torch.sum(embeddings ** 2)

    mean = sum_of_values / total_elements
    data_variance = (sum_of_squares / total_elements) - (mean ** 2)
    data_variance = data_variance.item() if data_variance > 0 else 1.0
    logging.info(f"Calculated total data variance: {data_variance:.4f}")

    logging.info("Pass 2/2: Calculating reconstruction errors...")
    total_sq_errors_per_level = [0.0] * num_nesting_levels
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calculating MSE"):
            embeddings = batch.to(config["DEVICE"])
            x_cent = embeddings - model.b_dec
            acts = F.relu(x_cent @ model.W_enc + model.b_enc)

            n_to_keep = model.top_k * embeddings.shape[0]
            top_k_values, top_k_indices = torch.topk(acts.flatten(), k=n_to_keep, sorted=False)
            batch_latents_topk = torch.zeros_like(acts.flatten()).scatter_(0, top_k_indices, top_k_values).view_as(acts)

            for i, num_features in enumerate(cumulative_features):
                latents_slice = batch_latents_topk[:, :num_features]
                w_dec_slice = model.W_dec[:num_features, :]
                reconstruction = model.b_dec + (latents_slice @ w_dec_slice)
                sq_error = (reconstruction - embeddings).pow(2).sum().item()
                total_sq_errors_per_level[i] += sq_error

    mses = [total_sq_err / total_elements for total_sq_err in total_sq_errors_per_level]
    explained_variances = [1 - (mse / data_variance) for mse in mses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.suptitle("SAE Performance vs. Dictionary Size (Matryoshka Levels)", fontsize=16)

    ax1.plot(cumulative_features, mses, 'o-', color='tab:red')
    ax1.set_title("Reconstruction Error")
    ax1.set_xlabel("Cumulative Dictionary Size Used")
    ax1.set_ylabel("MSE")
    ax1.set_xscale('log')
    ax1.set_xticks(cumulative_features, cumulative_features, rotation=45, ha="right")
    ax1.minorticks_off()
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    ax2.plot(cumulative_features, explained_variances, 'o-', color='tab:blue')
    ax2.set_title("Explained Variance")
    ax2.set_xlabel("Cumulative Dictionary Size Used")
    ax2.set_ylabel(r"Explained Variance ($R^2$)")
    ax2.set_xscale('log')
    ax2.set_xticks(cumulative_features, cumulative_features, rotation=45, ha="right")
    ax2.set_ylim(-0.05, 1.05) 
    ax2.axhline(y=1.0, color='gray', linestyle='--', label="100% Explained")
    ax2.axhline(y=0.95, color='green', linestyle=':', label="95% Explained")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.minorticks_off()
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plot_path = figures_dir / "sae_performance_vs_dict_size.png"
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Evaluation plot saved to {plot_path}")


def compute_matryoshka_activations(
    model: MatryoshkaSAE, 
    test_loader: DataLoader,
    config: Dict[str, Any]
) -> torch.Tensor:
    logging.info("Computing dense activations for all validation data...")
    model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing activations"):
            embeddings = batch.to(config["DEVICE"])
            
            x_cent = embeddings - model.b_dec
            pre_acts = x_cent @ model.W_enc + model.b_enc
            activations = F.relu(pre_acts)
            
            all_activations.append(activations.cpu())
    
    return torch.cat(all_activations, dim=0)


def plot_matryoshka_feature_examples(
    feature_idx: int, 
    activations: torch.Tensor,
    test_dataset: MAEEmbeddingDataset,
    n_examples: int = 30, 
    save_path: Path = None
):
    feature_acts = activations[:, feature_idx]
    top_acts, top_indices = torch.topk(feature_acts, k=min(n_examples, len(feature_acts)))
    
    # Get corresponding GZ data for images
    gz_data = test_dataset.gz_data
    
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(20, 4 * n_rows), dpi=150)
    
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


def visualize_top_activating_images(
    model: MatryoshkaSAE,
    test_loader: DataLoader,
    test_dataset: MAEEmbeddingDataset,
    config: Dict[str, Any],
    figures_dir: Path,
    feature_indices_to_inspect: List[int],
    num_top_images: int = 8,
):
    logging.info("Starting efficient feature visualization for Matryoshka SAE.")
    
    all_activations = compute_matryoshka_activations(model, test_loader, config)
    
    logging.info("Generating feature plots...")
    for feature_idx in tqdm(feature_indices_to_inspect, desc="Plotting features"):
        plot_path = figures_dir / f"feature_{feature_idx}.png"
        plot_matryoshka_feature_examples(
            feature_idx=feature_idx,
            activations=all_activations,
            test_dataset=test_dataset,
            n_examples=num_top_images,
            save_path=plot_path
        )
    
    logging.info(f"Saved feature visualization plots to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate Matryoshka SAE on Euclid Q1 MAE embeddings")
    parser.add_argument("--train", action="store_true", 
                       help="Run training mode (default: evaluation mode)")
    parser.add_argument("--config", type=str, default="src/config/mae_matryoshka_sae_config.toml",
                       help="Path to configuration file (default: src/config/mae_matryoshka_sae_config.toml)")
    args = parser.parse_args()
    
    config = load_config(config_path=args.config, train=args.train) 
    paths = setup_run_directory(config["BASE_RESULTS_DIR"])
    setup_logging(paths["logs"] / "run.log")
    
    # Initialize reproducible random state
    set_seed(config["RANDOM_SEED"])
    configure_torch_reproducibility()

    mode = "Training" if args.train else "Evaluation"
    logging.info(f"Running in {mode} mode")
    logging.info(f"Using device: {config['DEVICE']}")
    logging.info(f"Run directory created at: {paths['run']}")

    train_loader, test_loader, train_dataset, test_dataset = get_ssl_dataloaders(config)

    if args.train:
        wandb.init(
            project=config["WANDB_PROJECT"],
            config=config,
            dir=str(paths["run"])
        )

    model = MatryoshkaSAE(
        input_dim=config["INPUT_DIM"],
        group_sizes=config["GROUP_SIZES"],
        top_k=config["TOP_K"],
        l1_coeff=config["L1_COEFF"],
        aux_penalty=config["AUX_PENALTY"],
        n_batches_to_dead=config["N_BATCHES_TO_DEAD"],
        aux_k_multiplier=config["AUX_K_MULTIPLIER"],
    ).to(config["DEVICE"])

    optimizer = optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["EPOCHS"],
        eta_min=config["LEARNING_RATE"]/10
    )

    model_path = paths["weights"] / config["MODEL_FILENAME"]
    
    if config["EPOCHS"] > 0:
        model = train_sae(model, optimizer, scheduler, train_loader, config)
        torch.save(model.state_dict(), model_path)
        logging.info(f"SAE model saved to {model_path}")
    elif model_path.exists():
        logging.info(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=config["DEVICE"]))
    else:
        logging.error("EPOCHS is 0 and no pre-trained model found. Exiting.")
        logging.error(f"Please train a model first, or place a saved model at: {model_path}")
        return

    evaluate_sae(model, test_loader, config, paths["figures"])
    
    max_features = min(config["MAX_FEATURES_TO_VISUALIZE"], sum(config["GROUP_SIZES"]))
    features_to_look_at = list(range(0, max_features))
    visualize_top_activating_images(
        model=model,
        test_loader=test_loader,
        test_dataset=test_dataset,
        config=config,
        figures_dir=paths["figures"],
        feature_indices_to_inspect=features_to_look_at,
        num_top_images=config["TOP_IMAGES_PER_FEATURE"],
    )


if __name__ == "__main__":
    main()