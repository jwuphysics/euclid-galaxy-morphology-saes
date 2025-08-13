"""
This script trains and evaluates a Matryoshka Sparse Autoencoder (SAE)
on pre-computed Euclid Q1 embeddings. It also includes functionality
to visualize the galaxy images that most strongly activate learned SAE features.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_cent = x - self.b_dec
        
        pre_acts = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)

        batch_size = x.shape[0]
        n_to_keep = self.top_k * batch_size

        top_k_values, top_k_indices = torch.topk(acts.flatten(), k=n_to_keep, sorted=False)
        
        acts_topk = torch.zeros_like(acts.flatten()).scatter_(0, top_k_indices, top_k_values)
        acts_topk = acts_topk.view_as(acts)

        intermediate_reconstructions = []
        current_reconstruction = self.b_dec.expand_as(x)
        
        for i in range(len(self.group_sizes)):
            start_idx, end_idx = self.group_indices[i], self.group_indices[i+1]
            
            group_acts = acts_topk[:, start_idx:end_idx]
            group_W_dec = self.W_dec[start_idx:end_idx, :]
            
            # Add only this group's contribution to the reconstruction
            current_reconstruction = current_reconstruction + (group_acts @ group_W_dec)
            intermediate_reconstructions.append(current_reconstruction.clone())
            
        final_reconstruction = current_reconstruction

        l2_losses = [(recon.float() - x.float()).pow(2).mean() for recon in intermediate_reconstructions]
        mean_l2_loss = torch.stack(l2_losses).mean()
        l1_loss = self.l1_coeff * torch.norm(acts_topk, p=1, dim=-1).mean()
        aux_loss = self.get_auxiliary_loss(x, final_reconstruction, acts)
        total_loss = mean_l2_loss + l1_loss + aux_loss
        
        l0_norm = (acts_topk > 0).float().sum(dim=-1).mean()

        return {
            "loss": total_loss,
            "l2_loss": mean_l2_loss,
            "l1_loss": l1_loss,
            "aux_loss": aux_loss,
            "l0_norm": l0_norm,
            "feature_acts": acts_topk,
            "final_reconstruction": final_reconstruction
        }

    @torch.no_grad()
    def update_inactive_features(self, feature_acts: torch.Tensor):
        active_features_mask = feature_acts.sum(dim=0) > 0
        self.num_batches_not_active[~active_features_mask] += 1
        self.num_batches_not_active[active_features_mask] = 0

    def get_auxiliary_loss(
        self, x: torch.Tensor, final_reconstruction: torch.Tensor, dense_acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates a loss to encourage "dead" features to explain the residual
        (the part of the input not explained by the main reconstruction). This is a
        key mechanism for preventing feature collapse during training.
        """
        dead_features_mask = self.num_batches_not_active >= self.n_batches_to_dead
        
        if not dead_features_mask.any():
            return torch.tensor(0.0, device=x.device)
            
        residual = (x.float() - final_reconstruction.float()).detach()
        dead_acts = dense_acts[:, dead_features_mask]
        
        k_aux = self.top_k * self.aux_k_multiplier
        n_to_keep_aux = k_aux * x.shape[0]

        if dead_acts.numel() == 0 or n_to_keep_aux == 0:
            return torch.tensor(0.0, device=x.device)

        n_to_keep_aux = min(n_to_keep_aux, dead_acts.numel())
        top_k_aux_values, _ = torch.topk(dead_acts.flatten(), k=n_to_keep_aux)
        
        min_top_k_val = top_k_aux_values[-1]
        acts_aux = F.relu(dead_acts - min_top_k_val)
        
        recon_aux = acts_aux @ self.W_dec[dead_features_mask, :]
        aux_loss = self.aux_penalty * (recon_aux.float() - residual.float()).pow(2).mean()
        
        return aux_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """
        Normalizes decoder weights and projects their gradients to be orthogonal
        to the weights. This is a important step for training stability, ensuring
        the update step does not change the norm of the decoder weights.
        """
        if self.W_dec.grad is None:
            return
            
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed


class EmbeddingDataset(Dataset):
    def __init__(self, npy_path: str, normalize: bool = True):
        self.embeddings = torch.from_numpy(np.load(npy_path))
            
        if normalize:
            mean = self.embeddings.mean(0, keepdim=True)
            std = self.embeddings.std(0, keepdim=True)
            self.embeddings = (self.embeddings - mean) / (std + 1e-5)
            
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx]


def load_config(config_path: str = "matryoshka_sae_config.toml", train: bool = False) -> Dict[str, Any]:
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
        "EMBEDDING_PATH": toml_config["data"]["embedding_path"],
        "IMAGE_PATHS_FILE": toml_config["data"]["image_paths_file"],
        "ROOT_DIR": toml_config["data"]["root_dir"],
        "VALIDATION_SPLIT": toml_config["data"]["validation_split"],
        "RANDOM_SEED": toml_config["data"]["random_seed"],
        "BASE_RESULTS_DIR": Path(toml_config["output"]["results_dir"]),
        "MODEL_FILENAME": toml_config["output"]["model_filename"],
        "WANDB_PROJECT": toml_config["output"]["wandb_project"],
        "MAX_FEATURES_TO_VISUALIZE": toml_config["visualization"]["max_features_to_visualize"],
        "TOP_IMAGES_PER_FEATURE": toml_config["visualization"]["top_images_per_feature"],
    }
    
    device_config = toml_config["training"]["device"]
    if device_config == "auto":
        config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config["DEVICE"] = device_config
        
    return config


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


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    logging.info("Loading Euclid Q1 embedding dataset...")
    
    full_dataset = EmbeddingDataset(config["EMBEDDING_PATH"], normalize=True)
    
    total_size = len(full_dataset)
    val_size = int(config["VALIDATION_SPLIT"] * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(config["RANDOM_SEED"])
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["SAE_BATCH_SIZE"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    validation_loader = DataLoader(
        val_dataset, 
        batch_size=config["SAE_BATCH_SIZE"] * 2, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info("Dataloaders created.")
    return train_loader, validation_loader


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
    validation_loader: DataLoader,
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
        for batch in tqdm(validation_loader, desc="Calculating Variance"):
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
        for batch in tqdm(validation_loader, desc="Calculating MSE"):
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
    validation_loader: DataLoader,
    config: Dict[str, Any]
) -> torch.Tensor:
    logging.info("Computing dense activations for all validation data...")
    model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Computing activations"):
            embeddings = batch.to(config["DEVICE"])
            
            x_cent = embeddings - model.b_dec
            pre_acts = x_cent @ model.W_enc + model.b_enc
            activations = F.relu(pre_acts)
            
            all_activations.append(activations.cpu())
    
    return torch.cat(all_activations, dim=0)


def plot_matryoshka_feature_examples(
    feature_idx: int, 
    activations: torch.Tensor,
    image_paths: np.ndarray,
    val_indices: List[int],
    root_dir: str,
    n_examples: int = 30, 
    save_path: Path = None
):
    feature_acts = activations[:, feature_idx]
    top_acts, top_indices = torch.topk(feature_acts, k=min(n_examples, len(feature_acts)))
    
    top_examples = [(top_acts[i].item(), val_indices[top_indices[i].item()]) 
                   for i in range(len(top_acts))]
    
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(20, 4 * n_rows), dpi=150)
    
    for i, (act_val, img_idx) in enumerate(top_examples):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        rel_path = image_paths[img_idx]
        img_path = Path(root_dir) / rel_path
        try:
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'val: {act_val:.2f}')  # Same title format as analyze_features.py
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


def visualize_top_activating_images(
    model: MatryoshkaSAE,
    validation_loader: DataLoader,
    config: Dict[str, Any],
    figures_dir: Path,
    feature_indices_to_inspect: List[int],
    num_top_images: int = 8,
):
    logging.info("Starting efficient feature visualization for Matryoshka SAE.")
    
    all_activations = compute_matryoshka_activations(model, validation_loader, config)
    
    image_paths = np.load(config["IMAGE_PATHS_FILE"], allow_pickle=True)
    
    total_size = len(image_paths)
    val_size = int(config["VALIDATION_SPLIT"] * total_size)
    train_size = total_size - val_size
    
    indices = list(range(total_size))
    torch.manual_seed(config["RANDOM_SEED"])
    train_indices = torch.randperm(total_size)[:train_size].tolist()
    val_indices = [i for i in indices if i not in train_indices]
    
    logging.info("Generating feature plots...")
    for feature_idx in tqdm(feature_indices_to_inspect, desc="Plotting features"):
        plot_path = figures_dir / f"feature_{feature_idx}.png"  # Same naming as analyze_features.py
        plot_matryoshka_feature_examples(
            feature_idx=feature_idx,
            activations=all_activations,
            image_paths=image_paths,
            val_indices=val_indices,
            root_dir=config["ROOT_DIR"],
            n_examples=num_top_images,
            save_path=plot_path
        )
    
    logging.info(f"Saved feature visualization plots to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate Matryoshka SAE on Euclid Q1 data")
    parser.add_argument("--train", action="store_true", 
                       help="Run training mode (default: evaluation mode)")
    parser.add_argument("--config", type=str, default="matryoshka_sae_config.toml",
                       help="Path to configuration file (default: matryoshka_sae_config.toml)")
    args = parser.parse_args()
    
    config = load_config(config_path=args.config, train=args.train) 
    paths = setup_run_directory(config["BASE_RESULTS_DIR"])
    setup_logging(paths["logs"] / "run.log")

    mode = "Training" if args.train else "Evaluation"
    logging.info(f"Running in {mode} mode")
    logging.info(f"Using device: {config['DEVICE']}")
    logging.info(f"Run directory created at: {paths['run']}")

    train_loader, validation_loader = get_dataloaders(config)

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

    evaluate_sae(model, validation_loader, config, paths["figures"])
    
    max_features = min(config["MAX_FEATURES_TO_VISUALIZE"], sum(config["GROUP_SIZES"]))
    features_to_look_at = list(range(0, max_features))
    visualize_top_activating_images(
        model=model,
        validation_loader=validation_loader,
        config=config,
        figures_dir=paths["figures"],
        feature_indices_to_inspect=features_to_look_at,
        num_top_images=config["TOP_IMAGES_PER_FEATURE"],
    )


if __name__ == "__main__":
    main()