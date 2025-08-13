#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import math

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm

class KSparseAutoencoder(nn.Module):
    def __init__(self, n_dirs: int, d_model: int, k: int, auxk: int = None, multik: int = None):
        super().__init__()
        self.n_dirs = n_dirs
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.multik = multik or 4*k
        
        self.encoder = nn.Linear(d_model, n_dirs, bias=False)
        self.decoder = nn.Linear(n_dirs, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs))
        self.register_buffer('stats_last_nonzero', torch.zeros(n_dirs, dtype=torch.long))

    def forward(self, x):
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias
        
        topk_values, topk_indices = torch.topk(latents_pre_act, k=self.k, dim=-1)
        topk_values = F.relu(topk_values)
        
        multik_values, multik_indices = torch.topk(latents_pre_act, k=self.multik, dim=-1)
        multik_values = F.relu(multik_values)
        
        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, topk_indices, topk_values)
        
        multik_latents = torch.zeros_like(latents_pre_act)
        multik_latents.scatter_(-1, multik_indices, multik_values)
        
        with torch.no_grad():
            self.stats_last_nonzero += 1
            self.stats_last_nonzero.scatter_(0, topk_indices.unique(), 0)
        
        recons = self.decoder(latents) + self.pre_bias
        multik_recons = self.decoder(multik_latents) + self.pre_bias
        
        auxk_values, auxk_indices = None, None
        if self.auxk is not None:
            dead_mask = (self.stats_last_nonzero > 256).float()
            dead_latents_pre_act = latents_pre_act * dead_mask
            auxk_values, auxk_indices = torch.topk(dead_latents_pre_act, k=self.auxk, dim=-1)
            auxk_values = F.relu(auxk_values)
            
        return recons, {
            "topk_indices": topk_indices,
            "topk_values": topk_values,
            "multik_indices": multik_indices,
            "multik_values": multik_values,
            "multik_recons": multik_recons,
            "auxk_indices": auxk_indices,
            "auxk_values": auxk_values,
            "latents_pre_act": latents_pre_act,
            "latents_post_act": latents,
        }

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

def unit_norm_decoder_(autoencoder: KSparseAutoencoder) -> None:
    with torch.no_grad():
        autoencoder.decoder.weight.div_(
            autoencoder.decoder.weight.norm(dim=0, keepdim=True)
        )

def unit_norm_decoder_grad_adjustment_(autoencoder: KSparseAutoencoder) -> None:
    if autoencoder.decoder.weight.grad is not None:
        with torch.no_grad():
            proj = torch.sum(
                autoencoder.decoder.weight * autoencoder.decoder.weight.grad, 
                dim=0, keepdim=True
            )
            autoencoder.decoder.weight.grad.sub_(
                proj * autoencoder.decoder.weight
            )

def compute_losses(model, x, recons, info, auxk_coef=0.1, multik_coef=0.1):
    recons_loss = F.mse_loss(recons, x)
    multik_loss = F.mse_loss(info["multik_recons"], x)
    
    if model.auxk is not None and info["auxk_values"] is not None:
        e = x - recons.detach()
        auxk_latents = torch.zeros_like(info["latents_pre_act"])
        auxk_latents.scatter_(-1, info["auxk_indices"], info["auxk_values"])
        e_hat = model.decoder(auxk_latents)
        auxk_loss = F.mse_loss(e_hat, e)
    else:
        auxk_loss = torch.tensor(0.0).to(x.device)
    
    total_loss = recons_loss + multik_coef * multik_loss + auxk_coef * auxk_loss
    
    return {
        'total_loss': total_loss,
        'recons_loss': recons_loss,
        'multik_loss': multik_loss,
        'auxk_loss': auxk_loss,
    }

def init_model(model: KSparseAutoencoder, data_sample: torch.Tensor):
    model.pre_bias.data = torch.median(data_sample, dim=0).values
    nn.init.xavier_uniform_(model.decoder.weight)
    unit_norm_decoder_(model)
    model.encoder.weight.data = model.decoder.weight.t().clone()
    nn.init.zeros_(model.latent_bias)

def train_epoch(
    model: KSparseAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    auxk_coef: float = 0.1,
    multik_coef: float = 0.1,
    max_grad_norm: float = 1.0
):
    model.train()
    total_loss = 0
    num_batches = len(loader)
    
    for batch in tqdm(loader, desc='Training', leave=False):
        optimizer.zero_grad()
        
        x = batch.to(next(model.parameters()).device)
        recons, info = model(x)
        losses = compute_losses(model, x, recons, info, auxk_coef, multik_coef)
        losses['total_loss'].backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        unit_norm_decoder_grad_adjustment_(model)
        optimizer.step()
        unit_norm_decoder_(model)
        
        total_loss += losses['total_loss'].item()
        
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, 
                      default='./results/euclid_q1_embeddings.npy',
                      help='Path to numpy file containing embeddings')
    parser.add_argument('--output_dir', type=str, 
                      default='./results/sae',
                      help='Directory to save checkpoints and results')
    parser.add_argument('--n_dirs', type=int, default=5120, # = 8 * 640
                      help='Number of learned dictionary elements')
    parser.add_argument('--k', type=int, default=8,
                      help='How many features to use per reconstruction')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--auxk_coef', type=float, default=0.1)
    parser.add_argument('--multik_coef', type=float, default=0.1)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        project="euclid-q1-sae",
        config=vars(args),
        dir=str(output_dir)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = EmbeddingDataset(args.embedding_path, normalize=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = KSparseAutoencoder(
        n_dirs=args.n_dirs,
        d_model=dataset.embeddings.shape[1],
        k=args.k,
        auxk=args.k // 4,
    ).to(device)
    
    init_model(model, dataset.embeddings.to(device))
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr/10
    )
    
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(
            model,
            loader,
            optimizer,
            args.auxk_coef,
            args.multik_coef
        )
        
        scheduler.step()
        
        dead_features = (model.stats_last_nonzero > 256).sum().item()
        metrics = {
            'train_loss': train_loss,
            'dead_features': dead_features,
            'dead_feature_fraction': dead_features / model.n_dirs,
            'lr': scheduler.get_last_lr()[0]
        }
        wandb.log(metrics)
        
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
            
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {train_loss:.4f}")

if __name__ == '__main__':
    main()