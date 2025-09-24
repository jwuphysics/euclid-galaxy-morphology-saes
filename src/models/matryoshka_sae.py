"""
Matryoshka Sparse Autoencoder (SAE) implementation.

This module contains the MatryoshkaSAE model that learns a sparse, hierarchical 
dictionary of features from input embeddings. The model supports different 
dictionary sizes in a nested "matryoshka" structure.
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MatryoshkaSAE(nn.Module):
    """
    Matryoshka Sparse Autoencoder that learns a sparse, hierarchical dictionary
    of features from input embeddings.
    
    The model uses a hierarchical structure where features are organized into
    groups of increasing size, allowing for flexible feature budgets during
    inference while maintaining a shared learned representation.
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
        """
        Initialize the Matryoshka SAE.
        
        Args:
            input_dim: Dimension of input embeddings
            group_sizes: List of group sizes for hierarchical structure
            top_k: Number of top features to keep active per example
            l1_coeff: L1 regularization coefficient for sparsity
            aux_penalty: Auxiliary loss penalty for dead feature revival
            n_batches_to_dead: Number of batches before considering a feature "dead"
            aux_k_multiplier: Multiplier for auxiliary reconstruction (dead feature revival)
        """
        super().__init__()
        self.input_dim = input_dim
        self.group_sizes = group_sizes
        self.total_dict_size = sum(group_sizes)
        self.top_k = top_k
        self.l1_coeff = l1_coeff
        self.aux_penalty = aux_penalty
        self.n_batches_to_dead = n_batches_to_dead
        self.aux_k_multiplier = aux_k_multiplier

        # Decoder bias (learned data mean)
        self.b_dec = nn.Parameter(torch.zeros(self.input_dim))
        # Encoder bias
        self.b_enc = nn.Parameter(torch.zeros(self.total_dict_size))

        # Encoder and decoder weights
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.input_dim, self.total_dict_size))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.total_dict_size, self.input_dim))
        )

        # Initialize decoder weights as transpose of encoder weights (tied weights)
        with torch.no_grad():
            self.W_dec.data = self.W_enc.t().clone()
            self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True)

        # Track inactive features for dead feature revival
        self.register_buffer(
            "num_batches_not_active", torch.zeros(self.total_dict_size, dtype=torch.long)
        )
        
        # Pre-compute group indices for efficient slicing
        self.group_indices = [0] + list(np.cumsum(self.group_sizes))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Matryoshka SAE.
        
        Args:
            x: Input embeddings of shape [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
                - loss: Total training loss
                - l2_loss: Reconstruction loss  
                - l1_loss: L1 sparsity loss
                - aux_loss: Auxiliary loss for dead feature revival
                - l0_norm: Average number of active features
                - feature_acts: Sparse feature activations
                - final_reconstruction: Final reconstructed embeddings
        """
        # Center the input
        x_cent = x - self.b_dec
        
        # Compute pre-activations and apply ReLU
        pre_acts = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)

        # Apply top-k sparsity across the batch
        batch_size = x.shape[0]
        n_to_keep = self.top_k * batch_size

        # Get top-k activations across entire batch
        top_k_values, top_k_indices = torch.topk(acts.flatten(), k=n_to_keep, sorted=False)
        
        # Create sparse activation tensor
        acts_topk = torch.zeros_like(acts.flatten()).scatter_(0, top_k_indices, top_k_values)
        acts_topk = acts_topk.view_as(acts)

        # Compute hierarchical reconstructions for each matryoshka level
        intermediate_reconstructions = []
        current_reconstruction = self.b_dec.expand_as(x)
        
        for i in range(len(self.group_sizes)):
            start_idx, end_idx = self.group_indices[i], self.group_indices[i+1]
            
            # Get activations and weights for this group
            group_acts = acts_topk[:, start_idx:end_idx]
            group_W_dec = self.W_dec[start_idx:end_idx, :]
            
            # Update reconstruction incrementally
            current_reconstruction = current_reconstruction + (group_acts @ group_W_dec)
            intermediate_reconstructions.append(current_reconstruction.clone())
            
        final_reconstruction = current_reconstruction

        # Compute losses
        l2_losses = [(recon.float() - x.float()).pow(2).mean() for recon in intermediate_reconstructions]
        mean_l2_loss = torch.stack(l2_losses).mean()
        l1_loss = self.l1_coeff * torch.norm(acts_topk, p=1, dim=-1).mean()
        aux_loss = self.get_auxiliary_loss(x, final_reconstruction, acts)
        total_loss = mean_l2_loss + l1_loss + aux_loss
        
        # Compute L0 norm (number of active features)
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
        """
        Update tracking of inactive features for dead feature revival.
        
        Args:
            feature_acts: Sparse feature activations from forward pass
        """
        active_features_mask = feature_acts.sum(dim=0) > 0
        self.num_batches_not_active[~active_features_mask] += 1
        self.num_batches_not_active[active_features_mask] = 0

    def get_auxiliary_loss(
        self, x: torch.Tensor, final_reconstruction: torch.Tensor, dense_acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to encourage "dead" features to explain the residual.
        
        This mechanism helps prevent feature collapse during training by encouraging
        inactive features to learn to reconstruct the parts of the input that the
        main reconstruction misses.
        
        Args:
            x: Original input embeddings
            final_reconstruction: Reconstructed embeddings from main features
            dense_acts: Dense (non-sparse) feature activations
            
        Returns:
            Auxiliary loss tensor
        """
        dead_features_mask = self.num_batches_not_active >= self.n_batches_to_dead
        
        if not dead_features_mask.any():
            return torch.tensor(0.0, device=x.device)
            
        # Compute reconstruction residual
        residual = (x.float() - final_reconstruction.float()).detach()
        dead_acts = dense_acts[:, dead_features_mask]
        
        # Parameters for auxiliary top-k selection
        k_aux = self.top_k * self.aux_k_multiplier
        n_to_keep_aux = k_aux * x.shape[0]

        if dead_acts.numel() == 0 or n_to_keep_aux == 0:
            return torch.tensor(0.0, device=x.device)

        # Apply top-k to dead feature activations
        n_to_keep_aux = min(n_to_keep_aux, dead_acts.numel())
        top_k_aux_values, _ = torch.topk(dead_acts.flatten(), k=n_to_keep_aux)
        
        min_top_k_val = top_k_aux_values[-1]
        acts_aux = F.relu(dead_acts - min_top_k_val)
        
        # Compute auxiliary reconstruction
        recon_aux = acts_aux @ self.W_dec[dead_features_mask, :]
        aux_loss = self.aux_penalty * (recon_aux.float() - residual.float()).pow(2).mean()
        
        return aux_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """
        Normalize decoder weights and project their gradients to be orthogonal
        to the weights. This is important for training stability, ensuring
        the update step does not change the norm of the decoder weights.
        """
        if self.W_dec.grad is None:
            return
            
        # Normalize decoder weights to unit norm
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        
        # Project gradients to be orthogonal to weights
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        
        # Apply normalized weights
        self.W_dec.data = W_dec_normed

    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dense feature activations without applying top-k sparsity.
        Useful for analysis and visualization.
        
        Args:
            x: Input embeddings of shape [batch_size, input_dim]
            
        Returns:
            Dense feature activations of shape [batch_size, total_dict_size]
        """
        with torch.no_grad():
            x_cent = x - self.b_dec
            pre_acts = x_cent @ self.W_enc + self.b_enc
            return F.relu(pre_acts)