# Euclid Q1 Galaxy Feature Atlas: Zoobot and Matryoshka SAEs

## Overview

This project uses sparse autoencoders (SAEs) to identify learned features in galaxy embeddings from both supervised (Zoobot) and self-supervised (MAE) models. The analysis compares SAE features with PCA baselines against Galaxy Zoo morphology labels to understand what features are learned and which are novel.

All data are loaded directly from the Euclid Q1 [GalaxyZoo Huggingface dataset](https://huggingface.co/datasets/mwalmsley/gz_euclid) (see the general framework [here](https://github.com/mwalmsley/galaxy-datasets)). We use the [Zoobot-Euclid encoder model](https://huggingface.co/mwalmsley/zoobot-encoder-euclid) by Mike Walmsley et al; read more about this work on [arXiv](https://arxiv.org/abs/2503.15310).

## Repository Structure

```
src/
├── models/                     # Shared model definitions
│   ├── matryoshka_sae.py      # MatryoshkaSAE implementation
│   └── __init__.py
├── utils/                      # Shared utilities
│   ├── reproducibility.py     # Random seed management
│   ├── data_loading.py        # Dataset and DataLoader utilities
│   ├── visualization.py       # Plotting and visualization functions
│   └── __init__.py
├── config/                     # Configuration files
│   ├── matryoshka_sae_config.toml      # Supervised SAE config
│   └── mae_matryoshka_sae_config.toml  # Self-supervised SAE config
├── create_embeddings.py       # Extract embeddings from Zoobot encoder
├── train_matryoshka_sae.py    # Train supervised Matryoshka SAE
├── train_matryoshka_sae_ssl.py # Train self-supervised Matryoshka SAE
├── analyze_sae_features.py    # Visualize SAE feature activations
├── analyze_pca.py             # PCA baseline analysis
└── make_figures.py             # Generate publication figures
```

## Analysis Pipeline

### Supervised Analysis (Zoobot embeddings)
1. **Extract embeddings**: `uv run src/create_embeddings.py`
2. **Train SAE**: `uv run src/train_matryoshka_sae.py --train --config src/config/matryoshka_sae_config.toml`
3. **Analyze features**: `uv run src/analyze_sae_features.py`
4. **PCA baseline**: `uv run src/analyze_pca.py`

### Self-supervised Analysis (MAE embeddings)
1. **Train SSL SAE**: `uv run src/train_matryoshka_sae_ssl.py --train --config src/config/mae_matryoshka_sae_config.toml`
2. **Generate figures**: `uv run src/make_figures.py`

## Getting Started

### Environment Setup
```bash
uv sync
```

### Quick Start - Supervised Analysis
1. Extract embeddings from Zoobot encoder:
   ```bash
   uv run src/create_embeddings.py
   ```

2. Train Matryoshka SAE:
   ```bash
   uv run src/train_matryoshka_sae.py --train
   ```

3. Analyze learned features:
   ```bash
   uv run src/analyze_sae_features.py
   ```

4. Run PCA baseline:
   ```bash
   uv run src/analyze_pca.py
   ```

### Self-supervised Analysis
Train on MAE embeddings with Galaxy Zoo crossmatch:
```bash
uv run src/train_matryoshka_sae_ssl.py --train
```

### Generate Publication Figures
Create comparison plots between supervised/SSL and SAE/PCA methods:
```bash
uv run src/make_figures.py
```

## Configuration

Training parameters can be customized in TOML config files:
- `src/config/matryoshka_sae_config.toml` - Supervised analysis settings
- `src/config/mae_matryoshka_sae_config.toml` - Self-supervised analysis settings

## Key Features

- **Reproducible**: All scripts use consistent random seed initialization
- **Modular**: Shared utilities for data loading, visualization, and model definitions
- **Configurable**: TOML-based configuration for easy hyperparameter tuning
- **Scientific**: Compares learned representations against Galaxy Zoo morphology labels

## Results

Results are organized in the `results/` directory:
- `results/matryoshka_sae/` - Supervised SAE results
- `results/mae_matryoshka_sae/` - Self-supervised SAE results  
- `results/pca/` - PCA baseline results
- `results/figures/` - Publication-ready comparison figures

## Reproducibility

All scripts initialize random seeds consistently to ensure reproducible results. The codebase uses:
- Fixed random seeds for PyTorch, NumPy, and Python's random module
- Deterministic CUDA operations when available
- Reproducible DataLoader worker initialization

This ensures that scientific results can be reliably reproduced across different runs and environments.