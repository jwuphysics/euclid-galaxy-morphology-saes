# Re-envisioning Euclid Galaxy Morphology: Identifying and Interpreting Features with Sparse Autoencoders

## Overview

We use sparse autoencoders (SAEs) to identify learned features in galaxy embeddings from both supervised (Zoobot) and self-supervised (MAE) models. We compare SAE features with PCA baselines against Galaxy Zoo morphology labels in order to interpret them + examine novelty. 

All supervised data are loaded directly from the Euclid Q1 [GalaxyZoo Huggingface dataset](https://huggingface.co/datasets/mwalmsley/gz_euclid) (see the general framework [here](https://github.com/mwalmsley/galaxy-datasets)). We use the [Zoobot-Euclid encoder model](https://huggingface.co/mwalmsley/zoobot-encoder-euclid) by Mike Walmsley et al; read more about this work on [arXiv](https://arxiv.org/abs/2503.15310).

The self-supervised model is a [ViT trained via masked autoencoder reconstruction](https://huggingface.co/mwalmsley/euclid_encoder_mae_zoobot_vit_small_patch8_224), originating from the larger Euclid [Euclid "RR2" dataset](https://huggingface.co/mwalmsley/euclid-rr2-mae). Check out an *interactive demo* of masked reconstruction [here](https://huggingface.co/spaces/mwalmsley/euclid_masked_autoencoder).

Or skim this poster, which accompanies our accepted NeurIPS ML4PS workshop paper:
![](poster.png)

## Repository Structure

```
src/
├── models/
│   ├── matryoshka_sae.py
│   └── __init__.py
├── utils/
│   ├── reproducibility.py
│   ├── data_loading.py
│   ├── visualization.py
│   └── __init__.py
├── config/
│   ├── matryoshka_sae_config.toml
│   └── mae_matryoshka_sae_config.toml
├── create_embeddings.py
├── create_mae_embeddings.py
├── train_matryoshka_sae.py
├── train_matryoshka_sae_ssl.py
├── analyze_sae_features.py
├── analyze_pca.py
└── make_figures.py
```

## Quick Start

### Supervised Zoobot embeddings
1. Extract embeddings by running `uv run src/create_embeddings.py`
2. Train SAE via `uv run src/train_matryoshka_sae.py --train --config src/config/matryoshka_sae_config.toml`
3. Plot features using `uv run src/analyze_sae_features.py`
4. Compare against PCA baseline (+plotting) via `uv run src/analyze_pca.py`

### Self-supervised (MAE) embeddings
1. Train SSL SAE via `uv run src/train_matryoshka_sae_ssl.py --train --config src/config/mae_matryoshka_sae_config.toml`
2. Extract MAE embeddings for analysis by running `uv run src/create_mae_embeddings.py`
3. Generate PCA analysis on MAE embeddings via
    ```
    uv run src/analyze_pca.py \
      --embedding_path results/mae_analysis/mae_test_embeddings.npy \
      --embedding_ids_path results/mae_analysis/mae_test_ids.npy \
      --output_dir results/mae_analysis/pca \
      --n_components 64 
   ```
5. Generate figures with `uv run src/make_figures.py`

## Slightly more detailed setup...

You can load the virtual Python environment with `uv sync`.

### Extracting features from supervised models
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

### Extracting features from the self-supervised model
1. Train on MAE embeddings with Galaxy Zoo crossmatch:
   ```bash
   uv run src/train_matryoshka_sae_ssl.py --train
   ```

2. Extract MAE embeddings for analysis:
   ```bash
   uv run src/create_mae_embeddings.py
   ```

3. Generate PCA baseline on MAE embeddings:
   ```bash
   uv run src/analyze_pca.py --embedding_path results/mae_analysis/mae_test_embeddings.npy --embedding_ids_path results/mae_analysis/mae_test_ids.npy --output_dir results/mae_analysis/pca --n_components 64
   ```

### Generate figures 
Create comparison plots between supervised/SSL and SAE/PCA methods:
```bash
uv run src/make_figures.py
```

## Configuration

Training parameters can be customized in TOML config files:
- `src/config/matryoshka_sae_config.toml` - Supervised analysis settings
- `src/config/mae_matryoshka_sae_config.toml` - Self-supervised analysis settings

## Results

Results are organized in the `results/` directory:
- `results/sae/` - Supervised SAE results
- `results/mae_matryoshka_sae/` - Self-supervised SAE results  
- `results/pca/` - PCA baseline results (supervised)
- `results/mae_analysis/` - MAE embeddings and PCA baseline results (SSL)
- `results/figures/` - Publication-ready comparison figures
