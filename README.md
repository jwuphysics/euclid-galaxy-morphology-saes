# Euclid Q1 Galaxy Feature Atlas: Zoobot and Matryoshka SAEs

## Overview

This project uses sparse autoencoders (SAEs) to identify learned features in Zoobot, a neural network trained to classify galaxy morphology. 

All data are loaded directly from the Euclid Q1 [GalaxyZoo Huggingface dataset](https://huggingface.co/datasets/mwalmsley/gz_euclid) (see the general framework [here](https://github.com/mwalmsley/galaxy-datasets)). We use the [Zoobot-Euclid encoder model](https://huggingface.co/mwalmsley/zoobot-encoder-euclid) by Mike Walmsley et al; read more about this work on [arXiv](https://arxiv.org/abs/2503.15310).


## What's in this repository

- `src/create_embeddings.py` - Extracts feature vectors from Euclid Q1 galaxy images using Zoobot encoder model
- `src/train_sae.py` - Trains a k-sparse autoencoder to learn interpretable features from these embeddings
- `src/train_matryoshka_sae.py` - Trains a Matryoshka SAE that can operate at different feature budget levels
- `src/analyze_sae_features.py` - Visualizes which galaxy images most strongly activate specific learned features
- `src/analyze_pca.py` - PCA baseline analysis for comparison with SAE features


## Running the code

It's easiest to set up the environment with `uv sync`. You can then activate the venv and then run the commands below, or e.g. use `uv run src/create_embeddings.py`.

First, extract embeddings from the Huggingface dataset (this will download `mwalmsley/gz_euclid` automatically):
```bash
python src/create_embeddings.py
```

Next, train the SAE and analyze features. For the standard k-sparse SAE:

```bash
python src/train_sae.py

# then examine top features
python src/analyze_sae_features.py
```

For the Matryoshka SAE:
```bash
python src/train_matryoshka_sae.py --train
```

The Matryoshka SAE script handles feature visualization automatically. Remove `--train` to only run evaluation.

Customize training with config files:
```bash
python src/train_matryoshka_sae.py --train --config my_config.toml
```

## Baseline Comparison

For comparison with the SAE features, we can also run a principal components analysis (PCA) baseline on the same embeddings:

```bash
python src/analyze_pca.py
```

## Results

Check out the results in `results/sae`, `results/pca`, and `results/matryoshka_sae`.
