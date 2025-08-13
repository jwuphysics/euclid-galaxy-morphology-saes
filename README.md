# Euclid Q1 Galaxy Feature Atlas: Zoobot and Matryoshka SAEs

## Overview

This project uses sparse autoencoders (SAEs) to identify learned features in Zoobot, a neural network trained to classify galaxy morphology. 

All data used here are from the Euclid Q1 release and can be found on [Huggingface](https://huggingface.co/datasets/mwalmsley/gz_euclid) (see the general framework [here](https://github.com/mwalmsley/galaxy-datasets)). We use the [Zoobot-Euclid encoder model](https://huggingface.co/mwalmsley/zoobot-encoder-euclid) by Mike Walmsley et al; read more about this work on [arXiv](https://arxiv.org/abs/2503.15310).


## What's in this repository

- `src/create_embeddings.py` - Extracts feature vectors from Euclid Q1 galaxy images using Zoobot encoder model
- `src/train_sae.py` - Trains a k-sparse autoencoder to learn interpretable features from these embeddings
- `src/train_matryoshka_sae.py` - Trains a Matryoshka SAE that can operate at different feature budget levels
- `src/analyze_features.py` - Visualizes which galaxy images most strongly activate specific learned features
- `src/analyze_pca.py` - PCA baseline analysis for comparison with SAE features


## Running the code

It's easiest to set up the environment with `uv sync`. You can then activate the venv and then run the commands below, or e.g. use `uv run src/create_embeddings.py`.

First, we need to extract galaxy image embeddings from Zoobot.
```bash
python src/create_embeddings.py
```

Next, we can train the SAE to consolidate features and examine them. For the standard k-sparse SAE, you can train the SAE and identify top activations like so:

```bash
python src/train_sae.py

# and then examine top features
python src/analyze_sae_features.py
```

If you want to train a Matryoshka SAE (which performs worse, unfortunately), then you can run
```bash
python src/train_matryoshka_sae.py --train
```

Note that the Matryoshak SAE script also extracts features in a single go. You can remove the `--train` option if you just want to reprocess features.

It's possible to customize the training by supplying a config file or `matryoshka_sae_config.toml`:
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