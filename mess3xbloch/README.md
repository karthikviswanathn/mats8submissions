# MATS 9.0 Simplex Application - Code Submission

This repository contains the code for training and analyzing transformers on the joint Mess3 × Bloch Walk process.

## Core Research Question

When a transformer is trained on the Cartesian product of two independent stochastic processes (Mess3 and Bloch Walk), do the residual stream activations decompose into orthogonal subspaces representing the two independent belief geometries?

## Repository Structure

```
submission/
├── README.md                          # This file
├── analysis.pdf                       # Full writeup
├── src/                               # Source code
│   ├── processes.py                   # Mess3, BlochWalk, JointProcess implementations
│   ├── train_model.py                 # Training script for joint process
│   └── sae/                           # Sparse autoencoder analysis
│       ├── train_k4.py                # Train k=4 TopK SAE
│       ├── analyze_k4_features.py     # Analyze SAE feature specialization
│       └── sae_topk_k4.pt             # Trained k=4 SAE checkpoint
└── checkpoints/                       # Trained model checkpoints
    ├── checkpoint_100000.pt
    ├── checkpoint_200000.pt
    ...
    └── checkpoint_1200000.pt
```

## Training the Joint Process Model

Train a transformer on the Mess3 × Bloch Walk joint process:

```bash
cd src
python train_model.py --steps 1200000 --lr 1e-4 --batch-size 128
```

Key arguments:
- `--steps`: Number of training steps (default: 2000000)
- `--lr`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size (default: 128)
- `--n-layers`: Number of transformer layers (default: 4)
- `--d-model`: Model dimension (default: 64)
- `--seq-len`: Sequence length (default: 8)
- `--wandb`: Enable Weights & Biases logging (default: True)

The script trains a transformer with 12-token vocabulary (3 Mess3 tokens × 4 Bloch tokens) and saves checkpoints every 100k steps.

## Training the k=4 SAE

Train a TopK sparse autoencoder with k=4 on residual stream activations:

```bash
cd src/sae
python train_k4.py
```

This trains a 512-feature SAE on activations from the trained transformer checkpoint.

## Analyzing SAE Features

Analyze the learned SAE features to examine specialization:

```bash
cd src/sae
python analyze_k4_features.py
```

This script:
- Identifies monosemantic Mess3 features (selective for individual states)
- Identifies Bloch-selective features (correlated with x or z coordinates)
- Tests for mixed features (selective for both processes)

## Key Results

The trained transformer achieves:
- **Mess3 R² > 0.9**: Strong linear recovery of Mess3 belief states
- **Bloch R² > 0.9**: Strong linear recovery of Bloch Walk belief states
- **Orthogonality**: The two belief geometries occupy orthogonal subspaces in the residual stream

The k=4 SAE learns:
- **3 pure Mess3 features**: One per state (monosemantic)
- **8 pure Bloch features**: Encoding x and z coordinates
- **0 mixed features**: Perfect orthogonal separation

## Dependencies

- PyTorch
- TransformerLens
- scikit-learn
- NumPy
- Weights & Biases (optional, for logging)

## Reference

Full analysis and results are provided in `analysis.pdf`.
