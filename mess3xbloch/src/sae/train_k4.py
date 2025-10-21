#!/usr/bin/env python3
"""
Train optimized k=4 SAE with extended training (50k steps @ lr=5e-4).
Based on test_k4_with_validation.py results showing Bloch R²=0.934.
Saves the final trained SAE for feature analysis.
"""

import sys
from pathlib import Path

simplex_dir = Path(__file__).resolve().parents[2]
if str(simplex_dir) not in sys.path:
    sys.path.insert(0, str(simplex_dir))

import torch
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig
from src.processes import Mess3Process, BlochWalkProcess, JointProcess
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def train_with_validation(train_acts, val_acts, val_mess3, val_bloch,
                          encoder, decoder, bias, k, lr, n_steps, batch_size, device, eval_every=2000):
    """Train TopK SAE with periodic validation."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [bias],
        lr=lr
    )

    losses = []
    history = []
    n_samples = train_acts.shape[0]

    print(f"Training for {n_steps} steps with lr={lr}...")
    for step in range(n_steps):
        # Sample batch
        indices = torch.randperm(n_samples, device=device)[:batch_size]
        batch = train_acts[indices]

        # Forward pass
        x_centered = batch - bias
        pre_activations = encoder(x_centered)

        # TopK selection
        topk_values, topk_indices = torch.topk(pre_activations, k, dim=1)
        features = torch.zeros_like(pre_activations)
        features.scatter_(1, topk_indices, torch.relu(topk_values))

        # Reconstruction
        x_reconstructed = decoder(features) + bias

        # Loss
        loss = (batch - x_reconstructed).pow(2).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Periodic validation
        if (step + 1) % eval_every == 0 or step == 0:
            with torch.no_grad():
                # Validation reconstruction loss
                x_val_centered = val_acts - bias
                pre_acts_val = encoder(x_val_centered)
                topk_val, topk_idx_val = torch.topk(pre_acts_val, k, dim=1)
                feat_val = torch.zeros_like(pre_acts_val)
                feat_val.scatter_(1, topk_idx_val, torch.relu(topk_val))
                x_val_recon = decoder(feat_val) + bias
                val_loss = (val_acts - x_val_recon).pow(2).mean().item()

                # Bloch R² on validation set
                feat_val_np = feat_val.cpu().numpy()
                bloch_val_np = val_bloch.cpu().numpy()

                bloch_reg = LinearRegression(n_jobs=-1)
                bloch_reg.fit(feat_val_np, bloch_val_np)
                bloch_r2 = r2_score(bloch_val_np, bloch_reg.predict(feat_val_np))

            avg_train_loss = np.mean(losses[-eval_every:]) if len(losses) >= eval_every else np.mean(losses)

            history.append({
                'step': step + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'bloch_r2': bloch_r2
            })

            print(f"  Step {step+1:>6}/{n_steps}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}, Bloch R²={bloch_r2:.6f}")

    return losses, history


def main():
    device = "cuda"
    d_model = 64
    k = 4
    checkpoint_path = "../../checkpoints/checkpoint_1200000.pt"
    sae_path = "sae_topk_k4_1.2M.pt"
    output_path = "sae_topk_k4.pt"
    n_train_sequences = 2500  # 80% for training
    n_val_sequences = 500     # 20% for validation

    # Load model
    print("Loading transformer model...")
    config = HookedTransformerConfig(
        d_model=d_model, d_head=16, n_layers=4, n_ctx=8, n_heads=4,
        d_mlp=256, d_vocab=12, device=device, act_fn="relu"
    )
    model = HookedTransformer(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Initialize processes
    mess3 = Mess3Process()
    bloch = BlochWalkProcess()
    joint_process = JointProcess(mess3, bloch)

    # Generate training data
    print(f"Generating {n_train_sequences} training sequences...")
    train_sequences, _ = joint_process.generate_sequences(length=8, batch_size=n_train_sequences)
    train_tokens = train_sequences.to(device)

    with torch.no_grad():
        _, cache = model.run_with_cache(train_tokens)
        train_activations = cache["ln_final.hook_normalized"].reshape(-1, d_model)

    # Generate validation data
    print(f"Generating {n_val_sequences} validation sequences...")
    val_sequences, _ = joint_process.generate_sequences(length=8, batch_size=n_val_sequences)
    val_tokens = val_sequences.to(device)

    with torch.no_grad():
        _, cache = model.run_with_cache(val_tokens)
        val_activations = cache["ln_final.hook_normalized"].reshape(-1, d_model)

    # Get belief states for validation
    print("Computing validation belief states...")
    val_sequences_np = val_sequences.cpu().numpy()
    val_mess3_beliefs, val_bloch_beliefs = joint_process.find_belief_states(val_sequences_np)

    if hasattr(val_mess3_beliefs, 'cpu'):
        val_mess3_beliefs = val_mess3_beliefs.cpu().numpy()
    if hasattr(val_bloch_beliefs, 'cpu'):
        val_bloch_beliefs = val_bloch_beliefs.cpu().numpy()

    val_mess3_beliefs = val_mess3_beliefs[:, 1:, :]
    val_bloch_beliefs = val_bloch_beliefs[:, 1:, :]
    val_mess3_beliefs = torch.tensor(val_mess3_beliefs, dtype=torch.float32).reshape(-1, 3).to(device)
    val_bloch_beliefs = torch.tensor(val_bloch_beliefs, dtype=torch.float32).reshape(-1, 2).to(device)

    # Load baseline SAE
    print(f"\nLoading baseline k={k} SAE...")
    sae_state = torch.load(sae_path, map_location=device)

    # Initialize from baseline
    encoder = torch.nn.Linear(d_model, 512, bias=False).to(device)
    decoder = torch.nn.Linear(512, d_model, bias=False).to(device)
    encoder.load_state_dict(sae_state["encoder"])
    decoder.load_state_dict(sae_state["decoder"])
    bias = torch.nn.Parameter(sae_state["bias"].clone()).to(device)

    print("\n" + "="*60)
    print("TRAINING k=4 SAE (50k steps @ lr=5e-4)")
    print("="*60)

    losses, history = train_with_validation(
        train_activations, val_activations, val_mess3_beliefs, val_bloch_beliefs,
        encoder, decoder, bias, k,
        lr=5e-4, n_steps=50000, batch_size=128, device=device, eval_every=2000
    )

    # Save the optimized SAE
    print(f"\nSaving optimized k={k} SAE to {output_path}...")
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "bias": bias.data,
        "k": k,
        "d_model": d_model,
        "n_features": 512,
        "training_config": {
            "lr": 5e-4,
            "n_steps": 50000,
            "batch_size": 128,
            "n_train_sequences": n_train_sequences,
            "n_val_sequences": n_val_sequences,
        },
        "final_metrics": {
            "train_loss": history[-1]['train_loss'],
            "val_loss": history[-1]['val_loss'],
            "bloch_r2": history[-1]['bloch_r2'],
        },
        "training_history": history,
    }, output_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final validation Bloch R²: {history[-1]['bloch_r2']:.6f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
