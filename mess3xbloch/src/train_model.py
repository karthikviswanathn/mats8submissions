#!/usr/bin/env python3
"""
Joint process (Mess3 × Bloch) training with W&B logging and timing analysis.

This implementation uses the seq_len+1 fix with random sampling:
- Generates sequences of length seq_len+1 (e.g., 9 tokens for seq_len=8)
- Uses random sampling (NOT exhaustive enumeration)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time

import wandb

from processes import Mess3Process, BlochWalkProcess, JointProcess
from transformer_lens import HookedTransformer, HookedTransformerConfig


def findr2_joint(joint_process, model, seq_len):
    """
    Compute R² for both Mess3 and Bloch belief state recovery.

    Args:
        joint_process: JointProcess instance
        model: Trained HookedTransformer
        seq_len: Generated sequence length (e.g., 9)

    Returns:
        dict with r2_m, r2_b, rmse_mess3, rmse_b
    """
    # Generate test sequences (seq_len tokens, but only use first n_ctx)
    sample_sequences, (seq1, seq2) = joint_process.generate_sequences(seq_len, 1000)

    with torch.no_grad():
        # Only pass first n_ctx tokens to model
        _, cache = model.run_with_cache(sample_sequences[:, :model.cfg.n_ctx])
        activations = cache["ln_final.hook_normalized"]  # (batch, n_ctx, d_model)

    # Get ground truth belief states
    belief_mess3 = joint_process.process1.find_belief_states(seq1)  # (batch, seq_len+1, 3)
    belief_bloch = joint_process.process2.find_belief_states(seq2)  # (batch, seq_len+1, 2)

    # Match activations with belief states:
    # - activations[i] = state after token i, corresponds to belief_states[i+1]
    # - We need to match the first n_ctx activations with belief_states[1:n_ctx+1]
    activations_flat = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
    belief_mess3_flat = belief_mess3[:, 1:model.cfg.n_ctx+1].reshape(-1, 3).cpu().numpy()
    belief_bloch_flat = belief_bloch[:, 1:model.cfg.n_ctx+1].reshape(-1, 2).cpu().numpy()

    assert activations_flat.shape[0] == belief_mess3_flat.shape[0], "Shape mismatch!"
    assert activations_flat.shape[0] == belief_bloch_flat.shape[0], "Shape mismatch!"

    # Fit linear regressions
    reg_mess3 = LinearRegression()
    reg_mess3.fit(activations_flat, belief_mess3_flat)
    pred_mess3 = reg_mess3.predict(activations_flat)

    reg_bloch = LinearRegression()
    reg_bloch.fit(activations_flat, belief_bloch_flat)
    pred_bloch = reg_bloch.predict(activations_flat)

    # Compute metrics
    r2_m = r2_score(belief_mess3_flat, pred_mess3)
    r2_b = r2_score(belief_bloch_flat, pred_bloch)
    rmse_mess3 = np.sqrt(np.mean((belief_mess3_flat - pred_mess3)**2))
    rmse_b = np.sqrt(np.mean((belief_bloch_flat - pred_bloch)**2))

    return {
        'r2_m': r2_m,
        'r2_b': r2_b,
        'rmse_mess3': rmse_mess3,
        'rmse_b': rmse_b,
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Joint process (Mess3 × Bloch) training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--steps', type=int, default=2000000,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--d-model', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--d-mlp', type=int, default=256,
                        help='MLP hidden dimension')
    parser.add_argument('--seq-len', type=int, default=8,
                        help='Sequence length')
    parser.add_argument('--print-every', type=int, default=2000,
                        help='Print metrics every N steps')
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='huggingface',
                        help='W&B project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='W&B run name (auto-generated if not specified)')

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize wandb if requested
    use_wandb = args.wandb
    print(f"W&B logging requested: {use_wandb}")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'steps': args.steps,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'n_layers': args.n_layers,
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'd_mlp': args.d_mlp,
                'seq_len': args.seq_len,
                'vocab_size': 12,
                'process': 'Mess3 × Bloch',
            }
        )
        print(f"W&B logging enabled: {wandb.run.url}\n")

        # Create results directory using W&B run ID
        wandb_id = wandb.run.id
        RESULTS_DIR = Path(f"models/{wandb_id}")
    else:
        # Fallback if W&B is disabled
        RESULTS_DIR = Path("models/no_wandb")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved to: {RESULTS_DIR}\n")

    print("="*70)
    print(f"JOINT PROCESS TRAINING (Mess3 × Bloch) - {args.steps} steps")
    print("="*70)
    print(f"Device:      {device}")
    print(f"Steps:       {args.steps}")
    print(f"LR:          {args.lr}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Layers:      {args.n_layers}")
    print(f"d_model:     {args.d_model}")
    print(f"Vocab size:  12 (3 × 4)")
    print()

    # Initialize joint process
    mess3 = Mess3Process(device=device)
    bloch = BlochWalkProcess(device=device)
    joint_process = JointProcess(mess3, bloch, device=device)

    # Configure model (12-token vocabulary)
    config = HookedTransformerConfig(
        d_model=args.d_model,
        d_head=args.d_model // args.n_heads,
        n_layers=args.n_layers,
        n_ctx=args.seq_len,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        d_vocab=12,  # 3 × 4
        device=device,
        act_fn="relu",
        normalization_type="LN",
        positional_embedding_type="standard",
    )

    model = HookedTransformer(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Generate sequences of length seq_len+1 for proper training
    gen_seq_len = args.seq_len + 1
    print(f"NOTE: Generating sequences of length {gen_seq_len} (seq_len+1)")
    print(f"      Context window: {args.seq_len}, Input: X[:, :{args.seq_len}], Target: Y[:, 1:]\n")

    # Compute theoretical belief loss
    print("Computing theoretical belief loss...")
    test_sequences, _ = joint_process.generate_sequences(gen_seq_len, 1000)
    belief_losses = joint_process.find_belief_loss(test_sequences, include_start=True)
    theoretical_loss = belief_losses.mean().item()
    print(f"Theoretical: {theoretical_loss:.4f} nats\n")

    # Training
    print(f"Training for {args.steps} steps...")
    model.train()

    loss_fn = nn.CrossEntropyLoss()

    # Initialize timing
    start_time = time.time()
    last_print_time = start_time

    # Timing accumulators
    time_generation = 0.0
    time_forward = 0.0
    time_backward = 0.0

    for step in range(args.steps):
        # Time: Data generation
        t0 = time.time()
        sequences, _ = joint_process.generate_sequences(gen_seq_len, args.batch_size)
        time_generation += (time.time() - t0)

        # Time: Forward pass
        t0 = time.time()
        # Use only first seq_len tokens as input
        logits, _ = model.run_with_cache(sequences[:, :args.seq_len])
        # Target: positions 1 to seq_len (next token prediction)
        loss = loss_fn(logits.reshape(-1, 12), sequences[:, 1:gen_seq_len].reshape(-1))
        time_forward += (time.time() - t0)

        # Time: Backward pass
        t0 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_backward += (time.time() - t0)

        if (step + 1) % args.print_every == 0:
            model.eval()
            metrics = findr2_joint(joint_process, model, gen_seq_len)
            model.train()

            # Calculate timing
            current_time = time.time()
            elapsed_since_last = (current_time - last_print_time) / 60  # minutes
            total_elapsed = (current_time - start_time) / 60  # minutes
            steps_since_last = args.print_every if step > 0 else 1

            # Log to wandb
            if use_wandb:
                # Calculate average times per step
                steps_done = step + 1
                log_dict = {
                    'train/loss': loss.item(),
                    'train/r2_m': metrics['r2_m'],
                    'train/r2_b': metrics['r2_b'],
                    'train/rmse_m': metrics['rmse_mess3'],
                    'train/rmse_b': metrics['rmse_b'],
                    'train/elapsed_min': total_elapsed,
                    # Timing breakdown (in seconds per step)
                    'timing/generation_per_step': time_generation / steps_done,
                    'timing/forward_per_step': time_forward / steps_done,
                    'timing/backward_per_step': time_backward / steps_done,
                }
                wandb.log(log_dict, step=step+1)

            print(f"  Step {step+1}/{args.steps}: Loss = {loss.item():.4f}, "
                  f"r2_m = {metrics['r2_m']:.3f}, r2_b = {metrics['r2_b']:.3f} | "
                  f"rmse_m = {metrics['rmse_mess3']:.4f}, rmse_b = {metrics['rmse_b']:.4f} | "
                  f"{elapsed_since_last:.2f} min for last {steps_since_last} steps ")

            last_print_time = current_time

        # Save checkpoint every 100k steps
        if (step + 1) % 100000 == 0:
            checkpoint_path = RESULTS_DIR / f"checkpoint_{step+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

            # Log checkpoint to wandb
            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"checkpoint-{step+1}",
                    type="model",
                    description=f"Model checkpoint at step {step+1}"
                )
                artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(artifact)

    # Print timing breakdown
    total_train_time = time.time() - start_time
    print("\n" + "="*70)
    print("TIMING BREAKDOWN")
    print("="*70)
    print(f"\nTotal training time: {total_train_time:.2f} seconds")
    print(f"\nPer-step averages (over {args.steps} steps):")
    print(f"  Data generation: {time_generation/args.steps*1000:.2f} ms/step ({time_generation/total_train_time*100:.1f}%)")
    print(f"  Forward pass:    {time_forward/args.steps*1000:.2f} ms/step ({time_forward/total_train_time*100:.1f}%)")
    print(f"  Backward pass:   {time_backward/args.steps*1000:.2f} ms/step ({time_backward/total_train_time*100:.1f}%)")
    other_time = total_train_time - time_generation - time_forward - time_backward
    print(f"  Other (R² eval): {other_time/args.steps*1000:.2f} ms/step ({other_time/total_train_time*100:.1f}%)")
    print(f"\nTotal breakdown:")
    print(f"  Generation: {time_generation:.2f}s")
    print(f"  Forward:    {time_forward:.2f}s")
    print(f"  Backward:   {time_backward:.2f}s")
    print(f"  Other:      {other_time:.2f}s")

    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    model.eval()

    # Validation loss
    val_sequences, _ = joint_process.generate_sequences(gen_seq_len, 1000)
    with torch.no_grad():
        val_logits, _ = model.run_with_cache(val_sequences[:, :args.seq_len])
        val_loss = loss_fn(val_logits.reshape(-1, 12), val_sequences[:, 1:gen_seq_len].reshape(-1)).item()

    print(f"\nValidation Loss:  {val_loss:.4f} nats")
    print(f"Theoretical Min:  {theoretical_loss:.4f} nats")
    print(f"Gap to Optimal:   {val_loss - theoretical_loss:.4f} nats")

    # Prediction accuracy
    test_sequences, _ = joint_process.generate_sequences(gen_seq_len, 1000)
    with torch.no_grad():
        test_logits, _ = model.run_with_cache(test_sequences[:, :args.seq_len])
        predictions = torch.argmax(test_logits, dim=-1)

    targets = test_sequences[:, 1:gen_seq_len]
    accuracy = (predictions == targets).float().mean().item()

    print(f"\nAccuracy:         {accuracy*100:.2f}%")
    print(f"Random Baseline:  8.33%")
    print(f"Improvement:      {(accuracy/0.0833 - 1)*100:.1f}%")

    # Belief state recovery
    print("\n" + "="*70)
    print("BELIEF STATE RECOVERY")
    print("="*70)

    final_metrics = findr2_joint(joint_process, model, gen_seq_len)

    print(f"\nMess3 (3-simplex):")
    print(f"  R² Score:  {final_metrics['r2_m']:.4f}")
    print(f"  RMSE:      {final_metrics['rmse_mess3']:.4f}")

    if final_metrics['r2_m'] > 0.7:
        print("  ✓✓✓ STRONG recovery")
    elif final_metrics['r2_m'] > 0.4:
        print("  ✓✓ GOOD recovery")
    elif final_metrics['r2_m'] > 0.2:
        print("  ✓ WEAK recovery")
    else:
        print("  ✗ MINIMAL recovery")

    print(f"\nBloch Walk (x-z plane):")
    print(f"  R² Score:  {final_metrics['r2_b']:.4f}")
    print(f"  RMSE:      {final_metrics['rmse_b']:.4f}")

    if final_metrics['r2_b'] > 0.7:
        print("  ✓✓✓ STRONG recovery")
    elif final_metrics['r2_b'] > 0.4:
        print("  ✓✓ GOOD recovery")
    elif final_metrics['r2_b'] > 0.2:
        print("  ✓ WEAK recovery")
    else:
        print("  ✗ MINIMAL recovery")

    # Log final metrics to wandb
    if use_wandb:
        wandb.log({
            'final/val_loss': val_loss,
            'final/theoretical_loss': theoretical_loss,
            'final/loss_gap': val_loss - theoretical_loss,
            'final/accuracy': accuracy,
            'final/r2_m': final_metrics['r2_m'],
            'final/r2_b': final_metrics['r2_b'],
            'final/rmse_mess3': final_metrics['rmse_mess3'],
            'final/rmse_b': final_metrics['rmse_b'],
        })

    # Save final model
    final_model_path = RESULTS_DIR / f"final_model_{args.steps}.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")

    # Log final model artifact to wandb
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"final-model-{args.steps}",
            type="model",
            description=f"Final joint process model after {args.steps} steps"
        )
        artifact.add_file(str(final_model_path))
        wandb.log_artifact(artifact)

    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY ({args.steps} steps)")
    print("="*70)
    print(f"Loss Gap:       {val_loss - theoretical_loss:.4f} nats")
    print(f"Accuracy:       {accuracy*100:.2f}% (random: 8.33%)")
    print(f"R² Mess3:       {final_metrics['r2_m']:.4f}")
    print(f"R² Bloch:       {final_metrics['r2_b']:.4f}")

    # Overall assessment
    both_good = final_metrics['r2_m'] > 0.7 and final_metrics['r2_b'] > 0.7

    success_level = 0
    if both_good:
        print("\n✓✓✓ SUCCESS: Both geometries recovered!")
        success_level = 3
    elif final_metrics['r2_m'] > 0.4 or final_metrics['r2_b'] > 0.4:
        print("\n✓ PARTIAL: Some geometry recovery, needs more training")
        success_level = 1
    else:
        print("\n⚠ LOW R²: Geometries not well recovered")
        print("  Possible issues:")
        print("  - Need more training steps")
        print("  - Model capacity too small")
        print("  - Learning rate needs tuning")
        success_level = 0

    if use_wandb:
        wandb.log({'final/success_level': success_level})
        wandb.finish()

    print("="*70)


if __name__ == "__main__":
    main()
