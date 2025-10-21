#!/usr/bin/env python3
"""
Detailed feature analysis for k=4 TopK SAE.
Examines which features are monosemantic for Mess3 states,
Bloch-selective, and how features specialize.
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


def analyze_features(sae_path, checkpoint_path, n_sequences=4000, device="cuda"):
    """Analyze k=4 SAE features."""

    d_model = 64
    k = 4

    # Load transformer model
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
    print("Initializing processes...")
    mess3 = Mess3Process()
    bloch = BlochWalkProcess()
    joint_process = JointProcess(mess3, bloch)

    # Generate sequences
    print(f"Generating {n_sequences} sequences...")
    sequences, _ = joint_process.generate_sequences(length=8, batch_size=n_sequences)
    tokens = sequences.to(device)

    # Extract activations
    print("Extracting activations...")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        activations = cache["ln_final.hook_normalized"]
        activations = activations.reshape(-1, d_model)

    # Get belief states
    print("Computing belief states...")
    sequences_np = sequences.cpu().numpy()
    mess3_beliefs, bloch_beliefs = joint_process.find_belief_states(sequences_np)

    # Convert to tensors
    if hasattr(mess3_beliefs, 'cpu'):
        mess3_beliefs = mess3_beliefs.cpu().numpy()
    if hasattr(bloch_beliefs, 'cpu'):
        bloch_beliefs = bloch_beliefs.cpu().numpy()

    mess3_beliefs = mess3_beliefs[:, 1:, :]
    bloch_beliefs = bloch_beliefs[:, 1:, :]
    mess3_beliefs = torch.tensor(mess3_beliefs, dtype=torch.float32).reshape(-1, 3).to(device)
    bloch_beliefs = torch.tensor(bloch_beliefs, dtype=torch.float32).reshape(-1, 2).to(device)

    # Load SAE
    print(f"\nLoading k={k} SAE from {sae_path}...")
    sae_state = torch.load(sae_path, map_location=device, weights_only=False)

    encoder = torch.nn.Linear(d_model, 512, bias=False).to(device)
    decoder = torch.nn.Linear(512, d_model, bias=False).to(device)
    encoder.load_state_dict(sae_state["encoder"])
    decoder.load_state_dict(sae_state["decoder"])
    bias = sae_state["bias"].to(device)

    # Extract features
    print("Extracting SAE features...")
    with torch.no_grad():
        x_centered = activations - bias
        pre_activations = encoder(x_centered)

        # TopK selection
        topk_values, topk_indices = torch.topk(pre_activations, k, dim=1)
        features = torch.zeros_like(pre_activations)
        features.scatter_(1, topk_indices, torch.relu(topk_values))

    features_np = features.cpu().numpy()
    mess3_np = mess3_beliefs.cpu().numpy()
    bloch_np = bloch_beliefs.cpu().numpy()
    tokens_np = tokens.cpu().numpy().reshape(-1)

    n_features = features_np.shape[1]

    print("\n" + "="*70)
    print(f"K={k} SAE FEATURE ANALYSIS")
    print("="*70)

    # 1. Feature activation statistics
    print("\n1. FEATURE ACTIVATION STATISTICS")
    print("-" * 70)

    feature_activation_counts = (features_np > 0).sum(axis=0)
    active_features = np.where(feature_activation_counts > 0)[0]
    n_active = len(active_features)
    n_dead = n_features - n_active

    print(f"Total features: {n_features}")
    print(f"Active features (ever fire): {n_active} ({100*n_active/n_features:.1f}%)")
    print(f"Dead features (never fire): {n_dead} ({100*n_dead/n_features:.1f}%)")

    # For active features, compute statistics
    mean_activation = features_np[:, active_features].mean(axis=0)
    max_activation = features_np[:, active_features].max(axis=0)
    activation_frequency = (features_np[:, active_features] > 0).mean(axis=0)

    print(f"\nActive feature statistics:")
    print(f"  Mean activation: {mean_activation.mean():.4f} ± {mean_activation.std():.4f}")
    print(f"  Max activation: {max_activation.mean():.4f} ± {max_activation.std():.4f}")
    print(f"  Activation frequency: {activation_frequency.mean():.4f} ± {activation_frequency.std():.4f}")

    # 2. Mess3 monosemantic features
    print("\n2. MESS3 MONOSEMANTIC FEATURES")
    print("-" * 70)

    mess3_states = mess3_np.argmax(axis=1)
    mess3_monosemantic_features = []

    for feat_idx in active_features:
        selectivities = []
        for state in range(3):
            mask = (mess3_states == state)
            if mask.sum() > 0:
                mean_when = features_np[mask, feat_idx].mean()
                mean_when_not = features_np[~mask, feat_idx].mean()
                selectivities.append(mean_when - mean_when_not)

        selectivities = np.array(selectivities)
        # Monosemantic: selective for exactly one state with threshold > 0.3
        if selectivities.max() > 0.3 and (selectivities > 0.3).sum() == 1:
            preferred_state = selectivities.argmax()
            mess3_monosemantic_features.append({
                'feature': feat_idx,
                'state': preferred_state,
                'selectivity': selectivities[preferred_state],
                'selectivities': selectivities
            })

    print(f"Mess3 monosemantic features: {len(mess3_monosemantic_features)} / {n_active}")

    if mess3_monosemantic_features:
        print("\nMess3 monosemantic feature details:")
        for info in sorted(mess3_monosemantic_features, key=lambda x: -x['selectivity']):
            print(f"  Feature {info['feature']:>3}: State {info['state']} "
                  f"(selectivity={info['selectivity']:.3f}, "
                  f"all=[{info['selectivities'][0]:.3f}, {info['selectivities'][1]:.3f}, {info['selectivities'][2]:.3f}])")

    # 3. Bloch-selective features
    print("\n3. BLOCH-SELECTIVE FEATURES")
    print("-" * 70)

    bloch_selective_features = []

    for feat_idx in active_features:
        feat_acts = features_np[:, feat_idx]

        # Correlations with x and z
        corr_x = np.corrcoef(feat_acts, bloch_np[:, 0])[0, 1]
        corr_z = np.corrcoef(feat_acts, bloch_np[:, 1])[0, 1]

        # Consider selective if |correlation| > 0.3
        if abs(corr_x) > 0.3 or abs(corr_z) > 0.3:
            bloch_selective_features.append({
                'feature': feat_idx,
                'corr_x': corr_x,
                'corr_z': corr_z,
                'max_corr': max(abs(corr_x), abs(corr_z)),
                'coord': 'x' if abs(corr_x) > abs(corr_z) else 'z'
            })

    print(f"Bloch-selective features (|corr| > 0.3): {len(bloch_selective_features)} / {n_active}")

    if bloch_selective_features:
        print("\nBloch-selective feature details:")
        for info in sorted(bloch_selective_features, key=lambda x: -x['max_corr']):
            print(f"  Feature {info['feature']:>3}: {info['coord']} "
                  f"(corr_x={info['corr_x']:+.3f}, corr_z={info['corr_z']:+.3f})")

    # 4. Feature specialization
    print("\n4. FEATURE SPECIALIZATION")
    print("-" * 70)

    mess3_feature_set = set(f['feature'] for f in mess3_monosemantic_features)
    bloch_feature_set = set(f['feature'] for f in bloch_selective_features)

    pure_mess3 = mess3_feature_set - bloch_feature_set
    pure_bloch = bloch_feature_set - mess3_feature_set
    mixed = mess3_feature_set & bloch_feature_set
    neither = set(active_features) - mess3_feature_set - bloch_feature_set

    print(f"Pure Mess3 features: {len(pure_mess3)} (selective only for Mess3 states)")
    print(f"Pure Bloch features: {len(pure_bloch)} (selective only for Bloch coords)")
    print(f"Mixed features: {len(mixed)} (selective for both)")
    print(f"Neither: {len(neither)} (not strongly selective for either)")

    if pure_mess3:
        print(f"\nPure Mess3 features: {sorted(pure_mess3)}")
    if pure_bloch:
        print(f"Pure Bloch features: {sorted(pure_bloch)}")
    if mixed:
        print(f"Mixed features: {sorted(mixed)}")

    # 5. Position selectivity
    print("\n5. POSITION SELECTIVITY")
    print("-" * 70)

    sequences_reshaped = tokens_np.reshape(n_sequences, 8)
    features_reshaped = features_np.reshape(n_sequences, 8, n_features)

    position_selective_features = []

    for feat_idx in active_features:
        position_activations = features_reshaped[:, :, feat_idx]
        position_means = position_activations.mean(axis=0)
        position_stds = position_activations.std(axis=0)

        # Check if feature is position-selective (high variance across positions)
        if position_means.max() > 2 * position_means.mean():
            preferred_pos = position_means.argmax()
            position_selective_features.append({
                'feature': feat_idx,
                'position': preferred_pos,
                'mean_at_pos': position_means[preferred_pos],
                'overall_mean': position_means.mean()
            })

    print(f"Position-selective features: {len(position_selective_features)} / {n_active}")

    if position_selective_features:
        print("\nTop position-selective features:")
        for info in sorted(position_selective_features, key=lambda x: -x['mean_at_pos'])[:10]:
            print(f"  Feature {info['feature']:>3}: Position {info['position']} "
                  f"(mean={info['mean_at_pos']:.3f} vs overall={info['overall_mean']:.3f})")

    # 6. Token preferences
    print("\n6. TOKEN PREFERENCES")
    print("-" * 70)

    # For top active features, show token preferences
    top_features = active_features[np.argsort(-mean_activation)][:min(10, len(active_features))]

    print(f"Analyzing top {len(top_features)} most active features:")

    for feat_idx in top_features:
        token_activations = {}
        for token in range(12):
            mask = (tokens_np == token)
            if mask.sum() > 0:
                token_activations[token] = features_np[mask, feat_idx].mean()

        # Decode tokens to (mess3, bloch) pairs
        top_tokens = sorted(token_activations.items(), key=lambda x: -x[1])[:3]

        print(f"\n  Feature {feat_idx}:")
        for token, activation in top_tokens:
            mess3_tok = token // 4
            bloch_tok = token % 4
            print(f"    Token {token:>2} (M{mess3_tok}/B{bloch_tok}): {activation:.3f}")

    # 7. Summary comparison with other k values
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    print(f"k={k} SAE:")
    print(f"  Active features: {n_active} / {n_features}")
    print(f"  Mess3 monosemantic: {len(mess3_monosemantic_features)}")
    print(f"  Bloch-selective: {len(bloch_selective_features)}")
    print(f"  Pure Mess3: {len(pure_mess3)}")
    print(f"  Pure Bloch: {len(pure_bloch)}")
    print(f"  Mixed: {len(mixed)}")

    print("\nPrevious results for comparison:")
    print("  k=2:   Active=9,   Mono=8,   Bloch=? (not measured)")
    print("  k=16:  Active=167, Mono=31,  Bloch=17")
    print("  k=256: Active=492, Mono=175, Bloch=52")

    # 8. Check if saved metrics match
    if 'final_metrics' in sae_state:
        print("\n" + "="*70)
        print("SAVED METRICS (from training)")
        print("="*70)
        metrics = sae_state['final_metrics']
        print(f"Final validation Bloch R²: {metrics['bloch_r2']:.6f}")
        print(f"Final validation loss: {metrics['val_loss']:.6f}")
        print(f"Final training loss: {metrics['train_loss']:.6f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-path", type=str,
                       default="sae_topk_k4.pt",
                       help="Path to k=4 SAE checkpoint")
    parser.add_argument("--model-path", type=str,
                       default="../../checkpoints/checkpoint_1200000.pt",
                       help="Path to transformer checkpoint")
    parser.add_argument("--n-sequences", type=int, default=4000,
                       help="Number of sequences to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    args = parser.parse_args()

    analyze_features(args.sae_path, args.model_path, args.n_sequences, args.device)


if __name__ == "__main__":
    main()
