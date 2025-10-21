"""
Stochastic process implementations for the simplex experiment.

This module implements:
- StochasticProcess: Abstract base class
- Mess3Process: Classical 3-state HMM
- BlochWalkProcess: Quantum process on Bloch sphere
- JointProcess: Cartesian product of two independent processes
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class StochasticProcess(ABC):
    """Abstract base class for stochastic processes."""

    def __init__(self, vocab_size: int, device: str = 'cuda'):
        self.vocab_size = vocab_size
        self.device = device

    @abstractmethod
    def generate_sequences(self, length: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences from the process.

        Returns:
            sequences: Tensor of shape (batch_size, length) containing tokens
            states: Tensor of shape (batch_size, length) containing hidden states
        """
        pass

    @abstractmethod
    def find_belief_states(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute belief states for given sequences.

        Args:
            sequences: Tensor of shape (batch_size, length)

        Returns:
            belief_states: Tensor of shape (batch_size, length+1, belief_dim)
        """
        pass

    @abstractmethod
    def find_belief_loss(self, sequences: torch.Tensor, include_start: bool = True) -> torch.Tensor:
        """
        Compute loss for predicting next token from belief state.

        Args:
            sequences: Tensor of shape (batch_size, length)
            include_start: Whether to include the first token in loss

        Returns:
            loss: Tensor of shape (batch_size, length) or (batch_size, length-1)
        """
        pass


class Mess3Process(StochasticProcess):
    """
    Classical 3-state Hidden Markov Model.

    Based on arXiv:2405.15943
    """

    def __init__(self, x: float = 0.05, alpha: float = 0.85, device: str = 'cuda'):
        super().__init__(vocab_size=3, device=device)
        self.x = x
        self.alpha = alpha
        self.T = self.get_transition_matrix()
        # For a generic process, use the stationary distribution as prior
        self.prior = torch.ones(3, device=device) / 3

    def get_transition_matrix(self) -> torch.Tensor:
        """
        Construct transition matrix T[source, target, token].

        T[s, t, x] = P(hidden: s -> t) * P(emit: x | t)
        """
        x = self.x
        alpha = self.alpha
        T = torch.zeros((3, 3, 3))  # Shape: [source, target, token]

        for source in range(3):
            for target in range(3):
                for token in range(3):
                    # p(source -> target)
                    p_source_target = 1 - 2 * x if source == target else x
                    # p(token | target)
                    p_token = alpha if token == target else (1 - alpha) / 2
                    T[source, target, token] = p_source_target * p_token

        return T.to(self.device)

    def find_next_state(self, current_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next states and emissions.

        Args:
            current_states: Tensor of shape (B,) with integer states (0, 1, or 2)

        Returns:
            next_states: Tensor of shape (B,)
            tokens: Tensor of shape (B,)
        """
        B = current_states.shape[0]
        # For each state in the batch, select the [target x token] matrix
        probs = self.T[current_states]  # shape: (B, 3, 3)
        probs = probs.view(B, -1)       # flatten to (B, 9)

        # Sample an index for each element in the batch
        idx = torch.multinomial(probs, num_samples=1).squeeze(1)  # shape: (B,)
        next_states = idx // 3
        tokens = idx % 3

        return next_states, tokens

    def generate_sequences(self, length: int = 10, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences from the Mess3 process."""
        sequences = []   # to collect tokens from each time step
        states_seq = []  # to collect next states

        # Initialize starting states
        current_states = torch.randint(0, 3, (batch_size,), device=self.device)

        for _ in range(length):
            next_states, tokens = self.find_next_state(current_states)
            sequences.append(tokens)
            states_seq.append(next_states)
            current_states = next_states

        # Stack along the time dimension: (B, L)
        sequences = torch.stack(sequences, dim=1)
        states_seq = torch.stack(states_seq, dim=1)

        return sequences, states_seq

    def update_belief_state(self, current_belief: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Update belief state given observed tokens.

        Args:
            current_belief: Tensor of shape (B, 3)
            tokens: Tensor of shape (B,)

        Returns:
            next_belief: Tensor of shape (B, 3)
        """
        # Rearrange T: (token, source, target)
        T_batch = self.T.permute(2, 0, 1)[tokens]  # shape: (B, 3, 3)
        next_belief = torch.bmm(current_belief.unsqueeze(1), T_batch).squeeze(1)  # shape: (B, 3)

        # Normalize
        next_belief = next_belief / next_belief.sum(dim=1, keepdim=True)
        return next_belief

    def find_belief_states(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute belief states for sequences.

        Args:
            sequences: Tensor of shape (B, L)

        Returns:
            belief_states: Tensor of shape (B, L+1, 3)
        """
        B, L = sequences.shape
        current_belief = self.prior.unsqueeze(0).expand(B, -1)  # shape: (B, 3)
        belief_states = [current_belief]

        for t in range(L):
            current_belief = self.update_belief_state(current_belief, sequences[:, t])
            belief_states.append(current_belief)

        # Stack along time dimension: (B, L+1, 3)
        belief_states = torch.stack(belief_states, dim=1)
        return belief_states

    def find_belief_loss(self, sequences: torch.Tensor, include_start: bool = True) -> torch.Tensor:
        """
        Compute belief loss (negative log probability of observed tokens).

        Args:
            sequences: Tensor of shape (B, L)
            include_start: Whether to include the first token

        Returns:
            loss: Tensor of shape (B, L) or (B, L-1)
        """
        B, L = sequences.shape
        belief_states = self.find_belief_states(sequences)  # shape: (B, L+1, 3)

        # Sum T over target dimension: resulting shape (3, 3)
        transition_matrix = self.T.sum(dim=1)  # sum over target states

        # Compute predicted next state distribution
        predicted = torch.bmm(
            belief_states[:, :-1, :],  # (B, L, 3)
            transition_matrix.expand(B, -1, -1)  # (B, 3, 3)
        )  # shape: (B, L, 3)

        # Gather probability of actual observed token
        token_probs = predicted.gather(dim=2, index=sequences.unsqueeze(-1)).squeeze(-1)  # shape: (B, L)
        empirical_loss = -torch.log(token_probs)

        start_index = 0 if include_start else 1
        return empirical_loss[:, start_index:]


class BlochWalkProcess(StochasticProcess):
    """
    Quantum process on Bloch sphere (x-z plane).

    Based on arXiv:2507.07432, Appendix D.2
    4-token alphabet, requires 1 qubit of quantum memory.
    """

    def __init__(self, alpha: float = 1.0, beta: float = np.sqrt(51), device: str = 'cuda'):
        super().__init__(vocab_size=4, device=device)
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0 / (2.0 * np.sqrt(alpha**2 + beta**2))

        # Define Kraus operators
        self.K = self.get_kraus_operators()

        # Initial state: fully mixed (I/2)
        self.initial_density = torch.tensor([[0.5, 0.0], [0.0, 0.5]],
                                            dtype=torch.float32, device=device)

    def get_kraus_operators(self) -> torch.Tensor:
        """
        Get Kraus operators for the 4 tokens.

        K0 = gamma * [[alpha+beta, 0], [0, alpha-beta]]
        K1 = gamma * [[alpha-beta, 0], [0, alpha+beta]]
        K2 = gamma * [[alpha, beta], [beta, alpha]]
        K3 = gamma * [[alpha, -beta], [-beta, alpha]]

        Returns:
            K: Tensor of shape (4, 2, 2)
        """
        a, b, g = self.alpha, self.beta, self.gamma

        K = torch.zeros((4, 2, 2), dtype=torch.float32)

        K[0] = g * torch.tensor([[a + b, 0], [0, a - b]], dtype=torch.float32)
        K[1] = g * torch.tensor([[a - b, 0], [0, a + b]], dtype=torch.float32)
        K[2] = g * torch.tensor([[a, b], [b, a]], dtype=torch.float32)
        K[3] = g * torch.tensor([[a, -b], [-b, a]], dtype=torch.float32)

        return K.to(self.device)

    def apply_kraus(self, rho: torch.Tensor, token: int) -> torch.Tensor:
        """
        Apply Kraus operator to density matrix.

        Args:
            rho: Density matrix of shape (..., 2, 2)
            token: Token index (0-3)

        Returns:
            Updated density matrix (unnormalized)
        """
        K = self.K[token]  # (2, 2)
        # rho' = K @ rho @ K^T
        result = K @ rho @ K.T
        return result

    def density_to_bloch(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Convert density matrix to Bloch vector (x, z components only).

        For a 2x2 density matrix: rho = (I + x*sigma_x + z*sigma_z) / 2
        x = tr(rho * sigma_x) = rho[0,1] + rho[1,0] (real part)
        z = tr(rho * sigma_z) = rho[0,0] - rho[1,1]

        Args:
            rho: Tensor of shape (..., 2, 2)

        Returns:
            bloch: Tensor of shape (..., 2) with [x, z] components
        """
        x = rho[..., 0, 1] + rho[..., 1, 0]  # real part (since symmetric)
        z = rho[..., 0, 0] - rho[..., 1, 1]
        return torch.stack([x, z], dim=-1)

    def generate_sequences(self, length: int = 10, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences from Bloch Walk process (VECTORIZED for speed).

        Returns:
            sequences: Tensor of shape (B, L) with tokens 0-3
            states: Tensor of shape (B, L, 2, 2) with density matrices
        """
        sequences = []
        states = []

        # Initialize with fully mixed state
        current_rho = self.initial_density.unsqueeze(0).expand(batch_size, -1, -1).clone()

        for _ in range(length):
            # VECTORIZED: Compute probabilities for all 4 tokens at once
            # K: (4, 2, 2), current_rho: (B, 2, 2)
            # Compute K_k @ rho @ K_k^T for all k simultaneously
            # Result shape: (B, 4, 2, 2)
            K_expanded = self.K.unsqueeze(0)  # (1, 4, 2, 2)
            rho_expanded = current_rho.unsqueeze(1)  # (B, 1, 2, 2)

            # K @ rho
            temp = torch.matmul(K_expanded, rho_expanded)  # (B, 4, 2, 2)
            # (K @ rho) @ K^T
            K_T = self.K.transpose(-1, -2).unsqueeze(0)  # (1, 4, 2, 2)
            all_rho_k = torch.matmul(temp, K_T)  # (B, 4, 2, 2)

            # Compute traces: p(token k) = tr(K_k @ rho @ K_k^T)
            probs = torch.diagonal(all_rho_k, dim1=-2, dim2=-1).sum(dim=-1)  # (B, 4)

            # Sample tokens
            tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

            # VECTORIZED: Update density matrices (no batch loop!)
            # Select the Kraus operator for each sample based on sampled token
            selected_K = self.K[tokens]  # (B, 2, 2)
            selected_K_T = selected_K.transpose(-1, -2)  # (B, 2, 2)

            # Compute K @ rho @ K^T for each sample
            temp = torch.matmul(selected_K, current_rho)  # (B, 2, 2)
            new_rho = torch.matmul(temp, selected_K_T)  # (B, 2, 2)

            # VECTORIZED normalization
            traces = torch.diagonal(new_rho, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
            new_rho = new_rho / traces

            sequences.append(tokens)
            states.append(new_rho.clone())
            current_rho = new_rho

        sequences = torch.stack(sequences, dim=1)  # (B, L)
        states = torch.stack(states, dim=1)  # (B, L, 2, 2)

        return sequences, states

    def find_belief_states(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute belief states (Bloch vectors) for sequences (VECTORIZED).

        Args:
            sequences: Tensor of shape (B, L)

        Returns:
            belief_states: Tensor of shape (B, L+1, 2) with Bloch vectors (x, z)
        """
        B, L = sequences.shape

        # Start with fully mixed state
        current_rho = self.initial_density.unsqueeze(0).expand(B, -1, -1).clone()
        belief_states = [self.density_to_bloch(current_rho)]

        for t in range(L):
            # VECTORIZED: Update density matrix for all samples at once
            tokens = sequences[:, t]  # (B,)

            # Select Kraus operators for all samples
            selected_K = self.K[tokens]  # (B, 2, 2)
            selected_K_T = selected_K.transpose(-1, -2)  # (B, 2, 2)

            # Apply Kraus operators: K @ rho @ K^T
            temp = torch.matmul(selected_K, current_rho)  # (B, 2, 2)
            new_rho = torch.matmul(temp, selected_K_T)  # (B, 2, 2)

            # VECTORIZED normalization
            traces = torch.diagonal(new_rho, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
            new_rho = new_rho / traces

            belief_states.append(self.density_to_bloch(new_rho))
            current_rho = new_rho

        # Stack: (B, L+1, 2)
        belief_states = torch.stack(belief_states, dim=1)
        return belief_states

    def find_belief_loss(self, sequences: torch.Tensor, include_start: bool = True) -> torch.Tensor:
        """
        Compute negative log probability of observed sequences.

        Args:
            sequences: Tensor of shape (B, L)
            include_start: Whether to include first token

        Returns:
            loss: Tensor of shape (B, L) or (B, L-1)
        """
        B, L = sequences.shape

        # Start with fully mixed state
        current_rho = self.initial_density.unsqueeze(0).expand(B, -1, -1).clone()
        losses = []

        for t in range(L):
            # Compute probabilities for each token
            probs = torch.zeros((B, 4), device=self.device)
            for k in range(4):
                rho_k = self.apply_kraus(current_rho, k)
                probs[:, k] = torch.diagonal(rho_k, dim1=-2, dim2=-1).sum(dim=-1)

            # Get probability of observed token
            token_probs = probs.gather(1, sequences[:, t].unsqueeze(1)).squeeze(1)
            losses.append(-torch.log(token_probs + 1e-10))

            # Update states
            new_rho = torch.zeros_like(current_rho)
            for i in range(B):
                token = sequences[i, t].item()
                new_rho[i] = self.apply_kraus(current_rho[i], token)
                trace = torch.diagonal(new_rho[i], dim1=-2, dim2=-1).sum()
                new_rho[i] = new_rho[i] / trace

            current_rho = new_rho

        losses = torch.stack(losses, dim=1)  # (B, L)

        start_index = 0 if include_start else 1
        return losses[:, start_index:]


class JointProcess(StochasticProcess):
    """
    Joint process from Cartesian product of two independent processes.

    Combines Mess3 (3 tokens) × BlochWalk (4 tokens) = 12 tokens.
    """

    def __init__(self, process1: StochasticProcess, process2: StochasticProcess, device: str = 'cuda'):
        vocab_size = process1.vocab_size * process2.vocab_size
        super().__init__(vocab_size=vocab_size, device=device)

        self.process1 = process1
        self.process2 = process2
        self.vocab1 = process1.vocab_size
        self.vocab2 = process2.vocab_size

    def encode_joint_token(self, token1: torch.Tensor, token2: torch.Tensor) -> torch.Tensor:
        """Encode two tokens into a single joint token."""
        return token1 * self.vocab2 + token2

    def decode_joint_token(self, joint_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode joint token into two component tokens."""
        token1 = joint_token // self.vocab2
        token2 = joint_token % self.vocab2
        return token1, token2

    def generate_sequences(self, length: int = 10, batch_size: int = 32) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate sequences from joint process.

        Returns:
            joint_sequences: Tensor of shape (B, L) with joint tokens
            (belief_states1, belief_states2): Tuple of belief states for each process
        """
        # Generate from both processes independently
        seq1, _ = self.process1.generate_sequences(length, batch_size)
        seq2, _ = self.process2.generate_sequences(length, batch_size)

        # Encode as joint tokens
        joint_sequences = self.encode_joint_token(seq1, seq2)

        return joint_sequences, (seq1, seq2)

    def find_belief_states(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute belief states for both component processes.

        Args:
            sequences: Joint tokens of shape (B, L)

        Returns:
            (belief1, belief2): Belief states for each process
        """
        # Decode joint tokens
        seq1, seq2 = self.decode_joint_token(sequences)

        # Compute belief states independently
        belief1 = self.process1.find_belief_states(seq1)
        belief2 = self.process2.find_belief_states(seq2)

        return belief1, belief2

    def find_belief_loss(self, sequences: torch.Tensor, include_start: bool = True) -> torch.Tensor:
        """
        Compute combined loss from both processes.

        Args:
            sequences: Joint tokens of shape (B, L)

        Returns:
            loss: Combined loss
        """
        seq1, seq2 = self.decode_joint_token(sequences)

        loss1 = self.process1.find_belief_loss(seq1, include_start)
        loss2 = self.process2.find_belief_loss(seq2, include_start)

        return loss1 + loss2
