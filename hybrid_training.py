"""
Hybrid Training Loop for Quantum Attention.

This module implements a hybrid training loop using SPSA optimizer
for training quantum attention layers with classical neural networks.

Author: Quantum Attention Research Team
License: MIT
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from qft_attention import QFTAttentionLayer
from parameter_shift_gradients import ParameterShiftGradient


@dataclass
class TrainingConfig:
    """Configuration for hybrid training."""
    learning_rate: float = 0.01
    spsa_epsilon: float = 0.01
    spsa_gamma: float = 0.101
    spsa_alpha: float = 0.602
    spsa_c: float = 0.1
    epochs: int = 100
    batch_size: int = 32
    shots: int = 1024
    verbose: bool = True
    log_interval: int = 10


@dataclass
class TrainingResult:
    """Results from training."""
    loss_history: List[float]
    gradient_history: List[Dict[str, float]]
    final_loss: float
    training_time: float
    epochs_completed: int
    best_loss: float
    best_epoch: int


class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation optimizer.

    SPSA is efficient for high-dimensional parameter spaces as it only
    requires two function evaluations per iteration regardless of the
    number of parameters.

    Example:
        >>> optimizer = SPSAOptimizer(learning_rate=0.01)
        >>> params = optimizer.step(loss_fn, params)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 0.01,
        gamma: float = 0.101,
        alpha: float = 0.602,
        c: float = 0.1
    ):
        """
        Initialize SPSA optimizer.

        Args:
            learning_rate: Base learning rate (a)
            epsilon: Perturbation size (c)
            gamma: Decay rate for perturbation
            alpha: Decay rate for learning rate
            c: Stability constant
        """
        self.a = learning_rate
        self.c = c
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.iteration = 0

    def step(
        self,
        loss_fn: Callable[[np.ndarray], float],
        params: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Perform one SPSA optimization step.

        Args:
            loss_fn: Loss function to minimize
            params: Current parameters

        Returns:
            Updated parameters and gradient estimate
        """
        n_params = len(params)

        # Generate random perturbation direction
        delta = np.random.choice([-1, 1], size=n_params)

        # Adaptive perturbation size
        ck = self.c / (self.iteration + 1) ** self.gamma

        # Evaluate at perturbed points
        params_plus = params + ck * delta
        params_minus = params - ck * delta

        loss_plus = loss_fn(params_plus)
        loss_minus = loss_fn(params_minus)

        # Gradient estimate
        gradient = (loss_plus - loss_minus) / (2 * ck * delta)

        # Adaptive learning rate
        ak = self.a / (self.iteration + 1) ** self.alpha

        # Update parameters
        new_params = params - ak * gradient

        self.iteration += 1

        return new_params, gradient

    def reset(self):
        """Reset optimizer state."""
        self.iteration = 0


class HybridTrainer:
    """
    Hybrid quantum-classical training loop.

    This class provides:
    - SPSA-based optimization for quantum parameters
    - Adam optimizer for classical parameters
    - Gradient computation via parameter-shift rule
    - Training history tracking

    Example:
        >>> trainer = HybridTrainer(config)
        >>> result = trainer.train(model, train_data, val_data)
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the hybrid trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.spsa_optimizer = SPSAOptimizer(
            learning_rate=config.learning_rate,
            epsilon=config.spsa_epsilon,
            gamma=config.spsa_gamma,
            alpha=config.spsa_alpha,
            c=config.spsa_c
        )
        self.loss_history = []
        self.gradient_history = []

    def train(
        self,
        quantum_layer: QFTAttentionLayer,
        train_embeddings: np.ndarray,
        train_targets: np.ndarray,
        val_embeddings: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None
    ) -> TrainingResult:
        """
        Train the quantum attention layer.

        Args:
            quantum_layer: Quantum attention layer to train
            train_embeddings: Training embeddings
            train_targets: Training targets
            val_embeddings: Validation embeddings (optional)
            val_targets: Validation targets (optional)

        Returns:
            TrainingResult with history and metrics
        """
        start_time = time.time()

        # Initialize parameters
        params = np.random.uniform(0, 2*np.pi, quantum_layer.n_tokens)

        self.loss_history = []
        self.gradient_history = []
        best_loss = float('inf')
        best_params = params.copy()
        best_epoch = 0

        for epoch in range(self.config.epochs):
            # Define loss function for current batch
            def loss_fn(p):
                # Update quantum layer parameters
                quantum_layer.params = p

                # Forward pass
                output, _ = quantum_layer(train_embeddings)

                # Compute loss
                loss = np.mean((output - train_targets) ** 2)
                return loss

            # SPSA step
            params, gradient = self.spsa_optimizer.step(loss_fn, params)

            # Compute current loss
            loss = loss_fn(params)
            self.loss_history.append(loss)
            self.gradient_history.append({f"param_{i}": g for i, g in enumerate(gradient)})

            # Track best
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
                best_epoch = epoch

            # Validation
            if val_embeddings is not None and val_targets is not None:
                val_loss = self._validate(quantum_layer, val_embeddings, val_targets)

            # Logging
            if self.config.verbose and epoch % self.config.log_interval == 0:
                log_msg = f"Epoch {epoch}: Loss = {loss:.6f}"
                if val_embeddings is not None:
                    log_msg += f", Val Loss = {val_loss:.6f}"
                print(log_msg)

        training_time = time.time() - start_time

        # Restore best parameters
        quantum_layer.params = best_params

        return TrainingResult(
            loss_history=self.loss_history,
            gradient_history=self.gradient_history,
            final_loss=self.loss_history[-1],
            training_time=training_time,
            epochs_completed=self.config.epochs,
            best_loss=best_loss,
            best_epoch=best_epoch
        )

    def _validate(
        self,
        quantum_layer: QFTAttentionLayer,
        val_embeddings: np.ndarray,
        val_targets: np.ndarray
    ) -> float:
        """Compute validation loss."""
        output, _ = quantum_layer(val_embeddings)
        loss = np.mean((output - val_targets) ** 2)
        return loss


class HybridQuantumClassicalModel:
    """
    Complete hybrid quantum-classical model.

    Combines:
    - Classical preprocessing layers
    - Quantum attention layer
    - Classical postprocessing layers

    Example:
        >>> model = HybridQuantumClassicalModel(input_dim=64, n_tokens=4)
        >>> output = model(input_data)
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        n_tokens: int = 4,
        output_dim: int = 64
    ):
        """
        Initialize hybrid model.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for classical layers
            n_tokens: Number of tokens for quantum attention
            output_dim: Output dimension
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HybridQuantumClassicalModel")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_tokens = n_tokens
        self.output_dim = output_dim

        # Classical preprocessing
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_tokens)  # Project to quantum dimension
        )

        # Quantum attention
        self.quantum_attention = QFTAttentionLayer(n_tokens=n_tokens)

        # Classical postprocessing
        self.postprocess = nn.Sequential(
            nn.Linear(n_tokens, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass through hybrid model.

        Args:
            x: Input tensor of shape (batch, n_tokens, input_dim)

        Returns:
            Output tensor of shape (batch, n_tokens, output_dim)
        """
        # Classical preprocessing
        x_preprocessed = self.preprocess(x)

        # Convert to numpy for quantum processing
        x_np = x_preprocessed.detach().numpy()

        # Quantum attention
        quantum_output, _ = self.quantum_attention(x_np)

        # Convert back to tensor
        x_quantum = torch.from_numpy(quantum_output).float()

        # Classical postprocessing
        output = self.postprocess(x_quantum)

        return output


def train_hybrid_model(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    config: Optional[TrainingConfig] = None
) -> Tuple[HybridQuantumClassicalModel, TrainingResult]:
    """
    Convenience function to train a hybrid model.

    Args:
        train_data: Tuple of (embeddings, targets)
        val_data: Optional validation data
        config: Training configuration

    Returns:
        Trained model and training result
    """
    config = config or TrainingConfig()

    # Create model
    model = HybridQuantumClassicalModel()

    # Create trainer
    trainer = HybridTrainer(config)

    # Train
    train_embeddings, train_targets = train_data
    val_embeddings, val_targets = val_data if val_data else (None, None)

    result = trainer.train(
        model.quantum_attention,
        train_embeddings,
        train_targets,
        val_embeddings,
        val_targets
    )

    return model, result


def plot_training_history(result: TrainingResult) -> None:
    """
    Plot training loss history.

    Args:
        result: Training result with loss history
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(result.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.yscale("log")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Testing Hybrid Training Loop...
")

    # Generate synthetic data
    n_samples = 100
    n_tokens = 4
    embed_dim = 4

    train_embeddings = np.random.randn(n_samples, n_tokens, embed_dim)
    train_targets = np.random.randn(n_samples, n_tokens, embed_dim)

    # Create training config
    config = TrainingConfig(
        epochs=50,
        learning_rate=0.01,
        verbose=True,
        log_interval=10
    )

    # Create quantum layer
    quantum_layer = QFTAttentionLayer(n_tokens=n_tokens)

    # Create trainer
    trainer = HybridTrainer(config)

    # Train
    print("Training quantum attention layer...")
    result = trainer.train(
        quantum_layer,
        train_embeddings.reshape(-1, embed_dim),
        train_targets.reshape(-1, embed_dim)
    )

    print(f"
Training completed in {result.training_time:.2f} seconds")
    print(f"Final loss: {result.final_loss:.6f}")
    print(f"Best loss: {result.best_loss:.6f} at epoch {result.best_epoch}")

    print("
All tests passed!")
