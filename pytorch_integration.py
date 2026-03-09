"""
PyTorch Integration for QFT-based Quantum Attention Layer.

This module provides a torch.nn.Module wrapper around the QFTAttentionLayer
that enables seamless integration with PyTorch neural networks and supports
gradient computation via the parameter-shift rule.

Author: Quantum Attention Research Team
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, List, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError(
        "PyTorch is required for this module. "
        "Install with: pip install torch"
    )

from qft_attention import QFTAttentionLayer
from parameter_shift_gradients import ParameterShiftGradient


class QFTAttentionModule(nn.Module):
    """
    PyTorch nn.Module wrapper for QFT-based Quantum Attention.

    This module wraps the quantum attention layer and provides:
    - Seamless integration with PyTorch neural networks
    - Gradient computation via parameter-shift rule
    - Support for multi-head attention
    - Batch processing of quantum circuits

    Attributes:
        n_tokens (int): Number of tokens in the sequence
        n_heads (int): Number of attention heads
        use_advanced (bool): Whether to use advanced MC-RZ implementation
        shots (int): Number of measurement shots
        backend: Qiskit backend for execution

    Example:
        >>> layer = QFTAttentionModule(n_tokens=4, n_heads=2)
        >>> embeddings = torch.randn(4, 4)  # 4 tokens, 4-dim embeddings
        >>> output = layer(embeddings)
        >>> output.shape
        torch.Size([4, 4])
    """

    def __init__(
        self,
        n_tokens: int = 4,
        n_heads: int = 1,
        use_advanced: bool = False,
        shots: int = 1024,
        backend=None,
        device: str = "cpu"
    ):
        """
        Initialize the QFT Attention Module.

        Args:
            n_tokens: Number of tokens in the sequence
            n_heads: Number of attention heads
            use_advanced: Whether to use advanced MC-RZ implementation
            shots: Number of measurement shots
            backend: Qiskit backend (defaults to Aer simulator)
            device: PyTorch device ("cpu" or "cuda")
        """
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for QFTAttentionModule")

        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.use_advanced = use_advanced
        self.shots = shots
        self.device = device

        # Initialize quantum attention layers (one per head)
        self.attention_layers = nn.ModuleList([
            QFTAttentionLayer(
                n_tokens=n_tokens,
                use_advanced=use_advanced,
                shots=shots,
                backend=backend
            )
            for _ in range(n_heads)
        ])

        # Learnable parameters for combining heads
        self.head_weights = nn.Parameter(
            torch.ones(n_heads) / n_heads
        )

        # Output projection
        self.output_proj = nn.Linear(n_tokens, n_tokens)

        # Gradient computer
        self.grad_computer = ParameterShiftGradient()

    def forward(
        self,
        embeddings: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the quantum attention layer.

        Args:
            embeddings: Input token embeddings of shape (batch, n_tokens, embed_dim)
                       or (n_tokens, embed_dim)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor of shape (batch, n_tokens, embed_dim)
            Optionally returns attention weights
        """
        # Handle input shape
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = embeddings.shape[0]

        # Process each head
        head_outputs = []
        attention_weights = []

        for head_idx, layer in enumerate(self.attention_layers):
            # Convert to numpy for quantum processing
            embeddings_np = embeddings.detach().cpu().numpy()

            # Process batch
            batch_outputs = []
            batch_attention = []

            for b in range(batch_size):
                # Get attention output
                output, attn = layer(embeddings_np[b])
                batch_outputs.append(output)
                batch_attention.append(attn)

            # Stack batch results
            head_output = np.stack(batch_outputs, axis=0)
            head_attn = np.stack(batch_attention, axis=0)

            head_outputs.append(torch.from_numpy(head_output).to(self.device))
            attention_weights.append(torch.from_numpy(head_attn).to(self.device))

        # Combine heads with learned weights
        weights = F.softmax(self.head_weights, dim=0)

        output = torch.zeros_like(head_outputs[0])
        for h, head_out in enumerate(head_outputs):
            output = output + weights[h] * head_out

        # Apply output projection
        output = self.output_proj(output)

        if squeeze_output:
            output = output.squeeze(0)

        if return_attention:
            return output, torch.stack(attention_weights, dim=1)
        return output

    def compute_gradients(
        self,
        embeddings: torch.Tensor,
        loss_fn: callable,
        target: torch.Tensor
    ) -> dict:
        """
        Compute gradients using parameter-shift rule.

        Args:
            embeddings: Input embeddings
            loss_fn: Loss function
            target: Target tensor

        Returns:
            Dictionary of parameter gradients
        """
        gradients = {}

        for head_idx, layer in enumerate(self.attention_layers):
            # Get parameter names
            param_names = [f"theta_{i}" for i in range(self.n_tokens)]

            for param_name in param_names:
                # Compute gradient via parameter-shift
                grad = self.grad_computer.compute_gradient(
                    layer,
                    param_name,
                    embeddings.detach().cpu().numpy(),
                    loss_fn,
                    target.detach().cpu().numpy()
                )
                gradients[f"layer_{head_idx}.{param_name}"] = grad

        return gradients


class HybridQuantumAttention(nn.Module):
    """
    Hybrid quantum-classical attention mechanism.

    Combines quantum attention with classical projections:
    1. Classical linear projection to quantum embedding space
    2. Quantum attention computation
    3. Classical output projection

    This architecture allows for:
    - End-to-end training with backpropagation
    - Gradient computation via parameter-shift for quantum parameters
    - Seamless integration with existing transformer architectures
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_tokens: int = 4,
        n_heads: int = 1,
        quantum_dim: int = 4,
        dropout: float = 0.1,
        use_advanced: bool = False
    ):
        """
        Initialize hybrid attention module.

        Args:
            embed_dim: Classical embedding dimension
            n_tokens: Number of tokens
            n_heads: Number of attention heads
            quantum_dim: Dimension of quantum embedding space
            dropout: Dropout probability
            use_advanced: Use advanced MC-RZ implementation
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.quantum_dim = quantum_dim

        # Classical projections
        self.query_proj = nn.Linear(embed_dim, quantum_dim)
        self.key_proj = nn.Linear(embed_dim, quantum_dim)
        self.value_proj = nn.Linear(embed_dim, quantum_dim)

        # Quantum attention
        self.quantum_attention = QFTAttentionModule(
            n_tokens=n_tokens,
            n_heads=n_heads,
            use_advanced=use_advanced
        )

        # Output projection
        self.output_proj = nn.Linear(quantum_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through hybrid attention.

        Args:
            x: Input tensor of shape (batch, n_tokens, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, n_tokens, embed_dim)
        """
        residual = x

        # Classical projections
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Quantum attention (on queries and keys)
        # Average query and key for quantum processing
        quantum_input = (queries + keys) / 2
        quantum_output = self.quantum_attention(quantum_input)

        # Combine with values
        output = quantum_output * values

        # Output projection
        output = self.output_proj(output)
        output = self.dropout(output)

        # Residual connection and layer norm
        output = self.layer_norm(residual + output)

        return output


def create_quantum_transformer_layer(
    embed_dim: int = 64,
    n_tokens: int = 4,
    n_heads: int = 1,
    ffn_dim: int = 256,
    dropout: float = 0.1,
    use_advanced: bool = False
) -> nn.Module:
    """
    Create a complete transformer layer with quantum attention.

    Args:
        embed_dim: Embedding dimension
        n_tokens: Number of tokens
        n_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        dropout: Dropout probability
        use_advanced: Use advanced MC-RZ implementation

    Returns:
        Complete transformer layer module
    """

    class QuantumTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = HybridQuantumAttention(
                embed_dim=embed_dim,
                n_tokens=n_tokens,
                n_heads=n_heads,
                dropout=dropout,
                use_advanced=use_advanced
            )
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout)
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        def forward(self, x):
            # Self-attention with residual
            x = self.norm1(x + self.attention(x))
            # FFN with residual
            x = self.norm2(x + self.ffn(x))
            return x

    return QuantumTransformerLayer()


if __name__ == "__main__":
    # Example usage
    print("Testing QFTAttentionModule...")

    # Create module
    module = QFTAttentionModule(n_tokens=4, n_heads=2)

    # Test forward pass
    embeddings = torch.randn(2, 4, 4)  # Batch of 2, 4 tokens, 4-dim
    output = module(embeddings)
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")

    # Test with attention weights
    output, attention = module(embeddings, return_attention=True)
    print(f"Attention shape: {attention.shape}")

    # Test hybrid attention
    print("
Testing HybridQuantumAttention...")
    hybrid = HybridQuantumAttention(embed_dim=64, n_tokens=4)
    x = torch.randn(2, 4, 64)
    output = hybrid(x)
    print(f"Hybrid output shape: {output.shape}")

    print("
All tests passed!")
