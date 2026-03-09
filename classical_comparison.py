"""
Classical vs Quantum Attention Comparison.

This module provides benchmarking tools to compare quantum attention
against classical scaled-dot-product attention.

Author: Quantum Attention Research Team
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from qft_attention import QFTAttentionLayer


@dataclass
class ComparisonResult:
    """Results from classical vs quantum comparison."""
    kl_divergence: float
    correlation: float
    classical_time: float
    quantum_time: float
    classical_memory: float
    quantum_memory: float
    sequence_length: int
    embedding_dim: int


class ClassicalAttention:
    """Classical scaled-dot-product attention for comparison."""

    def __init__(self, embed_dim: int = 4):
        """
        Initialize classical attention.

        Args:
            embed_dim: Embedding dimension
        """
        self.embed_dim = embed_dim

    def __call__(
        self,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute classical attention.

        Args:
            embeddings: Input embeddings of shape (n_tokens, embed_dim)

        Returns:
            Output embeddings and attention weights
        """
        if TORCH_AVAILABLE:
            return self._torch_attention(embeddings)
        return self._numpy_attention(embeddings)

    def _torch_attention(
        self,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attention using PyTorch."""
        # Convert to tensor
        x = torch.from_numpy(embeddings).float()

        # Compute attention scores
        scores = torch.matmul(x, x.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention, x)

        return output.numpy(), attention.numpy()

    def _numpy_attention(
        self,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute attention using NumPy."""
        # Compute attention scores
        scores = np.matmul(embeddings, embeddings.T) / np.sqrt(self.embed_dim)

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply attention
        output = np.matmul(attention, embeddings)

        return output, attention


class AttentionComparator:
    """
    Compare quantum and classical attention mechanisms.

    This class provides comprehensive benchmarking tools:
    - KL divergence between attention distributions
    - Correlation of attention patterns
    - Computational time comparison
    - Memory usage comparison
    - Scaling analysis

    Example:
        >>> comparator = AttentionComparator()
        >>> results = comparator.compare(n_tokens=4, embed_dim=4)
        >>> print(f"KL Divergence: {results.kl_divergence:.4f}")
    """

    def __init__(
        self,
        use_advanced_quantum: bool = False,
        shots: int = 1024
    ):
        """
        Initialize the comparator.

        Args:
            use_advanced_quantum: Use advanced MC-RZ implementation
            shots: Number of measurement shots for quantum
        """
        self.use_advanced_quantum = use_advanced_quantum
        self.shots = shots

    def compare(
        self,
        n_tokens: int = 4,
        embed_dim: int = 4,
        n_runs: int = 10
    ) -> ComparisonResult:
        """
        Compare quantum and classical attention.

        Args:
            n_tokens: Number of tokens
            embed_dim: Embedding dimension
            n_runs: Number of runs for timing

        Returns:
            ComparisonResult with metrics
        """
        # Generate random embeddings
        embeddings = np.random.randn(n_tokens, embed_dim)

        # Initialize attention layers
        if self.use_advanced_quantum:
            from qft_attention_advanced import QFTAttentionLayerAdvanced
            quantum_attention = QFTAttentionLayerAdvanced(
                n_tokens=n_tokens,
                shots=self.shots
            )
        else:
            quantum_attention = QFTAttentionLayer(
                n_tokens=n_tokens,
                shots=self.shots
            )

        classical_attention = ClassicalAttention(embed_dim=embed_dim)

        # Time classical attention
        classical_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            classical_output, classical_attn = classical_attention(embeddings)
            classical_times.append(time.perf_counter() - start)

        # Time quantum attention
        quantum_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            quantum_output, quantum_attn = quantum_attention(embeddings)
            quantum_times.append(time.perf_counter() - start)

        # Compute KL divergence
        kl_div = self._kl_divergence(classical_attn, quantum_attn)

        # Compute correlation
        correlation = self._correlation(classical_attn, quantum_attn)

        return ComparisonResult(
            kl_divergence=kl_div,
            correlation=correlation,
            classical_time=np.mean(classical_times),
            quantum_time=np.mean(quantum_times),
            classical_memory=0,  # Placeholder
            quantum_memory=0,   # Placeholder
            sequence_length=n_tokens,
            embedding_dim=embed_dim
        )

    def _kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Compute KL divergence between attention distributions.

        Args:
            p: First distribution (classical)
            q: Second distribution (quantum)
            epsilon: Small value to avoid log(0)

        Returns:
            KL divergence
        """
        # Flatten and normalize
        p_flat = p.flatten() + epsilon
        q_flat = q.flatten() + epsilon

        p_flat = p_flat / p_flat.sum()
        q_flat = q_flat / q_flat.sum()

        # KL divergence
        kl = np.sum(p_flat * np.log(p_flat / q_flat))
        return float(kl)

    def _correlation(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Compute correlation between attention patterns.

        Args:
            p: First attention matrix
            q: Second attention matrix

        Returns:
            Correlation coefficient
        """
        p_flat = p.flatten()
        q_flat = q.flatten()

        correlation = np.corrcoef(p_flat, q_flat)[0, 1]
        return float(correlation)

    def benchmark_scaling(
        self,
        token_range: List[int] = [2, 4, 6, 8],
        embed_dim: int = 4,
        n_runs: int = 5
    ) -> Dict[int, ComparisonResult]:
        """
        Benchmark attention across different sequence lengths.

        Args:
            token_range: List of token counts to test
            embed_dim: Embedding dimension
            n_runs: Number of runs per configuration

        Returns:
            Dictionary mapping token count to results
        """
        results = {}

        for n_tokens in token_range:
            print(f"Benchmarking {n_tokens} tokens...")
            results[n_tokens] = self.compare(
                n_tokens=n_tokens,
                embed_dim=embed_dim,
                n_runs=n_runs
            )

        return results

    def generate_report(
        self,
        results: Dict[int, ComparisonResult]
    ) -> str:
        """
        Generate a formatted report from benchmark results.

        Args:
            results: Dictionary of benchmark results

        Returns:
            Formatted report string
        """
        report = []
        report.append("# Classical vs Quantum Attention Benchmark
")
        report.append("| Tokens | KL Divergence | Correlation | Classical Time | Quantum Time |
")
        report.append("|--------|---------------|-------------|----------------|---------------|
")

        for n_tokens, result in sorted(results.items()):
            report.append(
                f"| {n_tokens} | {result.kl_divergence:.4f} | {result.correlation:.4f} | "
                f"{result.classical_time*1000:.2f}ms | {result.quantum_time*1000:.2f}ms |
"
            )

        return "".join(report)


def run_comparison_benchmark(
    token_range: List[int] = [2, 4, 6],
    embed_dim: int = 4,
    use_advanced: bool = False
) -> str:
    """
    Convenience function to run a full comparison benchmark.

    Args:
        token_range: Token counts to test
        embed_dim: Embedding dimension
        use_advanced: Use advanced quantum implementation

    Returns:
        Formatted report string
    """
    comparator = AttentionComparator(use_advanced_quantum=use_advanced)
    results = comparator.benchmark_scaling(
        token_range=token_range,
        embed_dim=embed_dim
    )
    return comparator.generate_report(results)


if __name__ == "__main__":
    print("Running Classical vs Quantum Attention Benchmark...
")

    # Run benchmark
    report = run_comparison_benchmark(
        token_range=[2, 4, 6],
        embed_dim=4,
        use_advanced=False
    )

    print(report)

    # Run with advanced implementation
    print("
Running with Advanced Implementation...
")
    report_advanced = run_comparison_benchmark(
        token_range=[2, 4, 6],
        embed_dim=4,
        use_advanced=True
    )
    print(report_advanced)
