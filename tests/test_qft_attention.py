"""
Tests for QFT-based Quantum Attention Layer.

Author: Quantum Attention Research Team
License: MIT
"""

import pytest
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    pytest.skip("Qiskit not available", allow_module_level=True)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestQFTAttentionLayer:
    """Tests for QFTAttentionLayer class."""

    def test_initialization(self):
        """Test layer initialization."""
        from qft_attention import QFTAttentionLayer

        layer = QFTAttentionLayer(n_tokens=4)
        assert layer.n_tokens == 4
        assert layer.shots == 1024

    def test_circuit_construction(self):
        """Test quantum circuit construction."""
        from qft_attention import QFTAttentionLayer

        layer = QFTAttentionLayer(n_tokens=4)
        circuit = layer._build_circuit()

        assert circuit.num_qubits == 4
        assert circuit.depth() > 0

    def test_forward_pass(self):
        """Test forward pass with embeddings."""
        from qft_attention import QFTAttentionLayer

        layer = QFTAttentionLayer(n_tokens=4, shots=1024)
        embeddings = np.random.randn(4, 4)

        output, attention = layer(embeddings)

        assert output.shape == (4, 4)
        assert attention.shape == (4, 4)

    def test_attention_normalization(self):
        """Test that attention weights are normalized."""
        from qft_attention import QFTAttentionLayer

        layer = QFTAttentionLayer(n_tokens=4)
        embeddings = np.random.randn(4, 4)

        _, attention = layer(embeddings)

        # Check rows sum to approximately 1
        row_sums = attention.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4), decimal=1)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestQFTAttentionAdvanced:
    """Tests for advanced QFT attention implementation."""

    def test_initialization(self):
        """Test advanced layer initialization."""
        from qft_attention_advanced import QFTAttentionLayerAdvanced

        layer = QFTAttentionLayerAdvanced(n_tokens=4)
        assert layer.n_tokens == 4
        assert layer.n_ancilla == 3  # n_tokens - 1

    def test_circuit_construction(self):
        """Test advanced circuit construction."""
        from qft_attention_advanced import QFTAttentionLayerAdvanced

        layer = QFTAttentionLayerAdvanced(n_tokens=4)
        circuit = layer._build_circuit()

        # Should have n_tokens + n_ancilla qubits
        assert circuit.num_qubits == 7

    def test_forward_pass(self):
        """Test forward pass with advanced implementation."""
        from qft_attention_advanced import QFTAttentionLayerAdvanced

        layer = QFTAttentionLayerAdvanced(n_tokens=4, shots=1024)
        embeddings = np.random.randn(4, 4)

        output, attention = layer(embeddings)

        assert output.shape == (4, 4)
        assert attention.shape == (4, 4)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_multi_head_initialization(self):
        """Test multi-head attention initialization."""
        from qft_attention import MultiHeadQFTAttention

        mha = MultiHeadQFTAttention(n_tokens=4, n_heads=2)
        assert mha.n_heads == 2

    def test_multi_head_forward(self):
        """Test multi-head forward pass."""
        from qft_attention import MultiHeadQFTAttention

        mha = MultiHeadQFTAttention(n_tokens=4, n_heads=2, shots=512)
        embeddings = np.random.randn(4, 4)

        output = mha(embeddings)
        assert output.shape == (4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
