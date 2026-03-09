#!/usr/bin/env python3
"""
QFT-based Quantum Attention Layer - Research Prototype
=======================================================
This implements a quantum attention mechanism using QFT and multi-controlled RZ gates.

Key concepts:
- Token embeddings as learnable RY rotations
- QFT for frequency-domain attention
- Multi-controlled RZ for pairwise attention phases
- Inverse QFT to return to token basis

Reference: "Quantum Attention Networks" research
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QFT, MCXGate
from qiskit.quantum_info import Statevector
import numpy as np
from typing import List, Tuple, Optional

# ------------------------------------------------------------
# 1. QFT Attention Layer Class
# ------------------------------------------------------------
class QFTAttentionLayer:
    """
    Quantum Fourier Transform-based Attention Layer.
    
    Implements attention mechanism using:
    1. Token embeddings via RY rotations
    2. QFT to transform to frequency domain
    3. Multi-controlled RZ for attention phases
    4. Inverse QFT to return to token basis
    """
    
    def __init__(self, num_tokens: int = 3, num_heads: int = 1):
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.num_pairs = num_tokens * (num_tokens - 1) // 2
        
        self.phi = ParameterVector('phi', num_tokens)
        self.theta = ParameterVector('theta', self.num_pairs * num_heads)
        
        self.circuit = self._build_circuit()
        
    def _get_pair_indices(self) -> List[Tuple[int, int]]:
        pairs = []
        for i in range(self.num_tokens):
            for j in range(i + 1, self.num_tokens):
                pairs.append((i, j))
        return pairs
    
    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_tokens, self.num_tokens)
        
        # Token embeddings - learnable RY rotations
        for i in range(self.num_tokens):
            qc.ry(self.phi[i], i)
        
        # QFT on the token register (decomposed into basic gates)
        qft = QFT(num_qubits=self.num_tokens, do_swaps=False, inverse=False)
        qft_decomposed = transpile(qft, basis_gates=['u1', 'u2', 'u3', 'cx'])
        qc.compose(qft_decomposed, qubits=list(range(self.num_tokens)), inplace=True)
        
        # Diagonal phase block - controlled RZ for each pair
        pairs = self._get_pair_indices()
        
        for head in range(self.num_heads):
            for pair_idx, (i, j) in enumerate(pairs):
                theta_idx = head * self.num_pairs + pair_idx
                qc.crz(self.theta[theta_idx], i, j)
        
        # Inverse QFT (decomposed into basic gates)
        qft_inv = QFT(num_qubits=self.num_tokens, do_swaps=False, inverse=True)
        qft_inv_decomposed = transpile(qft_inv, basis_gates=['u1', 'u2', 'u3', 'cx'])
        qc.compose(qft_inv_decomposed, qubits=list(range(self.num_tokens)), inplace=True)
        
        # Measure token qubits
        qc.measure(list(range(self.num_tokens)), list(range(self.num_tokens)))
        
        return qc
    
    def get_circuit(self) -> QuantumCircuit:
        return self.circuit
    
    def bind_parameters(self, phi_values: np.ndarray, theta_values: np.ndarray) -> QuantumCircuit:
        param_dict = {}
        for i, val in enumerate(phi_values):
            param_dict[self.phi[i]] = val
        for i, val in enumerate(theta_values):
            param_dict[self.theta[i]] = val
        
        return self.circuit.assign_parameters(param_dict)
    
    def run(self, phi_values: np.ndarray, theta_values: np.ndarray, 
            shots: int = 4096) -> dict:
        bound_circuit = self.bind_parameters(phi_values, theta_values)
        # Transpile to basis gates for Aer
        transpiled = transpile(bound_circuit, basis_gates=['u1', 'u2', 'u3', 'cx', 'measure'])
        sim = AerSimulator()
        result = sim.run(transpiled, shots=shots).result()
        return result.get_counts()


# ------------------------------------------------------------
# 2. Multi-Head Attention Extension
# ------------------------------------------------------------
class MultiHeadQFTAttention:
    """
    Multi-head QFT-based attention.
    
    Each head learns different attention patterns,
    similar to classical multi-head attention.
    """
    
    def __init__(self, num_tokens: int = 3, num_heads: int = 2):
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.layers = [QFTAttentionLayer(num_tokens, 1) for _ in range(num_heads)]
        
    def run(self, phi_values: np.ndarray, all_theta_values: List[np.ndarray], 
            shots: int = 4096) -> List[dict]:
        results = []
        for layer, theta_values in zip(self.layers, all_theta_values):
            results.append(layer.run(phi_values, theta_values, shots))
        return results


# ------------------------------------------------------------
# 3. Demonstration and Testing
# ------------------------------------------------------------
def main():
    print("="*60)
    print("QFT-based Quantum Attention Layer - Research Prototype")
    print("="*60)
    
    print("\n1. Single-Head QFT Attention")
    print("-" * 40)
    
    qa = QFTAttentionLayer(num_tokens=3, num_heads=1)
    print(f"Circuit depth: {qa.circuit.depth()}")
    print(f"Number of parameters: {len(qa.circuit.parameters)}")
    
    phi_values = np.array([0.2, 0.4, 0.6])
    theta_values = np.array([0.3, 0.5, 0.7])
    
    print(f"\nToken embeddings (phi): {phi_values}")
    print(f"Attention phases (theta): {theta_values}")
    
    print("\nRunning simulation...")
    counts = qa.run(phi_values, theta_values, shots=4096)
    print(f"\nMeasurement results (top 5):")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for state, count in sorted_counts:
        print(f"  |{state}>: {count} shots ({count/4096*100:.1f}%)")
    
    print("\n2. Multi-Head QFT Attention")
    print("-" * 40)
    
    mha = MultiHeadQFTAttention(num_tokens=3, num_heads=2)
    print(f"Number of heads: {mha.num_heads}")
    
    theta_head1 = np.array([0.3, 0.5, 0.7])
    theta_head2 = np.array([0.1, 0.2, 0.9])
    
    print(f"\nHead 1 theta: {theta_head1}")
    print(f"Head 2 theta: {theta_head2}")
    
    print("\nRunning multi-head simulation...")
    results = mha.run(phi_values, [theta_head1, theta_head2], shots=4096)
    
    for i, counts in enumerate(results):
        print(f"\nHead {i+1} results (top 3):")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for state, count in sorted_counts:
            print(f"  |{state}>: {count} shots ({count/4096*100:.1f}%)")
    
    print("\n3. Circuit Structure")
    print("-" * 40)
    print(f"Total qubits: {qa.num_tokens}")
    print(f"Total parameters: {len(qa.circuit.parameters)}")
    print(f"Circuit depth: {qa.circuit.depth()}")
    print(f"Gate count: {qa.circuit.count_ops()}")
    
    return counts


if __name__ == "__main__":
    main()
