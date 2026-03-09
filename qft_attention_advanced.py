#!/usr/bin/env python3
"""
Advanced QFT-based Quantum Attention Layer
==========================================
Implements true multi-controlled RZ using MCXGate for genuine AND-based control.
This fixes the parity-encoding issue in the original implementation.
"""

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QFT, MCXGate
from qiskit.quantum_info import Statevector
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class AdvancedQFTAttention:
    """
    Advanced QFT-based Attention with true multi-controlled RZ.
    
    Features:
    - True AND-based multi-control (not parity)
    - Proper ancilla management
    - Support for larger token sequences
    """
    
    def __init__(self, num_tokens: int = 3):
        self.num_tokens = num_tokens
        self.num_pairs = num_tokens * (num_tokens - 1) // 2
        
        # Parameters
        self.phi = ParameterVector('phi', num_tokens)
        self.theta = ParameterVector('theta', self.num_pairs)
        
        # Build circuit
        self.circuit = self._build_circuit()
        
    def _get_pair_indices(self) -> List[Tuple[int, int]]:
        pairs = []
        for i in range(self.num_tokens):
            for j in range(i + 1, self.num_tokens):
                pairs.append((i, j))
        return pairs
    
    def _build_circuit(self) -> QuantumCircuit:
        """Build the quantum attention circuit with true multi-controlled RZ."""
        # Total qubits: tokens + ancilla for each pair
        total_qubits = self.num_tokens + self.num_pairs
        
        qc = QuantumCircuit(total_qubits, self.num_tokens)
        
        # Token qubits are 0 to num_tokens-1
        token_qubits = list(range(self.num_tokens))
        
        # Ancilla qubits start after token qubits
        ancilla_start = self.num_tokens
        
        # Token embeddings - learnable RY rotations
        for i in range(self.num_tokens):
            qc.ry(self.phi[i], i)
        
        # QFT on the token register
        qft = QFT(num_qubits=self.num_tokens, do_swaps=False, inverse=False)
        qft_decomposed = transpile(qft, basis_gates=['u1', 'u2', 'u3', 'cx'])
        qc.compose(qft_decomposed, qubits=token_qubits, inplace=True)
        
        # Diagonal phase block - true multi-controlled RZ for each pair
        pairs = self._get_pair_indices()
        
        for pair_idx, (i, j) in enumerate(pairs):
            # Target is the qubit not in the pair
            target = [k for k in range(self.num_tokens) if k not in (i, j)][0]
            
            # Ancilla qubit for this pair
            ancilla = ancilla_start + pair_idx
            
            # Multi-controlled X on ancilla (flips when ALL controls are 1)
            mcx = MCXGate(num_ctrl_qubits=2)
            qc.append(mcx, [i, j, ancilla])
            
            # RZ on target conditioned on ancilla
            qc.crz(self.theta[pair_idx], ancilla, target)
            
            # Uncompute ancilla
            qc.append(mcx, [i, j, ancilla])
        
        # Inverse QFT
        qft_inv = QFT(num_qubits=self.num_tokens, do_swaps=False, inverse=True)
        qft_inv_decomposed = transpile(qft_inv, basis_gates=['u1', 'u2', 'u3', 'cx'])
        qc.compose(qft_inv_decomposed, qubits=token_qubits, inplace=True)
        
        # Measure token qubits only
        qc.measure(token_qubits, list(range(self.num_tokens)))
        
        return qc
    
    def run(self, phi_values: np.ndarray, theta_values: np.ndarray, 
            shots: int = 4096) -> dict:
        """Run the quantum attention circuit."""
        param_dict = {}
        for i, val in enumerate(phi_values):
            param_dict[self.phi[i]] = val
        for i, val in enumerate(theta_values):
            param_dict[self.theta[i]] = val
        
        bound_circuit = self.circuit.assign_parameters(param_dict)
        transpiled = transpile(bound_circuit, basis_gates=['u1', 'u2', 'u3', 'cx', 'measure'])
        
        sim = AerSimulator()
        result = sim.run(transpiled, shots=shots).result()
        return result.get_counts()


def compare_implementations():
    """Compare simple CRZ vs true multi-controlled RZ."""
    print("="*60)
    print("Comparing Attention Implementations")
    print("="*60)
    
    phi = np.array([0.2, 0.4, 0.6])
    theta = np.array([0.3, 0.5, 0.7])
    
    print("\n1. Simple CRZ Implementation")
    print("-" * 40)
    from qft_attention import QFTAttentionLayer
    simple = QFTAttentionLayer(num_tokens=3, num_heads=1)
    counts_simple = simple.run(phi, theta, shots=4096)
    
    print(f"Circuit depth: {simple.circuit.depth()}")
    print(f"Total qubits: {simple.num_tokens}")
    print(f"Top 3 states:")
    for state, count in sorted(counts_simple.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  |{state}>: {count/4096*100:.1f}%")
    
    print("\n2. True Multi-Controlled RZ Implementation")
    print("-" * 40)
    advanced = AdvancedQFTAttention(num_tokens=3)
    counts_advanced = advanced.run(phi, theta, shots=4096)
    
    print(f"Circuit depth: {advanced.circuit.depth()}")
    print(f"Total qubits: {advanced.num_tokens + advanced.num_pairs} (tokens + ancilla)")
    print(f"Top 3 states:")
    for state, count in sorted(counts_advanced.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  |{state}>: {count/4096*100:.1f}%")
    
    print("\n3. Distribution Comparison")
    print("-" * 40)
    all_states = set(counts_simple.keys()) | set(counts_advanced.keys())
    print(f"{'State':<8} {'Simple':>10} {'Advanced':>10} {'Diff':>10}")
    print("-" * 40)
    for state in sorted(all_states):
        s = counts_simple.get(state, 0) / 4096 * 100
        a = counts_advanced.get(state, 0) / 4096 * 100
        print(f"|{state}>  {s:>9.1f}% {a:>9.1f}% {abs(s-a):>9.1f}%")
    
    print("\n4. Key Differences")
    print("-" * 40)
    print("Simple CRZ: Uses controlled-RZ directly between pairs")
    print("Advanced: Uses MCX + CRZ for true AND-based control")
    print(f"\nCircuit depth increase: {advanced.circuit.depth() - simple.circuit.depth()} gates")
    print(f"Ancilla overhead: {advanced.num_pairs} additional qubits")


if __name__ == "__main__":
    compare_implementations()
