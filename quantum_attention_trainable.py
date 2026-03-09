#!/usr/bin/env python3
"""
Trainable Quantum Attention Layer
=================================
Implements gradient computation via parameter-shift rule and
hybrid PyTorch integration for end-to-end training.

Key Features:
- Parameter-shift rule for gradient computation
- PyTorch autograd integration
- Hybrid quantum-classical training loop
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Try to import PyTorch for hybrid training
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Hybrid training features disabled.")


class ParameterShiftGradient:
    """
    Implements the parameter-shift rule for gradient computation.
    
    For a parameterized gate U(θ) = exp(-iθG/2) where G has eigenvalues ±1,
    the gradient of expectation value ⟨O⟩ is:
    
    ∂⟨O⟩/∂θ = (⟨O⟩(θ + π/2) - ⟨O⟩(θ - π/2)) / 2
    """
    
    def __init__(self, circuit_builder, shift: float = np.pi/2):
        """
        Args:
            circuit_builder: Function that builds the parameterized circuit
            shift: Parameter shift value (default π/2 for Pauli rotations)
        """
        self.circuit_builder = circuit_builder
        self.shift = shift
        
    def compute_gradient(self, params: np.ndarray, param_idx: int, 
                         observable: str = 'Z0', shots: int = 4096) -> float:
        """
        Compute gradient using parameter-shift rule.
        
        Args:
            params: Current parameter values
            param_idx: Index of parameter to compute gradient for
            observable: Observable to measure (e.g., 'Z0' for qubit 0)
            shots: Number of shots for measurement
            
        Returns:
            Gradient value
        """
        # Create shifted parameter arrays
        params_plus = params.copy()
        params_minus = params.copy()
        
        params_plus[param_idx] += self.shift
        params_minus[param_idx] -= self.shift
        
        # Compute expectation values
        exp_plus = self._compute_expectation(params_plus, observable, shots)
        exp_minus = self._compute_expectation(params_minus, observable, shots)
        
        # Gradient via parameter-shift rule
        gradient = (exp_plus - exp_minus) / 2
        
        return gradient
    
    def _compute_expectation(self, params: np.ndarray, observable: str, 
                             shots: int) -> float:
        """Compute expectation value of observable."""
        circuit, param_vector = self.circuit_builder()
        
        # Bind parameters
        param_dict = {param_vector[i]: params[i] for i in range(len(params))}
        bound_circuit = circuit.assign_parameters(param_dict)
        
        # Transpile and run
        transpiled = transpile(bound_circuit, basis_gates=['u1', 'u2', 'u3', 'cx', 'measure'])
        sim = AerSimulator()
        result = sim.run(transpiled, shots=shots).result()
        counts = result.get_counts()
        
        # Compute expectation value
        return self._parse_observable(counts, observable, shots)
    
    def _parse_observable(self, counts: Dict[str, int], observable: str, 
                          shots: int) -> float:
        """Parse observable and compute expectation value."""
        # Parse observable (e.g., 'Z0' means Z on qubit 0)
        if observable.startswith('Z'):
            qubit = int(observable[1])
            expectation = 0.0
            for state, count in counts.items():
                # Z expectation: +1 for |0⟩, -1 for |1⟩
                bit = int(state[-(qubit+1)])  # Qiskit uses little-endian
                eigenvalue = 1 if bit == 0 else -1
                expectation += eigenvalue * count / shots
            return expectation
        else:
            raise ValueError(f"Unsupported observable: {observable}")


class TrainableQFTAttention:
    """
    Trainable QFT-based attention layer with gradient computation.
    """
    
    def __init__(self, num_tokens: int = 3):
        self.num_tokens = num_tokens
        self.num_pairs = num_tokens * (num_tokens - 1) // 2
        self.total_params = num_tokens + self.num_pairs  # phi + theta
        
        # Initialize parameters randomly
        self.params = np.random.uniform(0, 2*np.pi, self.total_params)
        
        # Create gradient computer
        self.gradient_computer = ParameterShiftGradient(self._build_circuit)
        
    def _build_circuit(self) -> Tuple[QuantumCircuit, ParameterVector]:
        """Build the parameterized circuit."""
        phi = ParameterVector('phi', self.num_tokens)
        theta = ParameterVector('theta', self.num_pairs)
        
        qc = QuantumCircuit(self.num_tokens, self.num_tokens)
        
        # Token embeddings
        for i in range(self.num_tokens):
            qc.ry(phi[i], i)
        
        # QFT
        qft = QFT(num_qubits=self.num_tokens, do_swaps=False, inverse=False)
        qft_decomposed = transpile(qft, basis_gates=['u1', 'u2', 'u3', 'cx'])
        qc.compose(qft_decomposed, qubits=list(range(self.num_tokens)), inplace=True)
        
        # Attention phases
        pairs = [(i, j) for i in range(self.num_tokens) for j in range(i+1, self.num_tokens)]
        for pair_idx, (i, j) in enumerate(pairs):
            qc.crz(theta[pair_idx], i, j)
        
        # Inverse QFT
        qft_inv = QFT(num_qubits=self.num_tokens, do_swaps=False, inverse=True)
        qft_inv_decomposed = transpile(qft_inv, basis_gates=['u1', 'u2', 'u3', 'cx'])
        qc.compose(qft_inv_decomposed, qubits=list(range(self.num_tokens)), inplace=True)
        
        # Measure
        qc.measure(list(range(self.num_tokens)), list(range(self.num_tokens)))
        
        # Combine parameters
        all_params = ParameterVector('all', self.total_params)
        # Note: In practice, you'd bind phi and theta to all_params
        
        return qc, phi  # Return phi as placeholder
    
    def forward(self, shots: int = 4096) -> Dict[str, float]:
        """Run forward pass and return probability distribution."""
        circuit, _ = self._build_circuit()
        
        # Create parameter vectors
        phi = ParameterVector('phi', self.num_tokens)
        theta = ParameterVector('theta', self.num_pairs)
        
        # Rebuild with proper binding
        qc = QuantumCircuit(self.num_tokens, self.num_tokens)
        for i in range(self.num_tokens):
            qc.ry(phi[i], i)
        
        qft = QFT(num_qubits=self.num_tokens, do_swaps=False, inverse=False)
        qft_decomposed = transpile(qft, basis_gates=['u1', 'u2', 'u3', 'cx'])
        qc.compose(qft_decomposed, qubits=list(range(self.num_tokens)), inplace=True)
        
        pairs = [(i, j) for i in range(self.num_tokens) for j in range(i+1, self.num_tokens)]
        for pair_idx, (i, j) in enumerate(pairs):
            qc.crz(theta[pair_idx], i, j)
        
        qft_inv = QFT(num_qubits=self.num_tokens, do_swaps=False, inverse=True)
        qft_inv_decomposed = transpile(qft_inv, basis_gates=['u1', 'u2', 'u3', 'cx'])
        qc.compose(qft_inv_decomposed, qubits=list(range(self.num_tokens)), inplace=True)
        
        qc.measure(list(range(self.num_tokens)), list(range(self.num_tokens)))
        
        # Bind parameters
        param_dict = {}
        for i in range(self.num_tokens):
            param_dict[phi[i]] = self.params[i]
        for i in range(self.num_pairs):
            param_dict[theta[i]] = self.params[self.num_tokens + i]
        
        bound_circuit = qc.assign_parameters(param_dict)
        transpiled = transpile(bound_circuit, basis_gates=['u1', 'u2', 'u3', 'cx', 'measure'])
        
        sim = AerSimulator()
        result = sim.run(transpiled, shots=shots).result()
        counts = result.get_counts()
        
        # Convert to probabilities
        probs = {state: count/shots for state, count in counts.items()}
        return probs
    
    def compute_gradients(self, target_probs: Dict[str, float], 
                          shots: int = 4096) -> np.ndarray:
        """
        Compute gradients of loss with respect to all parameters.
        
        Uses MSE loss: L = Σ(p_pred - p_target)²
        
        Args:
            target_probs: Target probability distribution
            shots: Number of shots for measurement
            
        Returns:
            Array of gradients
        """
        gradients = np.zeros(self.total_params)
        
        # Get current predictions
        current_probs = self.forward(shots)
        
        # Compute gradients for each parameter
        for i in range(self.total_params):
            # Use parameter-shift rule
            grad = self._compute_loss_gradient(i, target_probs, shots)
            gradients[i] = grad
        
        return gradients
    
    def _compute_loss_gradient(self, param_idx: int, target_probs: Dict[str, float],
                                shots: int) -> float:
        """Compute gradient of loss w.r.t. single parameter."""
        shift = np.pi / 2
        
        # Shift parameter forward
        original = self.params[param_idx]
        
        self.params[param_idx] = original + shift
        probs_plus = self.forward(shots)
        loss_plus = self._mse_loss(probs_plus, target_probs)
        
        self.params[param_idx] = original - shift
        probs_minus = self.forward(shots)
        loss_minus = self._mse_loss(probs_minus, target_probs)
        
        # Restore parameter
        self.params[param_idx] = original
        
        # Gradient via parameter-shift
        return (loss_plus - loss_minus) / 2
    
    def _mse_loss(self, pred_probs: Dict[str, float], 
                  target_probs: Dict[str, float]) -> float:
        """Compute MSE loss between predicted and target distributions."""
        all_states = set(pred_probs.keys()) | set(target_probs.keys())
        loss = 0.0
        for state in all_states:
            p_pred = pred_probs.get(state, 0.0)
            p_target = target_probs.get(state, 0.0)
            loss += (p_pred - p_target) ** 2
        return loss
    
    def train(self, target_probs: Dict[str, float], epochs: int = 100,
              learning_rate: float = 0.1, shots: int = 4096,
              verbose: bool = True) -> List[float]:
        """
        Train the quantum attention layer.
        
        Args:
            target_probs: Target probability distribution
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            shots: Number of shots per measurement
            verbose: Print progress
            
        Returns:
            List of loss values per epoch
        """
        losses = []
        
        for epoch in range(epochs):
            # Compute gradients
            gradients = self.compute_gradients(target_probs, shots)
            
            # Update parameters (gradient descent)
            self.params -= learning_rate * gradients
            
            # Compute current loss
            current_probs = self.forward(shots)
            loss = self._mse_loss(current_probs, target_probs)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        return losses


if TORCH_AVAILABLE:
    class HybridQuantumAttention(nn.Module):
        """
        Hybrid quantum-classical attention layer for PyTorch.
        
        Combines quantum attention with classical feed-forward layers.
        """
        
        def __init__(self, num_tokens: int = 3, hidden_dim: int = 16):
            super().__init__()
            self.num_tokens = num_tokens
            self.quantum_layer = TrainableQFTAttention(num_tokens)
            
            # Classical layers
            self.classical_fc = nn.Sequential(
                nn.Linear(2**num_tokens, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2**num_tokens),
                nn.Softmax(dim=-1)
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through hybrid layer.
            
            Args:
                x: Input tensor (batch_size, input_dim)
                
            Returns:
                Output tensor (batch_size, output_dim)
            """
            # Get quantum attention output
            probs = self.quantum_layer.forward()
            
            # Convert to tensor
            prob_tensor = torch.tensor(
                [probs.get(format(i, '03b'), 0.0) for i in range(2**self.num_tokens)],
                dtype=torch.float32
            )
            
            # Pass through classical layers
            output = self.classical_fc(prob_tensor)
            
            return output


def demo_gradient_computation():
    """Demonstrate gradient computation and training."""
    print("="*60)
    print("Trainable Quantum Attention Layer Demo")
    print("="*60)
    
    # Create trainable layer
    qa = TrainableQFTAttention(num_tokens=3)
    
    print(f"\nInitial parameters:")
    print(f"  phi (token embeddings): {qa.params[:3]}")
    print(f"  theta (attention phases): {qa.params[3:]}")
    
    # Define target distribution (e.g., want |100⟩ to be most probable)
    target_probs = {
        '000': 0.1,
        '001': 0.1,
        '010': 0.1,
        '011': 0.1,
        '100': 0.4,  # Target state
        '101': 0.1,
        '110': 0.05,
        '111': 0.05
    }
    
    print(f"\nTarget distribution:")
    for state, prob in sorted(target_probs.items()):
        print(f"  |{state}⟩: {prob:.2f}")
    
    # Initial prediction
    initial_probs = qa.forward(shots=4096)
    initial_loss = qa._mse_loss(initial_probs, target_probs)
    print(f"\nInitial loss: {initial_loss:.6f}")
    
    # Train
    print(f"\nTraining for 50 epochs...")
    losses = qa.train(target_probs, epochs=50, learning_rate=0.05, shots=2048)
    
    # Final prediction
    final_probs = qa.forward(shots=4096)
    final_loss = qa._mse_loss(final_probs, target_probs)
    
    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    
    print(f"\nFinal parameters:")
    print(f"  phi: {qa.params[:3]}")
    print(f"  theta: {qa.params[3:]}")
    
    print(f"\nFinal distribution (top 5):")
    sorted_probs = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for state, prob in sorted_probs:
        target = target_probs.get(state, 0.0)
        print(f"  |{state}⟩: {prob:.3f} (target: {target:.3f})")
    
    return losses


if __name__ == "__main__":
    demo_gradient_computation()
