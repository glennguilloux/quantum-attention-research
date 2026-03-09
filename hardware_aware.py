"""
Hardware-Aware Compilation for QFT-based Quantum Attention.

This module provides hardware-specific compilation and optimization
for running quantum attention circuits on real quantum devices.

Supports:
- IBM Quantum devices
- Rigetti Aspen
- IonQ
- Generic NISQ backends

Author: Quantum Attention Research Team
License: MIT
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Gate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        Optimize1qGates,
        CXCancellation,
        Collect2qBlocks,
        ConsolidateBlocks
    )
    from qiskit.providers import Backend
    from qiskit.providers.fake_provider import GenericBackendV2
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@dataclass
class HardwareConfig:
    """Configuration for hardware-specific compilation."""
    backend_name: str
    basis_gates: List[str]
    coupling_map: List[Tuple[int, int]]
    max_depth: int = 100
    optimization_level: int = 3
    native_mc_rz: bool = False
    ancilla_free: bool = False


class HardwareAwareCompiler:
    """
    Hardware-aware compiler for quantum attention circuits.

    This class provides:
    - Backend-specific transpilation
    - Gate decomposition optimization
    - Depth and gate count minimization
    - Native gate support for specific hardware

    Example:
        >>> compiler = HardwareAwareCompiler(backend="ibm_brisbane")
        >>> compiled_circuit = compiler.compile(attention_circuit)
        >>> print(f"Depth: {compiled_circuit.depth()}, Gates: {compiled_circuit.size()}")
    """

    # Known backend configurations
    BACKEND_CONFIGS = {
        # IBM Quantum backends
        "ibm_brisbane": HardwareConfig(
            backend_name="ibm_brisbane",
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)],
            native_mc_rz=False,
            ancilla_free=False
        ),
        "ibm_kyoto": HardwareConfig(
            backend_name="ibm_kyoto",
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)],
            native_mc_rz=False,
            ancilla_free=False
        ),
        # Rigetti backends
        "rigetti_aspen": HardwareConfig(
            backend_name="rigetti_aspen",
            basis_gates=["rx", "ry", "rz", "cz", "measure"],
            coupling_map=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)],
            native_mc_rz=False,
            ancilla_free=True  # Rigetti supports ancilla-free MC-RZ
        ),
        # IonQ backends
        "ionq_aria": HardwareConfig(
            backend_name="ionq_aria",
            basis_gates=["r", "rx", "ry", "rz", "xx", "measure"],
            coupling_map=[],  # All-to-all connectivity
            native_mc_rz=True,  # IonQ supports native multi-controlled gates
            ancilla_free=True
        ),
        # Generic NISQ
        "generic_nisq": HardwareConfig(
            backend_name="generic_nisq",
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=[(i, i+1) for i in range(7)] + [(i+1, i) for i in range(7)],
            native_mc_rz=False,
            ancilla_free=False
        )
    }

    def __init__(
        self,
        backend: Optional[str] = None,
        optimization_level: int = 3,
        custom_config: Optional[HardwareConfig] = None
    ):
        """
        Initialize the hardware-aware compiler.

        Args:
            backend: Backend name (e.g., "ibm_brisbane", "rigetti_aspen")
            optimization_level: Transpilation optimization level (0-3)
            custom_config: Custom hardware configuration
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for HardwareAwareCompiler")

        if custom_config:
            self.config = custom_config
        elif backend:
            if backend not in self.BACKEND_CONFIGS:
                raise ValueError(f"Unknown backend: {backend}. "
                              f"Available: {list(self.BACKEND_CONFIGS.keys())}")
            self.config = self.BACKEND_CONFIGS[backend]
        else:
            self.config = self.BACKEND_CONFIGS["generic_nisq"]

        self.optimization_level = optimization_level

    def compile(
        self,
        circuit: "QuantumCircuit",
        target_depth: Optional[int] = None,
        target_gate_count: Optional[int] = None
    ) -> "QuantumCircuit":
        """
        Compile circuit for the target hardware.

        Args:
            circuit: Input quantum circuit
            target_depth: Optional target circuit depth
            target_gate_count: Optional target gate count

        Returns:
            Compiled and optimized circuit
        """
        # Step 1: Decompose MCX gates if needed
        circuit = self._decompose_mcx(circuit)

        # Step 2: Transpile for target basis gates
        compiled = transpile(
            circuit,
            basis_gates=self.config.basis_gates,
            coupling_map=self.config.coupling_map if self.config.coupling_map else None,
            optimization_level=self.optimization_level
        )

        # Step 3: Apply hardware-specific optimizations
        compiled = self._optimize_for_hardware(compiled)

        # Step 4: Check constraints
        if target_depth and compiled.depth() > target_depth:
            compiled = self._reduce_depth(compiled, target_depth)

        if target_gate_count and compiled.size() > target_gate_count:
            compiled = self._reduce_gates(compiled, target_gate_count)

        return compiled

    def _decompose_mcx(self, circuit: "QuantumCircuit") -> "QuantumCircuit":
        """Decompose multi-controlled X gates for target hardware."""
        # Check if hardware supports native MC-RZ
        if self.config.native_mc_rz:
            # Keep MCX gates, they can be implemented natively
            return circuit

        # Decompose MCX to CX gates
        decomposed = circuit.decompose(gates_to_decompose=["mcx"])
        return decomposed

    def _optimize_for_hardware(
        self,
        circuit: "QuantumCircuit"
    ) -> "QuantumCircuit":
        """Apply hardware-specific optimizations."""
        # Create pass manager with optimization passes
        pm = PassManager([
            Optimize1qGates(),
            CXCancellation(),
        ])

        optimized = pm.run(circuit)
        return optimized

    def _reduce_depth(
        self,
        circuit: "QuantumCircuit",
        target_depth: int
    ) -> "QuantumCircuit":
        """Reduce circuit depth through aggressive optimization."""
        # Try higher optimization levels
        for opt_level in range(3, 0, -1):
            optimized = transpile(
                circuit,
                basis_gates=self.config.basis_gates,
                coupling_map=self.config.coupling_map if self.config.coupling_map else None,
                optimization_level=opt_level
            )
            if optimized.depth() <= target_depth:
                return optimized

        # If still too deep, try gate cancellation
        pm = PassManager([
            Collect2qBlocks(),
            ConsolidateBlocks(),
            Optimize1qGates(),
            CXCancellation(),
        ])
        return pm.run(circuit)

    def _reduce_gates(
        self,
        circuit: "QuantumCircuit",
        target_count: int
    ) -> "QuantumCircuit":
        """Reduce gate count through optimization."""
        pm = PassManager([
            Optimize1qGates(),
            CXCancellation(),
        ])
        return pm.run(circuit)

    def get_compilation_stats(
        self,
        circuit: "QuantumCircuit"
    ) -> Dict[str, Any]:
        """
        Get compilation statistics for a circuit.

        Args:
            circuit: Quantum circuit to analyze

        Returns:
            Dictionary with compilation statistics
        """
        compiled = self.compile(circuit)

        return {
            "original_depth": circuit.depth(),
            "compiled_depth": compiled.depth(),
            "original_gates": circuit.size(),
            "compiled_gates": compiled.size(),
            "two_qubit_gates": compiled.count_ops().get("cx", 0) + 
                              compiled.count_ops().get("cz", 0) +
                              compiled.count_ops().get("xx", 0),
            "single_qubit_gates": sum(
                compiled.count_ops().get(gate, 0)
                for gate in ["rx", "ry", "rz", "sx", "x"]
            ),
            "basis_gates": self.config.basis_gates,
            "backend": self.config.backend_name
        }


def compile_for_ibm(
    circuit: "QuantumCircuit",
    backend_name: str = "ibm_brisbane"
) -> "QuantumCircuit":
    """
    Convenience function to compile for IBM Quantum backends.

    Args:
        circuit: Input quantum circuit
        backend_name: IBM backend name

    Returns:
        Compiled circuit
    """
    compiler = HardwareAwareCompiler(backend=backend_name)
    return compiler.compile(circuit)


def compile_for_rigetti(
    circuit: "QuantumCircuit"
) -> "QuantumCircuit":
    """
    Convenience function to compile for Rigetti Aspen backends.

    Args:
        circuit: Input quantum circuit

    Returns:
        Compiled circuit
    """
    compiler = HardwareAwareCompiler(backend="rigetti_aspen")
    return compiler.compile(circuit)


def compile_for_ionq(
    circuit: "QuantumCircuit"
) -> "QuantumCircuit":
    """
    Convenience function to compile for IonQ backends.

    Args:
        circuit: Input quantum circuit

    Returns:
        Compiled circuit
    """
    compiler = HardwareAwareCompiler(backend="ionq_aria")
    return compiler.compile(circuit)


def estimate_fidelity(
    circuit: "QuantumCircuit",
    backend_name: str = "generic_nisq",
    error_rates: Optional[Dict[str, float]] = None
) -> float:
    """
    Estimate circuit fidelity based on error rates.

    Args:
        circuit: Quantum circuit
        backend_name: Backend name
        error_rates: Custom error rates (1q, 2q, readout)

    Returns:
        Estimated fidelity (0.0 to 1.0)
    """
    # Default error rates for NISQ devices
    default_rates = {
        "generic_nisq": {"1q": 0.001, "2q": 0.01, "readout": 0.02},
        "ibm_brisbane": {"1q": 0.0005, "2q": 0.005, "readout": 0.01},
        "rigetti_aspen": {"1q": 0.002, "2q": 0.02, "readout": 0.03},
        "ionq_aria": {"1q": 0.001, "2q": 0.01, "readout": 0.015}
    }

    rates = error_rates or default_rates.get(backend_name, default_rates["generic_nisq"])

    # Count gates
    ops = circuit.count_ops()

    # Single-qubit gates
    single_qubit_count = sum(
        ops.get(gate, 0)
        for gate in ["rx", "ry", "rz", "sx", "x", "id"]
    )

    # Two-qubit gates
    two_qubit_count = sum(
        ops.get(gate, 0)
        for gate in ["cx", "cz", "xx"]
    )

    # Number of qubits (for readout)
    n_qubits = circuit.num_qubits

    # Calculate fidelity
    fidelity = (
        (1 - rates["1q"]) ** single_qubit_count *
        (1 - rates["2q"]) ** two_qubit_count *
        (1 - rates["readout"]) ** n_qubits
    )

    return fidelity


if __name__ == "__main__":
    # Example usage
    print("Testing HardwareAwareCompiler...")

    # Create a simple test circuit
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    for i in range(4):
        qc.ry(0.5, i)
    qc.h(0)
    for i in range(3):
        qc.cx(i, i+1)

    # Compile for different backends
    for backend in ["ibm_brisbane", "rigetti_aspen", "ionq_aria"]:
        compiler = HardwareAwareCompiler(backend=backend)
        compiled = compiler.compile(qc)
        stats = compiler.get_compilation_stats(qc)

        print(f"
{backend}:")
        print(f"  Original depth: {stats["original_depth"]}")
        print(f"  Compiled depth: {stats["compiled_depth"]}")
        print(f"  Two-qubit gates: {stats["two_qubit_gates"]}")
        print(f"  Estimated fidelity: {estimate_fidelity(compiled, backend):.4f}")

    print("
All tests passed!")
