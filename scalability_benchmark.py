#!/usr/bin/env python3
"""
Scalability Benchmark for QFT-based Quantum Attention
======================================================
Analyzes circuit depth, gate count, and qubit requirements
for different token counts to assess NISQ feasibility.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def build_attention_circuit(num_tokens: int, include_ancilla: bool = False) -> QuantumCircuit:
    """
    Build a QFT-based attention circuit for given number of tokens.
    
    Args:
        num_tokens: Number of tokens (qubits)
        include_ancilla: Whether to include ancilla qubits for MC-RZ
        
    Returns:
        QuantumCircuit
    """
    num_pairs = num_tokens * (num_tokens - 1) // 2
    
    if include_ancilla:
        total_qubits = num_tokens + num_pairs
    else:
        total_qubits = num_tokens
    
    qc = QuantumCircuit(total_qubits, num_tokens)
    
    # Token embeddings (RY rotations)
    for i in range(num_tokens):
        qc.ry(0.5, i)
    
    # QFT
    qft = QFT(num_qubits=num_tokens, do_swaps=False, inverse=False)
    qft_decomposed = transpile(qft, basis_gates=['u1', 'u2', 'u3', 'cx'])
    qc.compose(qft_decomposed, qubits=list(range(num_tokens)), inplace=True)
    
    # Attention phases (CRZ gates)
    pairs = [(i, j) for i in range(num_tokens) for j in range(i+1, num_tokens)]
    for pair_idx, (i, j) in enumerate(pairs):
        if include_ancilla:
            # Use ancilla for multi-controlled RZ
            ancilla = num_tokens + pair_idx
            qc.ccx(i, j, ancilla)
            qc.crz(0.5, ancilla, j)
            qc.ccx(i, j, ancilla)  # Uncompute
        else:
            qc.crz(0.5, i, j)
    
    # Inverse QFT
    qft_inv = QFT(num_qubits=num_tokens, do_swaps=False, inverse=True)
    qft_inv_decomposed = transpile(qft_inv, basis_gates=['u1', 'u2', 'u3', 'cx'])
    qc.compose(qft_inv_decomposed, qubits=list(range(num_tokens)), inplace=True)
    
    # Measure
    qc.measure(list(range(num_tokens)), list(range(num_tokens)))
    
    return qc


def benchmark_circuit(circuit: QuantumCircuit) -> dict:
    """
    Benchmark a quantum circuit.
    
    Returns:
        Dictionary with benchmark metrics
    """
    # Transpile to basis gates
    transpiled = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx', 'measure'])
    
    # Count operations
    ops = transpiled.count_ops()
    
    return {
        'qubits': transpiled.num_qubits,
        'depth': transpiled.depth(),
        'total_gates': len(transpiled),
        'cx_gates': ops.get('cx', 0),
        'u_gates': ops.get('u1', 0) + ops.get('u2', 0) + ops.get('u3', 0),
        'measurements': ops.get('measure', 0)
    }


def run_simulation_benchmark(circuit: QuantumCircuit, shots: int = 1024) -> dict:
    """
    Benchmark simulation time.
    """
    sim = AerSimulator()
    
    start = time.time()
    result = sim.run(circuit, shots=shots).result()
    elapsed = time.time() - start
    
    return {
        'simulation_time': elapsed,
        'shots': shots,
        'time_per_shot': elapsed / shots
    }


def run_full_benchmark(max_tokens: int = 8, include_ancilla_versions: bool = True):
    """
    Run full scalability benchmark.
    
    Args:
        max_tokens: Maximum number of tokens to test
        include_ancilla_versions: Whether to test MC-RZ versions
    """
    print("="*80)
    print("QFT-BASED QUANTUM ATTENTION SCALABILITY BENCHMARK")
    print("="*80)
    print()
    
    results = []
    
    for n in range(2, max_tokens + 1):
        print(f"\nBenchmarking {n} tokens...")
        
        # Simple CRZ version
        circuit_simple = build_attention_circuit(n, include_ancilla=False)
        bench_simple = benchmark_circuit(circuit_simple)
        
        result = {
            'tokens': n,
            'type': 'Simple CRZ',
            **bench_simple
        }
        results.append(result)
        
        print(f"  Simple CRZ: depth={bench_simple['depth']}, CX={bench_simple['cx_gates']}")
        
        # MC-RZ version (if requested and feasible)
        if include_ancilla_versions and n <= 5:  # Limit for practical reasons
            circuit_mc = build_attention_circuit(n, include_ancilla=True)
            bench_mc = benchmark_circuit(circuit_mc)
            
            result_mc = {
                'tokens': n,
                'type': 'True MC-RZ',
                **bench_mc
            }
            results.append(result_mc)
            
            print(f"  True MC-RZ: depth={bench_mc['depth']}, CX={bench_mc['cx_gates']}, qubits={bench_mc['qubits']}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    
    # Header
    print(f"{'Tokens':>6} | {'Type':>12} | {'Qubits':>6} | {'Depth':>6} | {'CX':>6} | {'Gates':>6}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['tokens']:>6} | {r['type']:>12} | {r['qubits']:>6} | {r['depth']:>6} | {r['cx_gates']:>6} | {r['total_gates']:>6}")
    
    # NISQ feasibility analysis
    print("\n" + "="*80)
    print("NISQ FEASIBILITY ANALYSIS")
    print("="*80)
    print()
    
    # Typical NISQ constraints
    nisq_depth_limit = 200  # Typical coherence limit
    nisq_cx_limit = 100     # Typical gate fidelity limit
    
    print(f"Typical NISQ constraints: depth < {nisq_depth_limit}, CX gates < {nisq_cx_limit}")
    print()
    
    feasible = [r for r in results if r['type'] == 'Simple CRZ' 
                and r['depth'] < nisq_depth_limit 
                and r['cx_gates'] < nisq_cx_limit]
    
    if feasible:
        max_feasible = max(r['tokens'] for r in feasible)
        print(f"✓ Maximum feasible tokens (Simple CRZ): {max_feasible}")
    else:
        print("✗ No feasible configurations found")
    
    # Scaling analysis
    print("\nScaling Analysis:")
    simple_results = [r for r in results if r['type'] == 'Simple CRZ']
    
    if len(simple_results) >= 3:
        # Fit quadratic scaling
        tokens = np.array([r['tokens'] for r in simple_results])
        depths = np.array([r['depth'] for r in simple_results])
        
        # Quadratic fit
        coeffs = np.polyfit(tokens, depths, 2)
        
        print(f"  Depth scaling: depth ≈ {coeffs[0]:.1f}n² + {coeffs[1]:.1f}n + {coeffs[2]:.1f}")
        print(f"  (Quadratic scaling due to QFT and pairwise attention)")
    
    return results


def estimate_hardware_requirements(num_tokens: int):
    """
    Estimate hardware requirements for a given number of tokens.
    """
    num_pairs = num_tokens * (num_tokens - 1) // 2
    
    # Simple CRZ version
    simple_qubits = num_tokens
    simple_depth = int(10 * num_tokens**2 + 5 * num_tokens)  # Approximate
    simple_cx = int(4 * num_tokens**2)  # Approximate
    
    # MC-RZ version
    mc_qubits = num_tokens + num_pairs
    mc_depth = int(15 * num_tokens**2 + 10 * num_tokens)  # Approximate
    mc_cx = int(8 * num_tokens**2)  # Approximate
    
    print(f"\nHardware Requirements for {num_tokens} tokens:")
    print(f"\n  Simple CRZ version:")
    print(f"    Qubits: {simple_qubits}")
    print(f"    Estimated depth: ~{simple_depth}")
    print(f"    Estimated CX gates: ~{simple_cx}")
    
    print(f"\n  True MC-RZ version:")
    print(f"    Qubits: {mc_qubits}")
    print(f"    Estimated depth: ~{mc_depth}")
    print(f"    Estimated CX gates: ~{mc_cx}")
    
    # Hardware recommendations
    print(f"\n  Hardware recommendations:")
    if simple_depth < 100:
        print(f"    ✓ Suitable for current NISQ devices")
    elif simple_depth < 200:
        print(f"    ⚠ May work on advanced NISQ devices with error mitigation")
    else:
        print(f"    ✗ Requires fault-tolerant quantum computers")


if __name__ == "__main__":
    # Run benchmark
    results = run_full_benchmark(max_tokens=8)
    
    # Estimate for larger systems
    print("\n" + "="*80)
    print("HARDWARE REQUIREMENTS ESTIMATION")
    print("="*80)
    
    for n in [3, 5, 8, 10]:
        estimate_hardware_requirements(n)
