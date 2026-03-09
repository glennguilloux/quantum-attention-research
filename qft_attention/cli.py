"""CLI tools for QFT-based Quantum Attention.

Provides command-line interfaces for benchmarking, training, and hardware compilation.
"""
import argparse
import json
import sys
from typing import Optional

import numpy as np


def benchmark(args):
    """Run quantum attention benchmarks."""
    from classical_comparison import AttentionComparator
    from scalability_benchmark import ScalabilityBenchmark
    
    print(f"Running benchmark with {args.n_tokens} tokens...")
    
    if args.compare_classical:
        comparator = AttentionComparator()
        result = comparator.compare(
            n_tokens=args.n_tokens,
            embed_dim=args.embed_dim,
            shots=args.shots
        )
        print(f"\n=== Classical vs Quantum Comparison ===")
        print(f"KL Divergence: {result.kl_divergence:.4f}")
        print(f"Correlation: {result.correlation:.4f}")
        print(f"Quantum Entropy: {result.quantum_entropy:.4f}")
        print(f"Classical Entropy: {result.classical_entropy:.4f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'kl_divergence': result.kl_divergence,
                    'correlation': result.correlation,
                    'quantum_entropy': result.quantum_entropy,
                    'classical_entropy': result.classical_entropy
                }, f, indent=2)
            print(f"Results saved to {args.output}")
    
    if args.scalability:
        benchmark = ScalabilityBenchmark()
        results = benchmark.run_full_benchmark(max_tokens=args.max_tokens)
        print(f"\n=== Scalability Benchmark ===")
        for r in results:
            print(f"Tokens: {r['n_tokens']}, Depth: {r['depth']}, Gates: {r['n_gates']}, "
                  f"Fidelity: {r['estimated_fidelity']:.4f}")


def train(args):
    """Train quantum attention parameters."""
    from hybrid_training import HybridTrainer, TrainingConfig
    
    print(f"Starting training for {args.epochs} epochs...")
    
    config = TrainingConfig(
        n_tokens=args.n_tokens,
        embed_dim=args.embed_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        shots=args.shots
    )
    
    trainer = HybridTrainer(config)
    history = trainer.train()
    
    print(f"\n=== Training Complete ===")
    print(f"Initial Loss: {history['losses'][0]:.4f}")
    print(f"Final Loss: {history['losses'][-1]:.4f}")
    print(f"Improvement: {(1 - history['losses'][-1]/history['losses'][0])*100:.2f}%")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {args.output}")


def hardware_compile(args):
    """Compile circuits for quantum hardware."""
    from hardware_aware import HardwareAwareCompiler
    from qft_attention import QFTAttentionLayer
    
    print(f"Compiling for {args.backend} backend...")
    
    # Create attention layer
    layer = QFTAttentionLayer(n_tokens=args.n_tokens, shots=args.shots)
    circuit = layer._build_circuit()
    
    # Compile for hardware
    compiler = HardwareAwareCompiler(backend=args.backend)
    compiled = compiler.compile(circuit, optimization_level=args.optimization_level)
    
    print(f"\n=== Compilation Results ===")
    print(f"Original depth: {circuit.depth()}")
    print(f"Compiled depth: {compiled.depth()}")
    print(f"Original gates: {len(circuit)}")
    print(f"Compiled gates: {len(compiled)}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'backend': args.backend,
                'original_depth': circuit.depth(),
                'compiled_depth': compiled.depth(),
                'original_gates': len(circuit),
                'compiled_gates': len(compiled),
                'qasm': compiled.qasm()
            }, f, indent=2)
        print(f"Compilation results saved to {args.output}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='QFT-based Quantum Attention CLI tools'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--n-tokens', type=int, default=4, help='Number of tokens')
    bench_parser.add_argument('--embed-dim', type=int, default=4, help='Embedding dimension')
    bench_parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    bench_parser.add_argument('--compare-classical', action='store_true', help='Compare with classical attention')
    bench_parser.add_argument('--scalability', action='store_true', help='Run scalability benchmark')
    bench_parser.add_argument('--max-tokens', type=int, default=8, help='Max tokens for scalability')
    bench_parser.add_argument('--output', type=str, help='Output file for results')
    bench_parser.set_defaults(func=benchmark)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train quantum attention')
    train_parser.add_argument('--n-tokens', type=int, default=4, help='Number of tokens')
    train_parser.add_argument('--embed-dim', type=int, default=4, help='Embedding dimension')
    train_parser.add_argument('--shots', type=int, default=512, help='Number of shots')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    train_parser.add_argument('--output', type=str, help='Output file for training history')
    train_parser.set_defaults(func=train)
    
    # Hardware compile command
    hw_parser = subparsers.add_parser('hardware', help='Compile for quantum hardware')
    hw_parser.add_argument('--backend', type=str, default='ibm_brisbane',
                          choices=['ibm_brisbane', 'ibm_kyiv', 'rigetti_aspen', 'ionq'],
                          help='Target backend')
    hw_parser.add_argument('--n-tokens', type=int, default=4, help='Number of tokens')
    hw_parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    hw_parser.add_argument('--optimization-level', type=int, default=2, 
                          choices=[0, 1, 2, 3], help='Optimization level')
    hw_parser.add_argument('--output', type=str, help='Output file for compilation results')
    hw_parser.set_defaults(func=hardware_compile)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
