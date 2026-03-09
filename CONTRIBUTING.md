# Contributing to Quantum Attention Research

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/quantum-attention-research.git
cd quantum-attention-research
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and return types
- Write docstrings for all public functions and classes
- Maximum line length: 100 characters

## Testing

Run tests with pytest:
```bash
pytest tests/ -v --cov=.
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request with a clear description

## Code Review

All submissions require review by maintainers. Be responsive to feedback and be prepared to make changes.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
