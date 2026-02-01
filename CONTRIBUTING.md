# Contributing to AlayaLite

Thank you for your interest in contributing to AlayaLite! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming and inclusive community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/AlayaLite.git
   cd AlayaLite
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/AlayaDB-AI/AlayaLite.git
   ```

## Development Setup

### Prerequisites

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake >= 3.15
- Python >= 3.8
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Conan 2.x (C++ package manager)

### Build from Source

```bash
# Install Python dev dependencies
make dev-install

# Build release version
make build

# Run tests
make test
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install them before making changes:

```bash
uvx pre-commit install
uvx pre-commit install --hook-type commit-msg
```

## Making Changes

1. **Create a new branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** and ensure:
   - Code follows the project style guidelines
   - All tests pass (`make test`)
   - Linting passes (`make lint`)

3. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

### Allowed Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring |
| `perf` | Performance improvements |
| `test` | Adding or updating tests |
| `build` | Build system changes |
| `ci` | CI/CD configuration changes |
| `chore` | Other changes (dependencies, etc.) |
| `revert` | Revert a previous commit |

### Examples

```bash
feat: add SQ4 quantization support
fix: resolve memory leak in graph search
docs: update installation instructions
refactor: simplify distance calculation logic
```

## Pull Request Process

1. **Create an issue first** to discuss the feature or bug fix
2. **Ensure all checks pass**:
   - CI builds successfully
   - All tests pass
   - Code coverage is maintained
3. **Update documentation** if needed
4. **Request review** from maintainers
5. **Address feedback** and update your PR as needed

### PR Title Format

Follow the same convention as commit messages:
```
feat: add support for cosine similarity metric
```

## Code Style

### C++ Style

- Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with modifications
- Use `clang-format` (automatically applied via pre-commit)
- Maximum line length: 100 characters
- Use C++20 features where appropriate

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use `ruff` for formatting and linting
- Use type hints for function signatures

### File Organization

```
include/           # C++ header-only library
  ├── index/       # Index implementations
  ├── space/       # Vector space abstractions
  ├── simd/        # SIMD optimizations
  └── utils/       # Utility functions

python/
  ├── src/         # Python SDK source
  └── tests/       # Python tests

tests/             # C++ tests
```

## Questions?

If you have any questions, feel free to:
- Open an issue on GitHub
- Contact us at dev@alayadb.ai

Thank you for contributing to AlayaLite!
