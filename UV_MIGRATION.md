# UV Migration Guide

This document explains the migration to `uv` and the new project structure.

## What Changed

ComfyStream has been updated to use modern Python packaging tools:

1. **Build System**: Migrated from `setuptools` to `hatchling`
2. **Package Manager**: Now supports `uv` for faster dependency management
3. **Build Script**: Added `uv run build` command to build the UI
4. **Dependencies**: Dependencies are now managed in both `pyproject.toml` and `requirements.txt`

## Why UV?

[uv](https://docs.astral.sh/uv/) is a modern, fast Python package manager that:

- Automatically manages virtual environments
- Installs packages 10-100x faster than pip
- Has better dependency resolution
- Compatible with existing Python projects

## Installation

### Installing UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip
pip install uv
```

### Installing ComfyStream

```bash
# Install from GitHub
uv pip install git+https://github.com/livepeer/comfystream.git

# Install locally
cd comfystream
uv pip install -e .

# Build the UI
uv run build
```

## Backward Compatibility

The `requirements.txt` file is still maintained and works as before:

```bash
# Traditional pip install still works
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt
```

## Managing Dependencies

Dependencies are defined in both `pyproject.toml` (primary) and `requirements.txt` (for compatibility).

### Syncing Dependencies

If you edit `requirements.txt`, sync it to `pyproject.toml`:

```bash
python sync_requirements.py --from-requirements
```

If you edit `pyproject.toml`, sync it to `requirements.txt`:

```bash
python sync_requirements.py --from-pyproject
```

### Adding New Dependencies

**Option 1: Edit pyproject.toml (recommended)**

Add the dependency to the `dependencies` array in `pyproject.toml`:

```toml
dependencies = [
    "existing-package",
    "new-package>=1.0.0",
]
```

Then sync to requirements.txt:

```bash
python sync_requirements.py --from-pyproject
```

**Option 2: Edit requirements.txt**

Add the dependency to `requirements.txt`:

```
new-package>=1.0.0
```

Then sync to pyproject.toml:

```bash
python sync_requirements.py --from-requirements
```

## Build Script

The new `build.py` script builds the UI (Next.js application):

```bash
# Run with uv
uv run build

# Or run directly
python build.py
```

This script:
1. Installs npm dependencies
2. Builds the Next.js UI
3. Verifies the build was successful

## Project Scripts

Scripts defined in `pyproject.toml` can be run with `uv run`:

```toml
[project.scripts]
build = "build:main"
```

Run with:

```bash
uv run build
```

## Development

### Installing Dev Dependencies

```bash
uv pip install -e ".[dev]"
```

### Code Formatting

The project now includes Ruff for linting and formatting:

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Format code
ruff format .

# Lint code
ruff check .
```

## Differences from pip

### Virtual Environments

`uv` automatically creates and manages virtual environments:

```bash
# uv automatically handles the venv
uv pip install -e .

# vs pip requiring manual venv creation
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Speed

`uv` is significantly faster:

```bash
# uv is 10-100x faster
time uv pip install -r requirements.txt

# vs pip
time pip install -r requirements.txt
```

## Troubleshooting

### "Command not found: uv"

Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Git Dependencies Not Installing

Some git dependencies may require authentication. Ensure you have git configured:

```bash
git config --global credential.helper store
```

### Build Fails

Ensure Node.js is installed:

```bash
node --version  # Should be >= 18.x
npm --version
```

## Migration Checklist

- [ ] Install uv
- [ ] Test `uv pip install -e .`
- [ ] Test `uv run build`
- [ ] Verify `requirements.txt` still works
- [ ] Update CI/CD pipelines (if any)
- [ ] Update documentation

## References

- [uv Documentation](https://docs.astral.sh/uv/)
- [Hatchling Documentation](https://hatch.pypa.io/latest/)
- [PEP 517 - Build System](https://peps.python.org/pep-0517/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)

