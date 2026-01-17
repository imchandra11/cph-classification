# Packaging Guide for cph-classification

This document explains how to build and publish the `cph-classification` package to PyPI.

## Package Structure

```
cph-classification/
├── cph_classification/           # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # CLI entry point
│   └── classification/          # Classification module
│       ├── __init__.py
│       ├── cli.py
│       ├── main.py
│       ├── mainfittest.py
│       ├── dataset.py
│       ├── datamodule.py
│       ├── modelfactory.py
│       ├── modelmodule.py
│       └── callbacks.py
├── pyproject.toml               # Package configuration
├── LICENSE                      # MIT License
├── README.md                    # Package documentation
├── .gitignore                   # Git ignore rules
└── requirements.txt             # Development dependencies (optional)
```

## Building the Package

### 1. Install Build Tools

```bash
pip install --upgrade build twine
```

### 2. Build Distribution Files

```bash
# Build source distribution and wheel
python -m build

# This creates:
# - dist/cph-classification-0.1.0.tar.gz (source distribution)
# - dist/cph-classification-0.1.0-py3-none-any.whl (wheel)
```

## Publishing to PyPI

### 1. Test on Test PyPI First (Recommended)

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ cph-classification
```

### 2. Publish to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*
```

You will need:
- PyPI account (create at https://pypi.org/account/register/)
- API token (create at https://pypi.org/manage/account/token/)

## Installing from Source

Users can install from source:

```bash
# From Git repository
pip install git+https://github.com/imchandra11/cph-classification.git

# From local source
pip install /path/to/cph-classification

# In development mode (editable)
pip install -e /path/to/cph-classification
```

## Updating the Package Version

1. Update version in:
   - `pyproject.toml`: `version = "0.1.0"`
   - `cph_classification/__init__.py`: `__version__ = "0.1.0"`

2. Build and upload new version:

```bash
python -m build
python -m twine upload dist/*
```

## Important Notes

### Config File Path Updates

When using the installed package, config files must use the full module path:

**Old (local development):**
```yaml
model:
  class_path: Classification.modelmodule.ModelModuleCLS
```

**New (installed package):**
```yaml
model:
  class_path: cph_classification.classification.modelmodule.ModelModuleCLS
```

Users need to update their config files when switching from local development to installed package.

### CLI Command

After installation, the CLI is available as:

```bash
cph-classification --config yourproject.yaml
cph-classification fit --config yourproject.yaml
cph-classification test --config yourproject.yaml
```

## Development

For development, install in editable mode:

```bash
pip install -e .
```

This allows you to make changes to the code without reinstalling.

## Verification

After publishing, verify the package:

```bash
# Install from PyPI
pip install cph-classification

# Verify CLI is available
cph-classification --help

# Check package info
pip show cph-classification
```

## Troubleshooting

### Import Errors

If you get import errors after installation:
- Ensure you're using the correct module paths: `cph_classification.classification.*`
- Check that the package is installed: `pip list | grep cph-classification`

### CLI Not Found

If the `cph-classification` command is not found:
- Ensure pip installed the package correctly
- Check that `scripts` are installed: `pip show -f cph-classification | grep Scripts`
- Try using: `python -m cph_classification.cli --config yourproject.yaml`
