# Installation

Get finchGE up and running in minutes.
Follow the steps below to install and verify your setup.

## Quick Install

??? note "Set up a virtual environment (recommended)"

    Using a virtual environment helps avoid dependency conflicts.

    === "Python venv"

        ```bash
        python -m venv finchge-env
        source finchge-env/bin/activate   # Linux / macOS
        finchge-env\Scripts\activate      # Windows
        python -m pip install finchge
        ```

    === "Conda"

        ```bash
        conda create -n finchge-env python=3.10
        conda activate finchge-env
        python -m pip install finchge
        ```



The quickest way to get started:

[![PyPI](https://img.shields.io/pypi/v/finchge?t=1234567890&color=blue)](https://pypi.org/project/finchge/)

```bash
pip install finchge
```

Verify the installation:

```python
import finchge
print(f"finchGE {finchge.__version__} installed successfully")
```

---

## Installation Options

### Standard Installation
```bash
# Latest stable version from PyPI
pip install finchge

# Specific version (e.g., beta release)
pip install finchge==1.0.1-beta.5
```

### With Optional Dependencies

finchGE supports optional integrations with pytorch for use cases such as Hyperparameter search, Neural Architecture Search etc.:

```bash

# With PyTorch support (for neural architecture search)
pip install finchge[pytorch]

```

### From Source (Development)

If you want to contribute or need the latest features:

```bash
# Clone the repository
git clone https://github.com/finchGE/finchge.git
cd finchge

# Install setup tools if not already installed
pip install -U pip setuptools wheel


# Install in development mode
pip install -e ".[dev]"

# Or install with specific optional dependencies
pip install -e ".[pytorch]"
```

### Using Conda (Alternative)

```bash
# Create and activate a new environment
conda create -n finchge python=3.9
conda activate finchge

# Install via pip in conda environment
pip install finchge
```

---

## System Requirements

### Python Versions
finchGE supports and  the following Python versions 3.10+.
Please check our Github Repo for any updates or known compatibility issues.

[![Python](https://img.shields.io/pypi/pyversions/finchge?t=1234567890&color=blue)](https://pypi.org/project/finchge/)

### Dependencies
Dependencies are installed during installation.

```yaml
numpy: ">=1.21.0"      # Numerical operations
matplotlib: ">=3.5.0"  # Visualization
pandas: ">=1.3.0"      # Data handling (optional but recommended)
tqdm: ">=4.0.0"        # Progress bars
scikit-learn: ">=1.0.0"   # Machine learning integration and Evaluation Metrics
```

#### Optional Dependencies
```yaml
torch: ">=1.9.0"          # PyTorch integration
jupyter: ">=1.0.0"        # Notebook support
```

---

## Next Steps

After successful installation:

1. **Try the [Getting Started](getting_started.md) Guide** - Run your first Grammatical Evolution program
2. **Explore [Examples](examples.md)** - See example use cases
3. **Check the [API docReference](api.md)** - Learn about all available features

---

## Need Help?

If you encounter issues during installation:

1. **Check the [GitHub Issues](https://github.com/finchGE/finchge/issues)** - See if others have similar problems
2. **Create a New Issue** - Report your specific problem
3. **Check Python Version** - Ensure you're using Python 3.8 or higher
4. **Update pip** - Run `pip install --upgrade pip`

---

<div class="admonition tip">
<p class="admonition-title">Pro Tip</p>
<p>For hassle-free use and reproducibility, pin your working dependencies in a <code>requirements.txt</code> file:</p>

```txt
finchge==1.0.1-beta.5
numpy==1.24.0
scikit-learn==1.3.0
pandas==2.0.0
matplotlib==3.7.0
```

Install with: <code>pip install -r requirements.txt</code>

<p>This ensures consistent results across different environments and over time.</p>
</div>

---

**Ready to start?**    [Getting Started](getting_started.md)
