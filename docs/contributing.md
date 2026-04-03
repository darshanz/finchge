# Contributing to finchGE

We welcome and value contributions to finchGE. To report a bug or submit a fix, please use our GitHub issue tracker. Meaningful contributions come in many forms:

- Reporting issues with clear, reproducible steps
- Submitting bug fixes
- Implementing new features (for major changes, please open an issue for discussion first)


## How to Contribute


<div class="admonition info">
<p class="admonition-title">First Time? </p>
<p>If you're new to open source, check out <a href="https://github.com/firstcontributions/first-contributions">First Contributions</a></p>
</div>


### 1. Report Bugs

Found a bug? Please [open an issue](https://github.com/finchGE/finchge/issues) with:

- **Description**: What happened vs what you expected
- **Steps to Reproduce**: Minimal code example
- **Environment**: Python version, OS, finchGE version
- **Error Message**: Complete traceback if any

### 2. Suggest Features
Have an idea? [Open an issue](https://github.com/finchGE/finchge/issues) with:
- **Use Case**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Alternatives Considered**: Other approaches you've thought about

### 3. Fix Bugs or Add Features
Ready to code? Follow these steps:

#### Step 1: Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/finchge.git
cd finchge
```

#### Step 2: Set Up Development Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import finchge; print(f'finchGE {finchge.__version__} ready for development!')"
```

#### Step 3: Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

#### Step 4: Make Your Changes
- Follow the [code style guidelines](#code-style)
- Write tests for new functionality
- Update documentation if needed

#### Step 5: Test Your Changes
```bash
# Run the test suite
pytest tests/ -v

# Run specific tests
pytest tests/test_grammars.py -v

# Check code coverage
pytest --cov=finchge tests/

# Run linting
black . --check
flake8 finchge/ tests/
```

#### Step 6: Commit Your Changes
```bash
# Add changed files
git add .

# Commit with descriptive message
git commit -m "feat: add new grammar parser"
# or
git commit -m "fix: resolve issue with fitness evaluation"
```

Use [conventional commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `style:` Code style changes
- `ci:` CI/CD changes
- `chore:` Maintenance tasks

#### Step 7: Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then go to the [GitHub repository](https://github.com/finchGE/finchge) and create a Pull Request.

### 4. Improve Documentation

Documentation is crucial! You can:

- Fix typos or clarify existing docs
- Add examples or tutorials
- Improve API documentation

Documentation files are in the `docs/` directory.

### 5. Review Pull Requests
Help review others' contributions! Look for:

- Code quality and style
- Test coverage
- Documentation updates
- Performance considerations


## Code Style

### Python Code
We use:

- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting
- **PEP 8** as style guide

We use pre-commit to make sure commits are formatted & linted. Please use ```pre-commit``` or run following tools before committing.

run ./scripts/format.sh once and it'll be installed
Run formatting before committing:
```bash

black .
isort .

flake8 finchge/ tests/
```


### Docstrings
Use Google-style docstrings:
```python
def function_name(param1: type, param2: type) -> return_type:
    """Brief description of function.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> function_name(1, 2)
        3
    """
```

## Testing

We are constantly improving the library with tests. We encourage you to write tests for your contributions.

## Documentation

During current beta relase the documents may not be up-to-date, however we are constantly improving the documentation. Any contribution to documentation are welcome.

## Recommended Development Workflow

1. **Sync your fork** with upstream:
   ```bash
   git remote add upstream https://github.com/finchGE/finchge.git
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```
2. **Create a new branch** for your changes
3. **Make your changes** with tests and documentation
4. **Run tests** to ensure nothing breaks
5. **Submit a Pull Request** with clear description


## Thank You!

Your contributions make open source awesome. Whether you're fixing a typo or adding a major feature, every contribution is appreciated!
