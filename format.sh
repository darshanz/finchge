#!/usr/bin/env sh
set -eu

echo "Installing development dependencies..."

python -m pip install -q \
  types_tqdm \
  types_tabulate \
  pandas-stubs \
  "black[jupyter]" \
  mypy \
  isort \
  flake8 \
  types-PyYAML

echo "✓ Development dependencies ready"


black .
isort .
flake8 finchge/ tests/
python -m mypy finchge/
