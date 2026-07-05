

# FinchGE: A Modular Grammatical Evolution Library

[![PyPI](https://img.shields.io/pypi/v/finchge?t=1234567890&color=blue)](https://pypi.org/project/finchge/)
[![Python](https://img.shields.io/pypi/pyversions/finchge?t=1234567890&color=blue)](https://pypi.org/project/finchge/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/readthedocs/finchge?color=blue)](https://finchge.readthedocs.io/)
[![Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/finchGE/finchge)
[![Tests](https://github.com/finchGE/finchge/actions/workflows/tests.yml/badge.svg)](https://github.com/finchGE/finchge/actions)


FinchGE is a modern Python library for grammar-constrained evolutionary search, built around grammatical evolution, modular operators, reproducible experiments, and benchmark-driven research workflows.

## Features

- Define grammars using BNF-style syntax
- Supports standard genetic operations: selection, crossover, mutation, replacement
- Flexible fitness evaluation for various problem domains
- Modular and extensible design allowing conveniently plugin custom components
- Easy-to-read in-built logging and visualization
- Intuitive API with extensive documentation and examples
- Benchmark suite for regression, logic and control problems


## Who is FinchGE for?

- Researchers experimenting with grammatical evolution and grammar-guided search.
- Python users who want to evolve programs, expressions, rules, or structured solutions from BNF grammars.
- Symbolic regression users who need grammar constraints or multi-objective search.
- Students learning genotype-to-phenotype mapping, derivation trees, and evolutionary search.
- Developers building custom evolutionary workflows with custom fitness, operators, or benchmarks.


## Why FinchGE?

FinchGE is designed around grammar-first evolutionary workflows:

- BNF-style grammars define valid programs, expressions, rules, or policies.
- Genotypes map to phenotypes through explicit GE mapping.
- Derivation trees and mapping metadata can be inspected.
- Operators, fitness functions, initializers, runners, and algorithms are modular.
- Runs can be configured, logged, checkpointed, and reproduced.
- Benchmark suites are included for symbolic regression, logic, and control problems.


## Installation
[![PyPI](https://img.shields.io/pypi/v/finchge?t=1234567890&color=blue)](https://pypi.org/project/finchge/)
```bash
python -m pip install finchge
```
For further details on installation, please check. [Installation](installation.md)


## Quick Example

Using **finchGE** is straightforward.

Step 1. Define grammar

```python
grammar_file = "grammar.bnf"
grammar = Grammar.from_file(grammar_file)
```

Step 2. Define a Fitness Evaluator

```python
# Initialize Fitness Evaluator
fitness_evaluator = FitnessEvaluator(
    fitness_functions=StringMatchFitness(target="hello"),
    mapper=GenotypeMapper(grammar=grammar)
)

```

Step 3. Create [`GrammaticalEvolution`][finchge.core.GrammaticalEvolution] instance and run

```python
ge = GrammaticalEvolution(config=FinchConfig.default(),
                           grammar=grammar,
                           fitness_evaluator=fitness_evaluator)
ge.run()

```

For further details and more advanced usage, please check. [Getting Started](getting_started.md), [API Reference](api.md) and [Examples](examples.md)

## Status

FinchGE is currently beta software. The core library is usable for experiments, but APIs may still evolve as the project moves toward a stable release.
