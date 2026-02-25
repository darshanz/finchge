

# FinchGE: A Modular Grammatical Evolution Library

[![PyPI](https://img.shields.io/pypi/v/finchge?t=1234567890&color=blue)](https://pypi.org/project/finchge/)
[![Python](https://img.shields.io/pypi/pyversions/finchge?t=1234567890&color=blue)](https://pypi.org/project/finchge/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/readthedocs/finchge?color=blue)](https://finchge.readthedocs.io/)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/finchGE/finchge)
[![Tests](https://github.com/finchGE/finchge/actions/workflows/tests.yml/badge.svg)](https://github.com/finchGE/finchge/actions)


FinchGE is a modern, modular, and user-friendly Python library for Grammatical Evolution (GE) - a powerful evolutionary algorithm that uses formal grammars to evolve programs, expressions, and solutions.

## Features

- Define grammars using BNF-style syntax
- Supports standard genetic operations: mutation, crossover, selection
- Flexible fitness evaluation for any problem domain
- Modular and extensible design allowing conveniently plugin custom Algorithms and Operators
- Easy-to-read in-built logging and visualization
- Intuitive API with extensive documentation and examples


## Why finchGE

- Modular and extensible: Plug-and-play mutation,  election, fitness, and search strategies.
- Designed for research and industry: Convenient and flexible API for quicker implementation.

## Installation
[![PyPI](https://img.shields.io/pypi/v/finchge?t=1234567890&color=blue)](https://pypi.org/project/finchge/)
```bash
pip install finchge
```
For further details on installation, please check. [Installation](/installation)


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
    fitness_functions=StringMatchFitness("hello"),
    mapper=GenotypeMapper(grammar=grammar)
)

```

Step 3. Create [`GrammaticalEvolution`](api/#finchge.evolution.GrammaticalEvolution) instance and run

```python
ge_ = GrammaticalEvolution(config=FinchConfig.default(),
                           grammar=grammar,
                           fitness_evaluator=fitness_evaluator)
ge_.run()

```

For further details and more advanced usage, please check. [Getting Started](/getting_started/), [API Reference](/api/) and [Examples](/examples/)

## Development Status
Note: This is version ```1.0.1-alpha.1``` - an alpha release. Expect breaking changes and bugs.

What to expect:
- Bugs and unexpected behavior
- Rapid API changes
- Frequent updates
- Limited test coverage (improving daily)
