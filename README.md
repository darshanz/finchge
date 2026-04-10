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

```bash
# Basic installation
pip install finchge

# With optional dependencies
pip install finchge[pytorch]    # PyTorch support for using pytorch models (for HPO or NAS)
pip install finchge[all]        # All optional dependencies
```


## Quick Example

Using **finchGE** is straightforward.

Step 1. Define grammar

```python
grammar_file = "grammar.bnf"
grammar = Grammar.from_file(grammar_file)
```

Step 2. Define a Fitness Evaluator ; `fitness_evaluator`

```python

fitness_evaluator = FitnessEvaluator(
    fitness_functions=StringMatchFitness(target="hello"),
    mapper=GenotypeMapper(grammar=grammar)
)

```

Step 3. Create `GrammaticalEvolution` instance and run

```python
ge = GrammaticalEvolution(config=FinchConfig.default(),
                           grammar=grammar,
                           fitness_evaluator=fitness_evaluator)
ge.run()

```

For further details and more advanced usage, please check documentation
at [finchge.readthedocs.io](https://finchge.readthedocs.io/), including [Getting Started](https://finchge.readthedocs.io/latest/getting_started/), [API docReference](https://finchge.readthedocs.io/latest/api/) and [Examples](https://finchge.readthedocs.io/latest/examples/)




## Development Status
Note: This is version ```1.0.1-beta.8``` - a beta release. Expect breaking changes and bugs.

What to expect:

- Bugs and unexpected behavior
- Rapid API changes
- Frequent updates
- Limited test coverage (improving daily)

## Contributing

All contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Bug Reports and Feature Requests

Found a bug or have a feature request? Please [open an issue](https://github.com/finchge/finchge/issues) on GitHub.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
