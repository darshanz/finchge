## Symbolic Regression

#### Symbolic Expression

[`SymbolicExpression`][finchge.symbolic.SymbolicExpression] class provides a lightweight interface for parsing,
validating, and evaluating symbolic mathematical expressions
used in symbolic regression workflows.

The class parses string-based mathematical expressions using Python's `ast` module and evaluates them numerically with FinchGE's protected mathematical operators. It supports multi-variable expressions, constant expressions, NumPy-style column syntax such as `x[:, 0]`, and automatic detection of required input features.

##### Key Features

- Parse mathematical expressions from strings (e.g., `sin(x0) + x1**2`)
- Automatic variable detection (`x0`, `x1`, `x2`, ... OR in array slice format `x[: 0]`, `x[: 1]`, `x[: 2]`, ...)
- Protected numerical functions for common symbolic regression operators
- Structural complexity metrics (node count and depth)
- Restricted AST validation to reject unsupported Python syntax
- Safe handling of constant-only expressions

##### Basic Usage

```python
from finchge.symbolic.expression import SymbolicExpression

phenotype = "sin(x0) + x1**2"
expr = SymbolicExpression(phenotype)
y = expr.eval(X)

```

#### GERegressor

[`GERegressor`][finchge.symbolic.GERegressor] is a *scikit-learn compatible estimator for symbolic regression*
built on top of [`GrammaticalEvolution`][finchge.core.GrammaticalEvolution] class.
It searches for mathematical expressions that best fit a dataset by evolving programs defined by a grammar.

The estimator follows the familiar **`fit` / `predict`** interface used in scikit-learn, making it easier to work with symbolic regression.

##### How it works

[`GERegressor`][finchge.symbolic.GERegressor] evolves candidate expressions using a grammar that defines the space of valid mathematical programs.
Each individual in the population represents a genotype that is mapped to a symbolic expression (phenotype),
which is then evaluated on the training data using one or more fitness functions.

After the evolutionary run finishes, the estimator stores the results and selects an individual
whose expression will be used for prediction.

##### Basic usage

```python

model = GERegressor(grammar=grammar,
                    config=config,
                    fitness_functions=fitness_fn)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

```

##### Selecting a model

For *single-objective optimization*, the best individual found during evolution is automatically selected.

For *multi-objective optimization*, multiple trade-off solutions may exist (e.g., accuracy vs. expression complexity).
In this case, a model can be manually selected:

```python
model.select_individual(individual)
```

This allows us to choose expressions that balance accuracy, simplicity,
or interpretability depending on their use case.
