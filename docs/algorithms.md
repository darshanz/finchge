## Algorithms

Algorithms define how FinchGE applies evolutionary operators to a population. They coordinate selection, crossover, mutation, evaluation, sorting, and replacement for each generation.

In FinchGE, algorithms are lower-level evolutionary components. A full Grammatical Evolution run is coordinated by the engine, which handles initialization, stopping criteria, checkpointing, logging, and final result construction.

## Supported Algorithms

| Algorithm | Objective type | Main use |
| --- | --- | --- |
| [`GeneticAlgorithm`][finchge.algorithm.GeneticAlgorithm] | Single-objective | Standard grammatical evolution with one scalar fitness objective. |
| [`NSGA2`][finchge.algorithm.NSGA2] | Multi-objective | Pareto-based optimization using non-dominated sorting and crowding distance. |
| [`NSGA3`][finchge.algorithm.NSGA3] | Multi-objective | Many-objective optimization using reference points. |

## GeneticAlgorithm

[`GeneticAlgorithm`][finchge.algorithm.GeneticAlgorithm] is the standard single-objective algorithm. It selects parents, applies crossover and mutation, evaluates offspring, sorts individuals by fitness, and applies a replacement strategy to form the next generation.

This is the usual starting point when the experiment has one fitness value, such as error, accuracy, reward, or expression complexity.

Typical operator choices include:

- tournament or rank selection
- one-point, two-point, or uniform crossover
- integer-flip mutation
- generational or elitist replacement

## NSGA-II

[`NSGA2`][finchge.algorithm.NSGA2] is used for multi-objective optimization. Instead of returning a single best individual, it organizes the population into Pareto fronts using non-dominated sorting and uses crowding distance to preserve diversity within each front.

NSGA-II is useful when objectives conflict, for example:

- minimizing prediction error while minimizing expression complexity
- maximizing reward while minimizing program size
- balancing accuracy and interpretability

Use NSGA-II with multi-objective fitness functions and NSGA-aware selection or replacement operators.

## NSGA-III

[`NSGA3`][finchge.algorithm.NSGA3] is designed for many-objective optimization. It uses reference points to help maintain diversity across the objective space when crowding distance alone becomes less effective.

NSGA-III is most useful when experiments have more than two objectives or when the search needs stronger coverage across a many-objective Pareto front.

## Algorithm and Engine Responsibilities

The algorithm evolves one generation at a time. The engine is responsible for the wider experiment workflow.

| Responsibility | Algorithm | Engine |
| --- | :---: | :---: |
| Apply selection, crossover, mutation, and replacement | Yes | No |
| Evaluate offspring during evolution | Yes | No |
| Initialize the first population | No | Yes |
| Manage generation loop and stopping criteria | No | Yes |
| Handle checkpointing and resuming | No | Yes |
| Build final experiment result | No | Yes |

This split keeps algorithms reusable. The same algorithm can be used in different experiment workflows as long as the grammar, mapper, fitness evaluator, and operators are compatible.

## Choosing an Algorithm

Use [`GeneticAlgorithm`][finchge.algorithm.GeneticAlgorithm] when there is one clear optimization target. Use [`NSGA2`][finchge.algorithm.NSGA2] when there are two or more objectives and a Pareto front is desired. Use [`NSGA3`][finchge.algorithm.NSGA3] when the experiment has many objectives and reference-point diversity is important.

For multi-objective algorithms, the result should be interpreted as a Pareto set rather than a single best solution. Selecting one final individual from that set is a modeling decision and should be documented as part of the experiment.
