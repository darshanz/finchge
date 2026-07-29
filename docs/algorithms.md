## Algorithms

Algorithms define how FinchGE applies evolutionary operators to a population. They coordinate selection, crossover, mutation, evaluation, sorting, and replacement for each generation.

In FinchGE, algorithms are lower-level evolutionary components. A full Grammatical Evolution run is coordinated by the engine, which handles initialization, stopping criteria, checkpointing, logging, and final result construction.

## Supported Algorithms

| Algorithm | Objective type | Main use |
| --- | --- | --- |
| [`GeneticAlgorithm`][finchge.algorithm.GeneticAlgorithm] | Single-objective | Standard grammatical evolution with one scalar fitness objective. |
| [`SteadyStateGA`][finchge.algorithm.SteadyStateGA] | Single-objective | Genitor-style steady-state replacement; offspring enter the population immediately. |
| [`IslandGA`][finchge.algorithm.IslandGA] | Single-objective | Parallel sub-populations with periodic migration to preserve diversity. |
| [`MuPlusLambdaES`][finchge.algorithm.MuPlusLambdaES] | Single-objective | (µ+λ)-ES: best µ from µ parents + λ offspring survive each generation. |
| [`MuCommaLambdaES`][finchge.algorithm.MuCommaLambdaES] | Single-objective | (µ,λ)-ES: best µ from λ offspring only; parents are always discarded. |
| [`OnePlusOneES`][finchge.algorithm.OnePlusOneES] | Single-objective | Simplest evolution strategy: one parent produces one offspring; the better survives. |
| [`MemeticGA`][finchge.algorithm.MemeticGA] | Single-objective | GA augmented with local search: each offspring is hill-climbed before replacement. |
| [`CLONALG`][finchge.algorithm.CLONALG] | Single-objective | Clonal selection: top individuals are cloned and hypermutated; best clones survive. |
| [`NSGA2`][finchge.algorithm.NSGA2] | Multi-objective | Pareto-based optimization using non-dominated sorting and crowding distance. |
| [`NSGA3`][finchge.algorithm.NSGA3] | Multi-objective | Many-objective optimization using reference points. |

These are the algorithms shipped with FinchGE, not an exhaustive list of what's possible.
Any custom evolutionary loop can be added by subclassing [`BaseAlgorithmSO`][finchge.algorithm.BaseAlgorithmSO]
(single-objective) or [`BaseAlgorithmMO`][finchge.algorithm.BaseAlgorithmMO] (multi-objective)
and implementing `evolve_one_generation`.



## GeneticAlgorithm

[`GeneticAlgorithm`][finchge.algorithm.GeneticAlgorithm] is the standard single-objective algorithm. It selects parents, applies crossover and mutation, evaluates offspring, sorts individuals by fitness, and applies a replacement strategy to form the next generation.

This is the usual starting point when the experiment has one fitness value, such as error, accuracy, reward, or expression complexity.

Typical operator choices include:

- tournament or rank selection
- one-point, two-point, or uniform crossover
- integer-flip mutation
- generational or elitist replacement

## SteadyStateGA

[`SteadyStateGA`][finchge.algorithm.SteadyStateGA] implements the Genitor model
(Whitley, 1989). Instead of building an entire new generation at once, it
selects two parents, produces two offspring, evaluates them, and immediately
replaces the two worst individuals in the live population. This repeats until
`population_size` offspring have been processed.

Because offspring enter the population immediately, they are eligible as
parents or replacement targets later in the same generation. No explicit
elitism is needed: the best individuals survive naturally since only the
worst individuals are ever displaced.

Use `SteadyStateGA` when you want finer-grained selection pressure than a
generational GA, or when you want to avoid the "generation boundary" effect
where all offspring must wait for the whole population to be rebuilt.

## IslandGA

[`IslandGA`][finchge.algorithm.IslandGA] partitions the population into
`num_islands` independent sub-populations that evolve in isolation using
standard GA operators. Every `migration_interval` generations, the best
`migration_size` individuals from each island migrate to the next island in
a ring topology, replacing that island's worst individuals.

Isolation lets different islands explore different regions of the search
space; periodic migration shares good solutions across islands to prevent
any one of them from stagnating in a local optimum.

`population_size` must be divisible by `num_islands`, and `migration_size`
must be smaller than the resulting island size.

Use `IslandGA` when a single population tends to converge prematurely and
you want to trade some convergence speed for sustained diversity.


## MuPlusLambdaES

[`MuPlusLambdaES`][finchge.algorithm.MuPlusLambdaES] implements the (µ+λ) evolution
strategy. Each generation, λ offspring are produced from the current µ parents by
mutation only (no crossover). The next generation is the best µ individuals from
the combined µ+λ pool. Parents can survive alongside their offspring.

This is elitist, ie, the best individual seen so far can never be lost. It is well
suited to GE because invalid individuals (those that produce no phenotype) are
naturally excluded. Only evaluated individuals with usable fitness compete.

A common starting point is λ = µ (equal parents and offspring). Increasing λ
relative to µ produces more offspring per generation and increases exploration.

```python
from finchge.algorithm import MuPlusLambdaES

algorithm = MuPlusLambdaES(
    lambda_=100,          # offspring per generation
    mutation=mutation,
    fitness_evaluator=fitness_evaluator,
)
```

## MuCommaLambdaES

[`MuCommaLambdaES`][finchge.algorithm.MuCommaLambdaES] implements the (µ,λ)
evolution strategy. Each generation, λ offspring are produced by mutation. Only
the best µ of these λ offspring form the next generation — parents are always
discarded, regardless of their fitness.

Requires λ ≥ µ so the next generation can always be filled from offspring alone.
A common choice is λ = 2µ to 7µ.

Unlike (µ+λ)-ES, this strategy is **non-elitist**: a good solution can be lost
if none of its offspring are competitive. This makes the algorithm more explorative
and better at escaping local optima, at the cost of convergence stability.

```python
from finchge.algorithm import MuCommaLambdaES

algorithm = MuCommaLambdaES(
    lambda_=200,          # must be >= population_size
    mutation=mutation,
    fitness_evaluator=fitness_evaluator,
)
```


## OnePlusOneES

[`OnePlusOneES`][finchge.algorithm.OnePlusOneES] is the simplest possible evolution
strategy: a single parent produces a single offspring via mutation each generation.
The better of the two survives (greedy acceptance). If the offspring is invalid or
has no usable fitness, the parent is always retained.

This is a special case of (µ+λ)-ES with µ=1 and λ=1. It requires
`population_size: 1` in `ge_config.yaml`. Use it as a fast sanity check or
baseline. It makes no assumptions about population structure and is trivial
to analyse theoretically.

```python
from finchge.algorithm import OnePlusOneES

algorithm = OnePlusOneES(
    mutation=mutation,
    fitness_evaluator=fitness_evaluator,
)
```


## MemeticGA

[`MemeticGA`][finchge.algorithm.MemeticGA] combines a standard genetic algorithm
with a local search phase applied to each offspring after mutation. The GA loop
handles global exploration; local search handles exploitation by iteratively
applying the mutation operator and keeping improvements.

The `local_search_probability` parameter controls what fraction of offspring
undergo local search each generation, letting you trade compute budget against
solution quality. Setting it to `1.0` applies local search to every offspring;
`0.0` degrades to a standard GA.

The same mutation operator is used for both the GA mutation step and the local
search step. Any `GEMutationStrategy` works.

```python
from finchge.algorithm import MemeticGA

algorithm = MemeticGA(
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    replacement=replacement,
    elite_size=1,
    fitness_evaluator=fitness_evaluator,
    local_search_steps=5,
    local_search_probability=0.5,
)
```


## CLONALG

[`CLONALG`][finchge.algorithm.CLONALG] implements a clonal selection algorithm
inspired by the biological immune system (de Castro & Von Zuben, 2002).

Each generation, the top `num_select` individuals are selected by fitness rank.
Each is cloned `num_clones` times and subjected to hypermutation, where the
mutation intensity is proportional to fitness rank: the best individual's clones
mutate least (to preserve good solutions) and the worst selected individual's
clones mutate most (to explore). The `hyper_factor` parameter controls the
steepness of this gradient.

After evaluation, the best clone from each group replaces the original individual
if it is an improvement. The rest of the population is carried forward unchanged.

```python
from finchge.algorithm import CLONALG

algorithm = CLONALG(
    num_select=10,        # top individuals to clone each generation
    num_clones=5,         # clones per selected individual
    hyper_factor=0.5,     # hypermutation intensity gradient
    mutation=mutation,
    fitness_evaluator=fitness_evaluator,
)
```



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

Use [`GeneticAlgorithm`][finchge.algorithm.GeneticAlgorithm] when there is one clear optimization target. Use [`SteadyStateGA`][finchge.algorithm.SteadyStateGA] when you want continuous, fine-grained replacement instead of discrete generations. Use [`IslandGA`][finchge.algorithm.IslandGA] when premature convergence is a concern and you want isolated sub-populations with periodic migration. Use [`NSGA2`][finchge.algorithm.NSGA2] when there are two or more objectives and a Pareto front is desired. Use [`NSGA3`][finchge.algorithm.NSGA3] when the experiment has many objectives and reference-point diversity is important.

For multi-objective algorithms, the result should be interpreted as a Pareto set rather than a single best solution. Selecting one final individual from that set is a modeling decision and should be documented as part of the experiment.
