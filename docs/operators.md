## Operators

Operators are the reusable building blocks that control how individuals are selected, modified, and carried into the next generation. FinchGE keeps these operators separate from the algorithm class so experiments can swap one part of the evolutionary process without rewriting the rest.

FinchGE groups operators into four categories:

| Category | Purpose | FinchGE examples |
| --- | --- | --- |
| Selection | Choose parents or survivors from the current population. | `TournamentSelection`, `RouletteWheelSelection`, `RankSelection`, `TruncationSelection`, `LexicaseSelection` |
| Crossover | Recombine two parent individuals to create offspring. | `OnePointCrossover`, `TwoPointCrossover`, `UniformCrossover`, `SubtreeCrossover` |
| Mutation | Modify an individual to introduce new variation. | `IntFlipMutation`, `SwapMutation`, `GaussianMutation`, `InversionMutation`, `CyclicMutation`, `DuplicationMutation`, `SubtreeMutation` |
| Replacement | Decide how offspring and existing individuals form the next generation. | `GenerationalReplacement`, `SteadyStateReplacement`, `RandomElitistReplacement`, `NSGA2ElitistReplacement` |

## Selection Operators

Selection operators decide which individuals are used as parents. Most selection strategies use scalar fitness values, but some methods need additional case-wise data from the fitness evaluator.

Supported selection strategies include:

- [`TournamentSelection`][finchge.operators.selection.TournamentSelection]
- [`RouletteWheelSelection`][finchge.operators.selection.RouletteWheelSelection]
- [`RankSelection`][finchge.operators.selection.RankSelection]
- [`TruncationSelection`][finchge.operators.selection.TruncationSelection]
- [`NSGA2TournamentSelection`][finchge.operators.selection.NSGA2TournamentSelection]
- [`NSGA3TournamentSelection`][finchge.operators.selection.NSGA3TournamentSelection]
- [`LexicaseSelection`][finchge.operators.selection.LexicaseSelection]
- [`EpsilonLexicaseSelection`][finchge.operators.selection.EpsilonLexicaseSelection]

Tournament, roulette-wheel, rank, and truncation selection are common choices for single-objective runs. NSGA-specific tournament selectors are intended for multi-objective runs where rank, crowding distance, or reference-point behavior is part of selection. Lexicase methods are useful when case-wise performance matters, for example when different training cases reward different candidate behaviors.

## Crossover Operators

Crossover operators combine two parent individuals. Genome-level crossover works on codon sequences, while subtree crossover works on derivation trees.

Supported crossover strategies include:

- [`OnePointCrossover`][finchge.operators.crossover.OnePointCrossover]
- [`TwoPointCrossover`][finchge.operators.crossover.TwoPointCrossover]
- [`UniformCrossover`][finchge.operators.crossover.UniformCrossover]
- [`SubtreeCrossover`][finchge.operators.crossover.SubtreeCrossover]

Use genome-level crossover when individuals are represented directly by integer genomes. Use subtree crossover when the experiment is operating on derivation trees or when preserving syntactic structure during variation is important.

## Mutation Operators

Mutation operators introduce local changes into individuals. FinchGE includes both genome-level mutations and tree-level mutation.

Supported mutation strategies include:

- [`IntFlipMutation`][finchge.operators.mutation.IntFlipMutation]
- [`SwapMutation`][finchge.operators.mutation.SwapMutation]
- [`GaussianMutation`][finchge.operators.mutation.GaussianMutation]
- [`InversionMutation`][finchge.operators.mutation.InversionMutation]
- [`CyclicMutation`][finchge.operators.mutation.CyclicMutation]
- [`DuplicationMutation`][finchge.operators.mutation.DuplicationMutation]
- [`MultipleMutation`][finchge.operators.mutation.MultipleMutation]
- [`SubtreeMutation`][finchge.operators.mutation.SubtreeMutation]

Genome mutations are useful for standard linear-genome GE. Subtree mutation is useful when working with derivation trees and grammar-aware initialization or variation.

## Replacement Operators

Replacement operators decide which individuals survive into the next generation after offspring have been produced and evaluated.

Supported replacement strategies include:

- [`GenerationalReplacement`][finchge.operators.replacement.GenerationalReplacement]
- [`SteadyStateReplacement`][finchge.operators.replacement.SteadyStateReplacement]
- [`RandomElitistReplacement`][finchge.operators.replacement.RandomElitistReplacement]
- [`NSGA2ElitistReplacement`][finchge.operators.replacement.NSGA2ElitistReplacement]

Replacement strategy affects selection pressure, elitism, diversity, and reproducibility of evaluation counts. For multi-objective experiments, use replacement methods that preserve Pareto-ranking metadata correctly.

## Choosing Operators

Operator choice should match the representation and the experimental goal.

- For simple single-objective GE, a common starting point is tournament selection, one-point or uniform crossover, integer-flip mutation, and elitist or generational replacement.
- For multi-objective runs, prefer NSGA-aware selection and replacement.
- For tree-based initialization and variation, use tree-aware operators such as subtree crossover and subtree mutation.
- For lexicase selection, configure the fitness evaluator to provide case-wise fitness data.

All operators inherit a common random-state pattern, so controlled seeds can be used for reproducible experiments.
