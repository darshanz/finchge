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

Selection operators decide which individuals are used as parents. Most strategies use scalar fitness values; Lexicase-based methods additionally require per-case fitness data from the fitness evaluator. Tournament, roulette-wheel, rank, and truncation selection are appropriate for single-objective runs. The NSGA selectors are designed for multi-objective runs and expect Pareto-ranking metadata on individuals.

### TournamentSelection

[`TournamentSelection`][finchge.operators.selection.TournamentSelection] samples `tournament_size` individuals at random without replacement and selects the best among them (default `tournament_size=3`). This repeats until the required number of parents is assembled. Selection pressure scales directly with tournament size: larger tournaments favour high-fitness individuals more strongly but reduce the chance of lower-fitness individuals contributing to the next generation.

### RouletteWheelSelection

[`RouletteWheelSelection`][finchge.operators.selection.RouletteWheelSelection] assigns each individual a selection probability proportional to its fitness. Fitness values are shifted to be non-negative before computing weights, and inverted for minimization problems. If all shifted weights collapse to zero, the operator falls back to uniform random selection and emits a warning. The known weakness of fitness-proportionate selection is sensitivity to fitness scaling: a small number of highly fit individuals can dominate early in a run, compressing the effective diversity of the selected pool.

### RankSelection

[`RankSelection`][finchge.operators.selection.RankSelection] assigns selection weights by ordinal rank rather than raw fitness, using Baker's linear rank formula. The `selection_pressure` parameter (default 1.5), constrained to [1.0, 2.0], controls the weight ratio between the best and worst individuals: 1.0 gives uniform selection, 2.0 assigns zero weight to the worst individual. This decouples selection pressure from the magnitude of fitness differences, making rank selection more stable than roulette-wheel across varied fitness landscapes and different stages of a run.

### TruncationSelection

[`TruncationSelection`][finchge.operators.selection.TruncationSelection] sorts the population by fitness and restricts parent candidates to the top `truncation_threshold` fraction (default 0.5), then samples from that pool uniformly with replacement. It is deterministic in which individuals qualify and imposes strong selection pressure, particularly at low threshold values.

### NSGA2TournamentSelection

[`NSGA2TournamentSelection`][finchge.operators.selection.NSGA2TournamentSelection] implements the crowded comparison operator from NSGA-II. In each tournament a candidate with a lower Pareto rank beats one with a higher rank; ties are broken by crowding distance, favouring individuals in less densely populated regions of objective space. An `exploration_prob` parameter (default 0.1) introduces occasional random winners to prevent the population from clustering too aggressively. This selector requires `rank` and `crowding_distance` metadata on individuals, which `NSGA2ElitistReplacement` populates.

### NSGA3TournamentSelection

[`NSGA3TournamentSelection`][finchge.operators.selection.NSGA3TournamentSelection] uses Pareto rank as the sole tournament criterion, with ties resolved randomly rather than by crowding distance. This reflects the NSGA-III design, where diversity is managed through reference-point niching during replacement rather than during parent selection. Requires `rank` metadata on individuals.

### LexicaseSelection

[`LexicaseSelection`][finchge.operators.selection.LexicaseSelection] filters the population case by case in a freshly shuffled order for each parent selection event, retaining only individuals that match the best performance on each case in sequence. Selection stops as soon as one individual remains; if multiple individuals survive all cases, one is picked at random from the survivors. Because each selection event uses a different case ordering, the operator naturally maintains behavioural diversity, favouring specialists rather than just globally high-fitness individuals. It requires per-case fitness data stored under `case_key` in individual metadata, which the fitness evaluator must be configured to supply.

### EpsilonLexicaseSelection

[`EpsilonLexicaseSelection`][finchge.operators.selection.EpsilonLexicaseSelection] extends lexicase selection with a tolerance parameter. Rather than requiring exact equality with the best case performance, an individual survives a case filter if it falls within `epsilon` of the best value. This is useful in symbolic regression and similar domains where fitness differences between near-equivalent candidates can be dominated by floating-point precision rather than genuine behavioural differences. Setting `epsilon` to zero recovers the behaviour of standard `LexicaseSelection`.

## Crossover Operators

Crossover operators combine two parent individuals. Genome-level crossover works on codon sequences; subtree crossover works on derivation trees and requires individuals to carry tree representations.

### OnePointCrossover

[`OnePointCrossover`][finchge.operators.crossover.OnePointCrossover] splits each parent genome at a crossover point and recombines the resulting segments. In `fixed` mode, the same point is chosen for both parents so offspring genome lengths match their parents. In `variable` mode, independent points are drawn for each parent, meaning offspring lengths can differ. The `within_used` flag restricts valid crossover points to the portion of each genome consumed during mapping, which avoids disruption of the unused tail. If genomes are too short to cross, parent copies are returned without error.

### TwoPointCrossover

[`TwoPointCrossover`][finchge.operators.crossover.TwoPointCrossover] generalises one-point crossover by selecting two cut points and exchanging the middle segment between them. In `fixed` mode, the same pair of points applies to both parents and genome lengths are preserved. In `variable` mode, independent point pairs are drawn for each parent, so the swapped segments can differ in length and offspring lengths vary. The `within_used` flag applies in the same way as in one-point crossover.

### UniformCrossover

[`UniformCrossover`][finchge.operators.crossover.UniformCrossover] independently decides at each position whether to swap the corresponding genes between the two parents, with probability 0.5 per position. Swapping is limited to the overlapping used-genome range of both parents when `within_used` is set. Among position-based operators, uniform crossover produces the most genetic mixing, which helps combine building blocks from both parents but can also be disruptive to well-adapted gene sequences.

### SubtreeCrossover

[`SubtreeCrossover`][finchge.operators.crossover.SubtreeCrossover] exchanges subtrees between two derivation trees at grammatically compatible crossover points. Only non-terminal symbols that appear in both trees are eligible; a shared symbol is chosen at random, then one matching node from each tree is selected and their subtrees are swapped. Syntactic validity is guaranteed because the exchanged subtrees are rooted at the same grammar non-terminal. If the two trees share no eligible symbols, the operator returns the original trees without modification.

## Mutation Operators

Mutation operators introduce local changes into individuals. FinchGE includes both genome-level mutations and tree-level mutation.

Genome mutations are useful for standard linear-genome GE. Subtree mutation is useful when working with derivation trees and grammar-aware initialization or variation.

### IntFlipMutation

[`IntFlipMutation`][finchge.operators.mutation.IntFlipMutation] is the standard GE mutation operator. In `per_codon` mode, each codon is independently replaced with a uniform random value with probability `mutation_probability`; when left unset, this defaults to `1/L` where L is the effective genome length, following the classical per-locus mutation convention. In `per_ind` mode, a fixed number of `mutation_events` are applied, each selecting a random position and replacing it. The `within_used` flag restricts mutation to codons consumed during the last mapping, leaving the unused tail unchanged. This is the most studied mutation strategy in the GE literature and a sensible default for most experiments.

### SwapMutation

[`SwapMutation`][finchge.operators.mutation.SwapMutation] rearranges existing codon values rather than replacing them; no new values are introduced. Positions are selected stochastically, shuffled, then swapped in complete pairs. Because GE mapping is sequential, swapping early-position codons can restructure the entire derivation tree while swapping late-position codons has a more local effect. If an odd number of positions is selected, the last is left unswapped by design.

Most useful in late-stage search or seeded populations where structural variation is needed without disturbing the codon value distribution. For general-purpose GE runs, prefer `IntFlipMutation`.

### GaussianMutation

[`GaussianMutation`][finchge.operators.mutation.GaussianMutation] adds Gaussian noise to selected codons, rounds to the nearest integer, and clamps to zero from below. Unlike `IntFlipMutation`, which replaces a codon with a uniform random value, Gaussian mutation perturbs codons locally, with the magnitude of change controlled by `std_dev`. There is no upper bound clamping: large noise values can produce codons above `codon_size`, which is harmless in practice because GE mapping applies modulo arithmetic against the number of production rules at each expansion step. Most applicable when fine-grained local search near the current solution is preferred over random replacement.

### InversionMutation

[`InversionMutation`][finchge.operators.mutation.InversionMutation] selects a contiguous segment of the genome at random and reverses it in place, with probability `segment_probability`. Like `SwapMutation`, this preserves the multiset of codon values, rearranging positions rather than replacing values. The impact on GE mapping depends on where the inverted segment falls: reversing an early block can restructure the derivation tree substantially, while inverting a late segment typically affects only rarely-reached expansions. The operator fires per individual, not per codon, so a single application inverts at most one segment per call. Biologically, segment inversion is a known chromosomal event and has precedent in GA literature as a building-block preserving operator.

### CyclicMutation

[`CyclicMutation`][finchge.operators.mutation.CyclicMutation] independently selects each genome position with `mutation_probability`. When triggered, the contiguous segment of length `segment_size` (default 3) starting at that position is rotated right by one step: the last element moves to the front. Segments that extend past the genome boundary are skipped without mutation. Within each rotated segment the codon multiset is preserved, but the sequential ordering changes, shifting which grammar productions are invoked at those mapping steps. Multiple overlapping rotations can occur in a single call if adjacent positions are both selected.

### DuplicationMutation

[`DuplicationMutation`][finchge.operators.mutation.DuplicationMutation] copies a source segment of length `segment_size` (default 2) and writes it over a non-overlapping target segment of equal length, with probability `mutation_probability`. The genome length is unchanged because this is an in-place overwrite, not an insertion. Because the target's original values are replaced, the codon multiset is not preserved. The biological analogue is gene duplication, widely considered a primary driver of evolutionary novelty. In GE, repeated codon patterns interact with wrapping: the same codon block being consumed at multiple wraps can reinforce particular grammar paths, making this operator relevant for deep or recursive grammars.

### MultipleMutation

[`MultipleMutation`][finchge.operators.mutation.MultipleMutation] is a meta-operator that selects one strategy from a provided list and delegates to it per call. Probabilities are normalised automatically if they do not sum to one. Exactly one strategy is applied per mutation event. The intended use is mixing complementary operators (for example, running `IntFlipMutation` most of the time with occasional `DuplicationMutation`) without writing a custom operator class. Each constituent strategy retains its own parameters and random state.

### SubtreeMutation

[`SubtreeMutation`][finchge.operators.mutation.SubtreeMutation] operates on derivation trees rather than integer genomes. With probability `mutation_probability`, one or more non-terminal nodes are selected and their subtrees replaced with freshly generated ones up to `mutation_max_depth` levels deep. Candidate nodes are restricted to non-terminals specified at construction, so syntactic validity is maintained throughout. This is the tree analogue of GP subtree mutation, adapted for BNF-guided derivation. Because it operates on the tree representation, it is only applicable when individuals carry derivation trees and is not compatible with genotype-only individuals.

## Replacement Operators

Replacement operators decide which individuals survive into the next generation after offspring have been produced and evaluated. All single-objective strategies accept an `elite_size` argument; elites are always drawn from the old population before any offspring are considered.

### GenerationalReplacement

[`GenerationalReplacement`][finchge.operators.replacement.GenerationalReplacement] replaces the entire population each generation. The `elite_size` best individuals from the old population are preserved, merged with all new offspring, and the combined pool is sorted by fitness to select the final population. This is the standard generational model in the GA literature and gives offspring a fair chance to displace poor elites if the offspring are strong enough.

### SteadyStateReplacement

[`SteadyStateReplacement`][finchge.operators.replacement.SteadyStateReplacement] keeps the `elite_size` best of the old population and fills the remaining slots with the best-ranked offspring. Offspring compete only among themselves for non-elite positions; they do not compete against elites. The distinction from `GenerationalReplacement` matters when offspring quality is mixed: weaker offspring that would lose against elites in a combined sort still get placed here.

### RandomElitistReplacement

[`RandomElitistReplacement`][finchge.operators.replacement.RandomElitistReplacement] preserves elites by fitness and fills the remaining positions by random sampling without replacement from the pool of non-elite old individuals and new offspring. The randomness introduces diversity without any fitness ranking for non-elite slots. Only compatible with single-objective fitness; passing multi-objective fitness lists raises an error.

### NSGA2ElitistReplacement

[`NSGA2ElitistReplacement`][finchge.operators.replacement.NSGA2ElitistReplacement] implements the environmental selection step of NSGA-II. The old and new populations are combined and subjected to fast non-dominated sorting to assign Pareto ranks. The next generation is filled front by front; when a front only partially fits the remaining slots, individuals within it are ranked by crowding distance and those in less crowded objective-space regions are preferred. This balances convergence toward the Pareto front with spatial diversity. The `maximize_flags` list specifies the optimisation direction per objective, supporting mixed maximisation and minimisation problems. Use alongside `NSGA2TournamentSelection`.

## Usage

Operators in FinchGE are instantiated independently and passed to the algorithm constructor. The same pattern applies regardless of which operators are chosen; only the class names and their parameters change.

**Standard single-objective run**

```python
from finchge.algorithm import GeneticAlgorithm
from finchge.operators.selection import TournamentSelection
from finchge.operators.crossover import OnePointCrossover
from finchge.operators.mutation import IntFlipMutation
from finchge.operators.replacement import GenerationalReplacement

ga = GeneticAlgorithm(
    selection=TournamentSelection(max_best=False, tournament_size=3),
    crossover=OnePointCrossover(codon_size=255, crossover_proba=0.8),
    mutation=IntFlipMutation(mutation_probability=None, codon_size=255),
    replacement=GenerationalReplacement(max_best=False),
    elite_size=1,
    fitness_evaluator=fitness_evaluator,
)
```

**Multi-objective run with NSGA-II**

`NSGA2TournamentSelection` and `NSGA2ElitistReplacement` must be used together. The replacement step computes Pareto ranks and crowding distances that the selection step reads.

```python
from finchge.algorithm import NSGA2
from finchge.operators.selection import NSGA2TournamentSelection
from finchge.operators.replacement import NSGA2ElitistReplacement

mo_algorithm = NSGA2(
    selection=NSGA2TournamentSelection(tournament_size=2),
    crossover=OnePointCrossover(codon_size=255, crossover_proba=0.8),
    mutation=IntFlipMutation(mutation_probability=None, codon_size=255),
    replacement=NSGA2ElitistReplacement(maximize_flags=[False, True]),
    elite_size=2,
    fitness_evaluator=fitness_evaluator,
)
```

**Lexicase selection**

`LexicaseSelection` requires the fitness evaluator to be configured with `require_case_data=True` so it stores per-case results on each individual.

```python
from finchge.operators.selection import LexicaseSelection

fitness_evaluator = FitnessEvaluator(
    runner=runner,
    fitness_functions=fitness_fn,
    mapper=mapper,
    require_case_data=True,
)

ga = GeneticAlgorithm(
    selection=LexicaseSelection(),
    crossover=OnePointCrossover(codon_size=255, crossover_proba=0.8),
    mutation=IntFlipMutation(mutation_probability=None, codon_size=255),
    replacement=GenerationalReplacement(max_best=False),
    elite_size=1,
    fitness_evaluator=fitness_evaluator,
)
```

Full working examples covering symbolic regression, control problems, multi-objective optimisation, and lexicase selection are in the `finchge-examples` repository under `examples/`.

## Choosing Operators

Operator choice should match the representation and the experimental goal.

- For simple single-objective GE, a common starting point is tournament selection, one-point or uniform crossover, integer-flip mutation, and elitist or generational replacement.
- For multi-objective runs, prefer NSGA-aware selection and replacement.
- For tree-based initialization and variation, use tree-aware operators such as subtree crossover and subtree mutation.
- For lexicase selection, configure the fitness evaluator to provide case-wise fitness data.

All operators inherit a common random-state pattern, so controlled seeds can be used for reproducible experiments.

??? "References"

    - Ryan, C., Collins, J.J. & O'Neill, M. (1998). Grammatical Evolution: Evolving Programs for an Arbitrary Language. *EuroGP 1998*, LNCS 1391, pp. 83–96. Springer.
    - O'Neill, M. & Ryan, C. (2003). *Grammatical Evolution: Evolutionary Automatic Programming in an Arbitrary Language.* Springer.
    - Baker, J.E. (1985). Adaptive selection methods for genetic algorithms. *Proceedings of ICGA 1985*, pp. 101–111.
    - Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
    - Deb, K. & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.
    - Spector, L. (2012). Assessment of problem modality by differential performance of lexicase selection in genetic programming: a preliminary report. *GECCO Companion 2012*, pp. 401–408.
    - La Cava, W., Spector, L. & Danai, K. (2016). Epsilon-lexicase selection for regression. *GECCO 2016*, pp. 741–748.
    - Koza, J.R. (1992). *Genetic Programming.* MIT Press.
