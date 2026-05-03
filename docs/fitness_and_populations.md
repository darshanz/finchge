## Fitness Evaluation

In finchGE fitness evaluation is handled by [```FitnessEvaluator```][finchge.fitness.FitnessEvaluator] class which allows
flexible and convenient fitness evaluation especially for use cases where data and models are involved.
The `FitnessEvaluator` is responsible for managing all aspects of fitness function evaluation.
It coordinates key supporting components including the fitness functions themselves, a cache manager,
parallel executors, and benchmark runners to handle the evaluation process efficiently.
By decoupling fitness evaluation from the main evolutionary loop, FinchGE allows different evaluation strategies
to be applied across diverse problem domains, all while exposing a consistent interface to the rest of the evolutionary machinery.

A fitness evaluator can be created as follows:


```python
fitness_evaluator = FitnessEvaluator(
        runner=phenotype_runner,
        fitness_functions=acc_fitness,
        mapper=mapper,
        parallel_config=ge_config.parallel
    )
```

!!! Note

    In finchGE, tree-based initialization produces individuals without a genotype representation.
    Genotype-based operators (e.g., OnePointCrossover, IntFlipMutation) require genotypes to function correctly.
    To use these operators with tree-based initialization,
    enable genotype encoding by setting `encode_trees=True` in the FitnessEvaluator.



Fitness evaluator handles the [```PhenotypeRunner```][finchge.runners.PhenotypeRunner], fitness functions
and parallel executors to evaluate the individuals. The phenotype runner instances such as
 [`SymbolicRegressionRunner`][finchge.runners.SymbolicRegressionRunner],
[`ControlRunner`][finchge.runners.ControlRunner], [`MLModelRunner`][finchge.runners.MLModelRunner] etc.,
are responsible for running the phenotype against the data to
output the solution or predictions which can be used to evaluate by the fitness functions.


### Fitness Functions

The **fitness function** defines how good a generated solution is by assigning it a numerical score based on the problem objectives.
**Fitness evaluation** is the process of applying this function to each individual in the population,
guiding evolution by favoring higher-quality solutions for selection and reproduction.

FinchGE includes various fitness functions as shown below.

- [```AccuracyFitness```][finchge.fitness.AccuracyFitness] -  Computes accuracy using 'y_pred' and 'y_val' from the context.
- [```MAEFitness```][finchge.fitness.MAEFitness] - Computes Mean Absolute Error using 'y_pred' and 'y_val' from the context.
- [```RMSEFitness```][finchge.fitness.RMSEFitness]  - Computes Root Mean Square Error using 'y_pred' and 'y_val' from the context.
- [```RewardFitness```][finchge.fitness.RewardFitness] -  Fitness function for control problems with reward maximization.


To use custom fitness functions, FinchGE provides a common interface
named [```GEFitnessFunction```][finchge.fitness.GEFitnessFunction].
This standard interface lets users plug in their own evaluation logic without needing to modify any core evolutionary components.


---

## Population and Individuals

In FinchGE, the evolutionary state is explicitly represented through
the [`Individual`][finchge.core.Individual] and
[`Population`][finchge.core.Population]
classes. Each individual encodes a candidate solution derived from the grammar,
while the population manages the collective state of evolution across generations.

Each individual maintains:

- an integer-valued **genotype** (codon sequence)
- a derived **phenotype** produced via grammar expansion
- associated **fitness values**
- auxiliary metadata (e.g., number of codons consumed during mapping)

Populations act as containers for individuals and provide a consistent interface for evaluation,
selection, and replacement.

By maintaining genotypes, phenotypes, and fitness values as separate entities, finchGE enables:

- detailed inspection of evolutionary dynamics
- post hoc analysis of mapping behavior
- reproducibility and traceability of results


---

### Population Initialisation Strategies

Population initialisation plays a critical role in Grammatical Evolution (GE). The structure, depth, and diversity of initial individuals strongly influence search efficiency, premature convergence, and solution quality.

FinchGE implements several academically established initialisation methods. These methods generate individuals either from **linear genomes** (canonical GE) or **derivation trees** (tree-based GE). The tree-based methods follow established research in Genetic Programming and Grammatical Evolution literature.



#### Random Genome Initialisation

Random Genome Initialisation is the classical initialisation strategy used in canonical Grammatical Evolution.
[`RandomGenomeInitialiser`][finchge.initialisation.RandomGenomeInitialiser] creates individuals
by randomly generating linear genomes (codon sequences),
which are later mapped to phenotypes using grammar-based decoding.

##### Key Characteristics

- Grammar-independent during genome generation.
- Provides high genotype diversity.
- May produce invalid or incomplete phenotypes depending on mapping constraints.
- Common baseline initialisation method in GE systems.


#### Random Initialisation with Valids and No Duplicates (RVD)
[`RVDInitializer`][finchge.initialisation.RandomGenomeInitialiser] is an implementation of
Random Initialisation with Valids and No Duplicates (RVD)
[[Nicolau, 2017](https://link.springer.com/article/10.1007/s10710-017-9309-9#Bib1)], a genome-based
population initialisation strategy designed to improve the quality of randomly
generated individuals in Grammatical Evolution (GE).
Unlike standard random genome initialisation, RVD ensures that all generated individuals map to
valid phenotypes and that duplicate phenotypes are avoided during initialisation.
This approach improves initial population diversity while retaining the simplicity
and efficiency of uniform random genome sampling.

##### Key Characteristics
- Ensures all generated individuals produce valid phenotypes.
- Prevents duplicate phenotypes during initialisation.
- Maintains uniform random genome sampling behaviour.
- Improves diversity and reduces wasted evaluations caused by invalid individuals.
- Particularly useful for canonical GE representations where genotype-to-phenotype mapping may produce invalid derivations.


#### Grow Initialisation

Grow Initialisation constructs derivation trees by expanding grammar productions without enforcing
strict structural constraints. Terminals may appear at any depth provided grammar
and depth limits are respected.

##### Key Characteristics

- Produces irregularly shaped trees.
- Promotes structural diversity.
- May generate shallow trees with fewer internal nodes.
- Useful for avoiding structural bias toward large or uniform trees.

#### Full Initialisation

Full Initialisation constructs trees where all internal nodes expand using productions containing at least one recursive or non-terminal symbol until a predefined maximum depth is reached. Terminal productions are only selected at the maximum depth.

##### Key Characteristics

- Produces uniformly structured trees.
- Ensures individuals reach the intended depth limit.
- May fail for grammars that cannot support strictly full derivations.
- Provides controlled structural complexity.

####  Ramped Half-and-Half Initialization (Sensible Initialisation)

Ramped Half-and-Half Initialization in FinchGE is implemented based on Sensible Initialisation
[[Ryan and Azad, 2003](https://link.springer.com/chapter/10.1007/3-540-36599-0_37)] which
is the Grammatical Evolution adaptation of Koza’s Ramped Half-and-Half (RHH) method.
It generates individuals using both Full and Grow strategies across a range of depth limits.
In GE, recursive grammar productions assume the role of GP function nodes in controlling tree expansion.
Grammar analysis is used to ensure depth feasibility and termination.
In finchGE, [`RHHInitializer`][finchge.initialisation.RHHInitialiser] class implements this initialisation.

##### Key Characteristics

- Combines structural diversity from Grow with depth control from Full.
- Distributes individuals across multiple depth levels.
- Widely adopted in GE research and implementations such as PonyGE2.
- Considered the standard initialisation strategy for grammar-based evolutionary systems.

#### Position-Independent Grow (PI-Grow)

Position-Independent Grow extends Grow Initialisation by expanding tree nodes in random positional order rather than strict depth-first order. This reduces positional bias during tree construction.

##### Key Characteristics

- Produces more structurally varied trees compared to standard Grow.
- Helps reduce bias introduced by expansion order.
- Often improves population diversity and search exploration.

#### Probabilistic Tree Creation 2 (PTC2)

Probabilistic Tree Creation 2 (PTC2) is a stochastic tree generation algorithm
designed to provide precise control over the size (number of expansions)
and shape of initial solutions.
Originally proposed for Genetic Programming by [Luke (2000)](https://ieeexplore.ieee.org/abstract/document/873237),
it was adapted for Grammatical Evolution (GE) by [Harper (2012)](https://ieeexplore.ieee.org/abstract/document/5586336)
and further refined by [Nicolau (2017)](https://link.springer.com/article/10.1007/s10710-017-9309-9).

Unlike traditional depth-based routines, PTC2 focuses on the number of grammar expansions performed.
It maintains a **frontier** of active non-terminals and selects the next node to expand **uniformly at random**.
This process effectively eliminates the structural and left-recursive biases
common in standard depth-first initialisation.

??? note "Refined PTC2 vs. PTC2D"

    The algorithm supports two distinct operational modes based on the presence of structural constraints.

    === "Refined PTC2 (Unconstrained)"

        Operates without a maximum tree depth limit. It has been demonstrated that
        this refined version consistently outperforms other routines [Nicolau (2017)](https://link.springer.com/article/10.1007/s10710-017-9309-9).
        By removing depth constraints, the algorithm samples a wider variety of "skinnier"
        and more diverse tree shapes that are highly suited for the linear genetic operators used in GE.

    === "PTC2D (Depth-Limited)"

        Incorporates a strict maximum depth constraint. While this ensures trees remain within specific structural bounds,
        it tends to produce "bushier," denser trees [Nicolau (2017)](https://link.springer.com/article/10.1007/s10710-017-9309-9).
        This variant uses feasibility checks—based on the minimum number of expansions
        required to terminate. This ensures every branch can close within both the size budget and the depth limit.

---

##### Key Characteristics

- Manages the population distribution through the total number of non-terminal expansions rather than just tree depth.
- Prevents the "leftmost" structural bias found in traditional mapping by randomly selecting the next expansion point from the frontier.
- Uses pre-calculated grammar analysis and minimum termination costs to ensure every generated individual is valid and satisfies all constraints.
- Produces a more uniform distribution of tree shapes and solution lengths, leading to improved search efficiency and better generalization.

> **Note:** "The best results were obtained by a refined version of the PTC2 algorithm... as it sampled a wider variety of tree shapes and solution lengths." [Nicolau (2017)](https://link.springer.com/article/10.1007/s10710-017-9309-9).


#### Ramped PTC2 Initialisation
Ramped PTC2 initialisation distributes individuals across a range of target tree sizes using
PTC-based generation methods. This approach extends ramped initialisation concepts
commonly used in Genetic Programming by combining structural diversity with probabilistic size control.

In FinchGE, ramped PTC2 supports both size-only (PTC2) and size-and-depth constrained (PTC2D) variants,
allowing flexible population initialisation tailored to experimental requirements.

##### Key Characteristics

- Promotes structural diversity by sampling individuals across multiple size levels.
- Maintains probabilistic tree construction properties.
- Supports both unconstrained and depth-constrained PTC variants.
- Reduces premature convergence by increasing variation in structural complexity.

??? "References"

      - Ryan, C., Collins, J.J. and Neill, M.O., 1998, April. Grammatical evolution: Evolving programs for an arbitrary language. In European conference on genetic programming (pp. 83-96). Berlin, Heidelberg: Springer Berlin Heidelberg.
      - O’Neil, M. and Ryan, C., 2003. Grammatical evolution. In Grammatical Evolution: Evolutionary Automatic Programming in an Arbitrary Language (pp. 33-47). Boston, MA: Springer US.      - Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection.* MIT Press.
      - Ryan, C. and Azad, R.M.A., 2003, July. Sensible initialisation in grammatical evolution. In GECCO (pp. 142-145). Menlo Park: AAAI.
      - Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E. and O'Neill, M., 2017, July. Ponyge2: Grammatical evolution in python. In Proceedings of the genetic and evolutionary computation conference companion (pp. 1194-1201).
      - Nicolau, M., 2017. Understanding grammatical evolution: initialisation. Genetic Programming and Evolvable Machines, 18(4), pp.467-507.
      - Luke, S., 2002. Two fast tree-creation algorithms for genetic programming. IEEE Transactions on Evolutionary Computation, 4(3), pp.274-283.
      - Harper, R., 2010, July. GE, explosive grammars and the lasting legacy of bad initialisation. In IEEE Congress on Evolutionary Computation (pp. 1-8). IEEE.

---


#### Using Tree-Based Initializers in FinchGE

This example shows the general workflow for population initialisation
using grammar-based derivation trees. The same pattern applies to all
tree-based initialisers (RHH, Full, Grow, PI-Grow, PTC2, etc.).

```python

# Define Grammar
from finchge.grammar import Grammar

grammar = Grammar("""
<expr> ::= <expr> <op> <expr>
         | <var>

<op> ::= + | - | * | /
<var> ::= x | y | 1 | 2
""")

# Create Tree Generator
# Responsible for constructing grammar-valid derivation trees
from finchge.grammar.tree_generator import TreeGenerator

tree_generator = TreeGenerator(
    grammar=grammar,
    max_tree_depth=6
)

# Select Initializer
from finchge.initialisation.initialisers import RHHInitialiser

initialiser = RHHInitialiser(
    init_min_depth=2,
    init_max_depth=4,
    population_size=20,
    random_state=42
)

# Tree-based initialisers require the TreeGenerator
initialiser.set_tree_generator(tree_generator)

# Initialize single individual
ind = initialiser.initialise()

phenotype = ind.phenotype
print("Phenotype:", phenotype)



```

Initializers can also be declared in a config file with the `init_type` key under the `ge` section:


```yaml

ge:
  init_type: random_genome
  genome_length: 100
  codon_size: 127
```

All initialisers support config-based construction through the `from_config()` method.
For example, [`RandomGenomeInitialiser`][finchge.initialisation.RandomGenomeInitialiser] can be used as follows.

```python
from finchge.initialisation import  RandomGenomeInitialiser
from finchge.config import  FinchConfig

config = FinchConfig.from_yaml("config.yaml")
initialiser = RandomGenomeInitialiser.from_config(config=config)

```

Note: To initialise using config, all parameters required by the selected initializer must be provided in the config file.
