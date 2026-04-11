finchGE is a modular Python framework for **Grammatical Evolution (GE)**, designed to support both
**reproducible experimental workflows** and **fine-grained algorithmic control**.

Rather than encapsulating Grammatical Evolution as a monolithic procedure, finchGE exposes the
core components of the evolutionary process as explicit, composable abstractions. This design
enables users to reason about, modify, and analyze each stage of evolution independently.

This section introduces basic concepts and building blocks of *finchGE* and their roles within the GE
pipeline.

---
## Grammatical Evolution

Grammatical Evolution is a type of genetic programming where solutions are generated and evolved using grammars (like BNF grammars).
It is primarily used for evolving programs or expressions in a structured way, guided by grammar rules and evolutionary operators.
It is often used in symbolic regression, automated programming, and other optimization tasks.


## Grammar

In Grammatical Evolution, the grammar defines the structure of valid solutions.
Grammar serves as a set of mapping rules that translate genetic representations into syntactically correct programs or expressions.
By enforcing these structural constraints, the grammar ensures that
the evolutionary process generates only valid and meaningful solutions, preventing the formation of syntactically incorrect
or nonsensical individuals.

The grammar determines:

- the syntactic validity of solutions
- the structural bias of the search space
- the expressiveness of evolved programs or solutions

Importantly, grammars are **problem-independent** and contain no information about
fitness or optimization objectives.


### Context-free grammars

finchGE uses **Backus-Naur Form (BNF)** to define the grammar that guides Grammatical Evolution.
A grammar specifies the **syntactic rules** that determine how valid individuals are generated.
A context-free grammar consists of the following elements:

- **Terminal symbols** The concrete symbols of the language (e.g., keywords, numbers, operators).
- **Non-terminal symbols** Abstract placeholders that represent classes of expressions or values. These symbols are expanded during generation (e.g., ```<expr>```, ```<operator>```).
- **Production rules** Rules that define how a non-terminal symbol can be replaced by one or more terminals and/or non-terminals.
- **Start symbol** A designated non-terminal symbol from which generation begins.

In BNF-style notation, where alternative productions are separated by the | symbol and non-terminals are written using angle brackets. Terminals can be literals, character ranges, or numeric ranges

In finchGE grammar is defined as a following:

```python
    grammar_str = """
     <expr> ::= <expr> <op> <expr>
                    | <func> ( <expr> )
                    | <var>
                    | <const>
    <op> ::= + | - | * | /
    <func> ::= sin | cos | exp | plog | psqrt
    <const> ::= 0.1 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 0.01 | 0.001
    <var> ::= x0
    """
```

This grammar provides a set of production rules that map genetic representations into
syntactically correct mathematical expressions. Starting from the `<expr>` nonterminal,
the grammar recursively builds expressions by combining subexpressions with arithmetic operators,
applying functions (e.g., `sin`, `cos`, `exp`, `plog`, `psqrt`), or terminating at variables (`x0`) or predefined constants.

Once defined, the grammar is passed to the [`Grammar`][finchge.grammar.Grammar] class, which parses and manages these rules during evolution.

```python
    grammar = Grammar(grammar_str)
```
This grammar ensures that all evolved solutions are syntactically valid by construction, allowing the evolutionary process to focus on optimizing behavior rather than fixing invalid structures.


!!! info "More on Grammars for Grammatical Evolution"
    This documentation provides a practical overview of grammar usage in FinchGE. For a formal and in-depth treatment of Grammatical Evolution and context-free grammars, readers are referred to the original GE literature and standard texts on formal grammars.

    - [Grammatical Evolution: Evolving Programs for an Arbitrary Language](https://dl.acm.org/doi/10.5555/646806.706289)
    - [Handbook of Grammatical Evolution](https://www.springerprofessional.de/en/handbook-of-grammatical-evolution/16114834)
    - [Understanding Grammatical Evolution Design](https://www.researchgate.net/publication/332549000_Understanding_grammatical_evolution_Grammar_design)

### Defining a Grammar in finchGE

FinchGE uses an extended Backus-Naur Form (BNF) syntax for defining grammars. However, finchGE's BNFParser does not
fully support the every syntax of eBNF. The syntax followed by finchGE is as following:

#### Basic Syntax

Each grammar rule follows the format:

```bnf
<non-terminal> ::= production1 | production2 | production3
```

- Non-terminals: Enclosed in angle brackets: `<expr>`, `<digit>`, `<start>`
- Terminals: Quoted strings or unquoted symbols .
- Choice operator: `|` separates alternative productions

For example:

```bnf
<expr> ::= <term> | <expr> + <term>
<term> ::= <factor> | <term> * <factor>
<factor> ::= <number> | ( <expr> )
<number> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7| 8 | 9
```

#### Multi-line Rules

Rules can span multiple lines. This grammar:
```bnf
<expression> ::= <term>
                 + <expression>
                 | <term>
                 - <expression>
                 | <term>

```
is processed exactly the same as:

```bnf
<expression> ::= <term> + <expression> | <term> - <expression> | <term>
```

##### Grammar writing

Although, the BNFParser in finchGE can handle multiple lines correctly,
Following general rules are recommended for smooth parsing with finchGE.

**Rule 1: Start Each Rule with `::=` on Its Own Line**

Always begin a new rule on a line that contains the `::=` operator:

**Correct:**
```bnf

<arithmetic> ::= <expression>
                 | <expression> + <term>
                 | <expression> - <term>

```
**Incorrect:** (Don't put `::=` in the middle of a line)

```bnf
# This won't work as expected
Some text <arithmetic> ::= <expression> | <expression> "+" <term>
```


**Rule 2: Any Line Without `::=` Continues the Current Rule**

Lines without `::=` automatically continue the previous rule:

```bnf
<complex> ::= ( <expression> )
              * <factor>
              | <factor> /
              <term>
              | <value>
```


**Rule 3: Empty Lines Separate Rules Visually**

Use blank lines to separate rules for better readability:

```bnf
# Rule for expressions
<expression> ::= <term>
                 + <expression>
                 | <term> - <expression>
                 | <term>

# Blank line separates rules visually

# Rule for terms
<term> ::= <factor>
           * <term>
           | <factor> / <term>
           | <factor>
```









??? info "Tips: Readability"

    **1. Indent Continuation Lines**

    Indent continuation lines to visually group them with their rule definition:
    ```bnf
        <polynomial> ::= <term>
                     + <polynomial>
                     | <term> - <polynomial>
                     | <term>
                     | ε  # empty string
    ```

    **2. Break Before Choice Operators (`|`)**

    Place choice operators at the beginning of continuation lines for clarity:
    ```bnf
    <function> ::= <name> "(" <arguments> ")"
                   | <name> "()"  # No arguments
                   | <constant>
                   | <variable>
    ```
    **3. Group Related Productions**

    Keep related productions together and separate distinct ones:
    ```bnf
    <statement> ::= <assignment>
                    | <if_statement>
                    | <loop>
                    | <return>

                    # Mathematical operations
                    | <expression> "+" <term>
                    | <expression> "-" <term>
                    | <expression> "*" <factor>

                    # Boolean operations
                    | <expression> "&&" <term>
                    | <expression> "||" <term>
    ```
    **4.  Use Comments to Document Complex Rules**

    Add comments to explain complex multi-line rules:

    ```bnf
    # Complex number operations
    <complex_op> ::= <num> "+" <num> "i"  # a + bi
                     | <num> "-" <num> "i"  # a - bi
                     | <num>  # Pure real
                     | <num> "i"  # Pure imaginary
    <num> ::= 1 | 2
    ```


#### Range Notation

FinchGE supports a rich set of range notations through an extensible handler system.
The parser comes with built-in handlers for common range types:


1. Character class ranges

```bnf
<letters> ::= [a-z]           # Lowercase letters a through z
<upper> ::= [A-Z]             # Uppercase letters A through Z
<digits> ::= [0-9]            # Digits 0 through 9
<alphanum> ::= [A-Za-z0-9_]   # Letters, digits, and underscore
<hex> ::= [A-Fa-f0-9]         # Hexadecimal characters
<custom> ::= [13579]          # Specific characters
<range_mix> ::= [a-zA-Z0-9_]  # Multiple ranges combined

```

2. Numeric Ranges

```bnf
<simple> ::= 10..50           # All integers from 10 to 50 (inclusive)
<negative> ::= -5..5          # Negative to positive range
<stepped> ::= 0..100 step 10  # Increment by 10: 0, 10, 20, ..., 100
<reverse> ::= 10..1           # Descending range: 10, 9, 8, ..., 1
<neg_step> ::= 10..1 step -1  # Negative step
<float> ::= 0.0..1.0          # Floating point ranges
```

3. Character Ranges (elepsis noation)

```bnf

<chars> ::= 'a'..'z'                 # Character range (single quotes)
<chars-step> ::= 'a'..'z' step 2     # Every second character
<unicode> ::= 'α'..'ω'               # Unicode character ranges
<unicode_step> ::=  'क'..'छ' step 2  # Unicode character ranges
<quoted_nums> ::= '10'..'50'         # Quoted numbers (treated as strings)
```


??? tip "Pro Tip : Custom Ranges in Grammar "

     While finchGE supports various range notations in grammar, custom ranges can be used if any special range is required by any problem.
    FinchGE uses a plugin-based architecture for range handling. Using [```RangeHandler```][finchge.grammar.range_handlers.RangeHandler]
    interface for creating custom range handlers which can be registered
    with [`BNFGrammarParser`][finchge.grammar.parser.BNFGrammarParser] as following:
    ```python

    # create a custom range handler
    class MyCustomRangeHandler(RangeHandler):
        # implement required functions: can_handle, expand, pattern methods
        def can_handle(self, token: str) -> bool:
             # Check if this handler can process the token.

        def expand(self, token: str) -> list[str]:
             # Expand the range into individual symbols.

        def pattern(self) -> str:
             # Return regex pattern that matches this range type.

    # Register the handler
    parser = BNFGrammarParser(grammar_string)
    parser.range_registry.register(MyCustomRangeHandler())
    ```

---

#### Symbolic Variable Notation
FinchGE supports symbolic variable notation in the form of zero‑indexed column names in the dataset.
For example, the first column in the dataset is represented as  `x0`.

```bnf

<expr> ::= <expr> <op> <expr> | <var> | <const>
<op> ::= + | -
<var> ::= x0 | x3 | x6
<const> ::= 1.0 | 2.0 | 3.0

```

#### Symbolic Variable Range Notation
To represent ranges of symbolic variables in grammars, FinchGE supports symbolic variable range notation,
allowing evolved mathematical expressions to refer to multiple variables using a compact range syntax.
For example, `x[0..4]` is alternative way to write the columns `x0 | x1 | x2 | x3 | x4`

```bnf

<expr> ::= <expr> <op> <expr> | <var> | <const>
<op> ::= + | -
<var> ::= x[0..4]
<const> ::= 1.0 | 2.0 | 3.0

```

#### Array Slicer Notation

Alternative to Symbolic Variable Range Notation, FinchGE also supports **array slicer notation** in grammars, allowing
evolved mathematical expressions to directly reference the dataset columns using Python/NumPy slicing syntax.

In symbolic regression, we need expressions that can operate on real data.
Array slicers bridge this gap by providing clear mapping from symbolic variables to actual dataset columns
so that the expressions can be executed without parsing transformations. Array slicers, in FinchGE, can be written
in grammar as following:

```bnf

<expr> ::= <expr> <op> <expr> | <var> | <const>
<op> ::= + | -
<var> ::= x[:, 4]
<const> ::= 1.0 | 2.0 | 3.0

```

Here, `x[:, 4]` represents the 4rth column in the data x. Similarly, to allow multiple columns as
production choices, desired columns can be separated by choice operator `|` .

*This syntax is particularly useful when we are interested in selected columns only from the dataset.*

##### Multiple Random Columns
Columns 3, 4, 7, and 8 can be used as production choices as shown in the example grammar below:


```bnf
<expr> ::= <var> | <const>
<var> ::= x[:, 3] | x[:, 7] | x[:, 8]
<const> ::= 1.0 | 2.0
```

##### Columns in a range
When there are several columns to be included and are in a sequence, the rule becomes very long compromising readability.
To include multiple columns in a sequence, finchGE provides a range notation for array slices.
The column ranges can be defined in the grammar using the range notation. This keeps the grammar concise and readable. Ranges in array slicer can
be defined as shown below.

```bnf
<expr> ::= <var> | <const>
<var> ::= x[:, 0..2]
<const> ::= 1.0 | 2.0
```

FinchGE expands the rule `<var> ::= x[:, 0..2]` into:

```bnf
<expr> ::= <var> | <const>
<var> ::= x[:, 0] | x[:, 1] | x[:, 2]
<const> ::= 1.0 | 2.0
```
during parsing process.

Similarly, for less common use cases, stepped range pattern is also supported for array slices. For example `
x[:, 0..10 step 2]` is expanded to `x[:, 0] | x[:, 2] | x[:, 4] | x[:, 6] | x[:, 8] | x[:, 10]`.

### Genotype Mapper

Grammatical Evolution is primarily driven an important process of _genotype-phenotype mapping_.
In FinchGE, this responsibility is handled by the [`GenotypeMapper`][finchge.grammar.GenotypeMapper].

!!! info "NOTE"

    Although the canonical Grammatical Evolution,
    _mapping_ is generally considered as a one way process of mapping the genotype to phenotype,
    in FinchGE, [`GenotypeMapper`][finchge.grammar.GenotypeMapper]
    has been implemented as a _codec_, providing a bi-directional process of encoding and decoding.

    - The reconstructed genome is **not guaranteed to be identical to the original
    genome**
    - but it is **guaranteed to decode to the same tree and phenotype.**

The GenotypeMapper is responsible for **two symmetric operations**:

1. **Decoding**:
   `genotype -> derivation tree -> phenotype`
   Implemented by the `map()` method.

2. **Encoding**:
   `derivation tree -> genotype`
   Implemented by the `reverse_map()` method.

Together, these form a *bidirectional codec* for Grammatical Evolution.


#### Decoder: `map(...)`

##### Purpose

The **decoder** consumes a list of integers (codons) and applies grammar rules
to construct a derivation tree. Once the tree is complete, its terminal yield
is concatenated to produce the phenotype.

##### Key properties

- Stack-based depth-first expansion
- Rightmost-first expansion order (LIFO)
- Modulo-based production selection
- Codon wrapping up to `max_wraps`
- Explicit recursion depth control
- Detects and flags invalid individuals

##### API

```python
result = mapper.map(genotype)
```

Where:

- `genotype` is a `list[int]`
- `result` is a [`MappingResult`][finchge.grammar.mapper.MappingResult] containing: (`phenotype`, `tree`, `used_genome`, `used_codon_count`, `invalid`)

Example:

```python
from finchge.grammar import Grammar
from finchge.grammar.mapper import GenotypeMapper

grammar = Grammar("""
<string> ::= <letter> | <letter> <string>
<letter> ::= _ | [a-z]
""")

mapper = GenotypeMapper(grammar=grammar)

genome = [23, 5, 99, 17]

result = mapper.map(genome)

print(result.phenotype)
print(result.invalid)
```



#### Encoder: `reverse_map(...)`

##### Purpose

The **encoder** performs the inverse operation:
given a derivation tree, it reconstructs a genome that will reproduce
the same tree when decoded with the same mapper configuration.

This is essential for:

- Tree-based initialisation methods (RHH, Pi-Grow, PTC2 etc.)
- Subtree crossover and mutation
- Hybrid genome-tree evolutionary workflows
- Checkpointing and reproducibility



##### API

```python
    genome = mapper.reverse_map(tree)
```

Optional arguments allow padding or truncation to a fixed genome length.

Example:

```python
from finchge.grammar import Grammar
from finchge.grammar.mapper import GenotypeMapper

grammar = Grammar("""
<string> ::= <letter> | <letter> <string>
<letter> ::= _ | [a-z]
""")

mapper = GenotypeMapper(grammar=grammar)

# Decode
result1 = mapper.map([23, 5, 99, 17])

# Encode
genome2 = mapper.reverse_map(
    result1.tree,
    codon_size=127,
    pad_to_length=10,
)

# Decode again
result2 = mapper.map(genome2)

assert result1.phenotype == result2.phenotype

```

This round-trip check ensures that the mapper does not lose information when converting between genomes and trees.

---

## Fitness Evaluation

In finchGE fitness evaluation is handled by [```FitnessEvaluator```][finchge.fitness.FitnessEvaluator] class which allows
flexible and convenient fitness evaluation especially for use cases where data and models are involved.
The `FitnessEvaluator` is responsible for managing all aspects of fitness function evaluation.
It coordinates key supporting components including the fitness functions themselves, a cache manager,
parallel executors, and benchmark runners to handle the evaluation process efficiently.
By decoupling fitness evaluation from the main evolutionary loop, FinchGE allows different evaluation strategies
to be applied across diverse problem domains, all while exposing a consistent interface to the rest of the evolutionary machinery.

A fitness evaluator can be created as following:


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

Ramped Half-and-Half Initialization in FinchGE is implemented based on Sensible Initialziation
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

Initializers can also be declared in config file with `init_type` key under the section `ge` as following:


```yaml

ge:
  init_type: random_genome
  genome_length: 100
  codon_size: 127
```

All the intialser support initialisation using config files through `from_config()` method.
For example  [`RandomGenomeInitialiser`][finchge.initialisation.RandomGenomeInitialiser] can be used as following.

```python
from finchge.initialisation import  RandomGenomeInitialiser
from finchge.config import  FinchConfig

config = FinchConfig.from_yaml("config.yaml")
initialiser = RandomGenomeInitialiser.from_config(config=config)

```

Note: To intitialise using config, all the parameters required by the respective initializers must be provided in the config files.

## Genetic Operators

Evolutionary behavior in finchGE is defined through the composition of **genetic operators**.

These include:

- selection
- crossover
- mutation
- replacement

Each operator is implemented as an independent component with a well-defined interface.
No operator is implicitly embedded in the algorithm.

This design allows developers to substitute operators without modifying core logic and prototype new evolutionary strategies.

---

## Algorithms

finchGE distinguishes between **evolutionary mechanics** and **execution orchestration**.

At the lower level:

- a generic Genetic Algorithm implementations perform population-level evolution,
  typically one generation at a time

At the higher level:

  - a Grammatical Evolution controller coordinates full evolutionary runs, including:
  - initialisation
  - termination criteria
  - logging and result tracking

This layered architecture allows the same core components to be reused in fully automated experimental pipelines
as well as in interactive and exploratory evolutionary runs.

## Configuration

FinchGE can be configured using **plain Python dictionaries** or **external configuration files** (INI or YAML).
FinchGE uses a **structured configuration system** to describe experiments in a clear, reproducible, and extensible way.

!!! info "Principle"
    FinchGE follows these principles for project configuration:

    - FinchGE configuration is explicit, modular, and optional
      - Users can choose between:
        - programmatic configuration (Python dicts)
        - file-based configuration (INI / YAML)
      - Configuration can be **copied and modified**, enabling adaptive or meta-evolutionary experiments.

### Configuration Overview


The configuration is centered around **three top-level sections**:

- `experiment` - run-level concerns (how long, how big, logging, caching, randomness)
- `ge` - all Grammatical Evolution-specific parameters (grammar, initialisation, operators)
- `parallel` - configuration related to parallelised fitness evaluation.

This design avoids artificial separation between tightly coupled GE components (grammar, initialisation, operators) and ensures consistent parameter sharing.

The **`experiment`** section controls how an experiment is executed.

Typical parameters include:

- `random_seed` - global seed for reproducibility
- `population_size` - number of individuals
- `num_generations` - evolution length
- `verbose` - logging verbosity
- `cache_type`, `cache_size` - optional fitness caching

The **`ge`** section defines the Grammatical Evolution system itself.

Typical parameters include:

- Grammar and mapping:
    - `grammar_file`
    - `codon_size`
    - `max_wraps`
    - `max_recursion_depth`
    - `genome_length`

- Initialisation:
    - `init_type` (e.g. `random_genome`, `pi_grow`, `rhh`)
    - `min_depth`, `max_depth` (tree-based initialisers)
- Operators:
    - `mutation_probability`
    - `crossover_probability`
    - `elite_size`

All GE components (grammar, mapper, initialisers, operators) read from this same `ge` section, ensuring consistency.

The **`parallel`** section defines the configuration related to parallelization of the fitness evaluation.

The parameters include:

  - `parallel_enabled` - flag to enable/disable parallel evaluation
  - `executor_type` - `thread` or `process`
  - `max_workers`
  - `batch_size`

---


??? note "Example Configuration in YAML and INI"

    The algorithm supports two distinct operational modes based on the presence of structural constraints.

    === "YAML"

        ```yaml
            experiment:
              random_seed: 42
              num_generations: 200
              verbose: true
              cache_type: lru
              cache_size: 128

            ge:
              population_size: 100
              grammar_file: grammar.bnf
              codon_size: 127
              max_wraps: 6
              max_recursion_depth: 20
              genome_length: 100

              init_type: pi_grow
              max_depth: 6

              mutation_probability: 0.01
              crossover_probability: 0.5
              elite_size: 1

            parallel:
              parallel_enabled: true
              executor_type: process
              max_workers: 4
              batch_size: 25

        ```

    === "INI"


        ```ini
            [experiment]
            random_seed = 42
            num_generations = 200
            verbose = true
            cache_type = "lru"
            cache_size = 128

            [ge]
            population_size = 100
            grammar_file = grammar.bnf
            codon_size = 127
            max_wraps = 6
            max_recursion_depth = 20
            genome_length = 100
            init_type = pi_grow
            max_depth = 6
            mutation_probability = 0.01
            crossover_probability = 0.5
            elite_size = 1

            [parallel]
            parallel_enabled = true
            executor_type = process
            max_workers = 4
            batch_size = 25

        ```

#### Required vs Optional Parameters

FinchGE distinguishes between two kinds of configuration parameters to ensure both safety and flexibility.

###### Required (no defaults)

**Structural parameters** that define the problem space and evolutionary system:

- `population_size`
- `grammar_file`
- `codon_size`
- `genome_length`

Missing required parameters raise **immediate errors** to prevent silent misconfiguration and undefined behavior.

###### Optional (safe defaults)

**Behavioral or convenience parameters** that influence execution but have sensible defaults:

- `cache_size`
- `verbose`
- `elite_size`

Defaults are documented and consistently applied across the library.

### Loading Configuration with `FinchConfig`

FinchGE provides an optional helper class, [`FinchConfig`][finchge.config.FinchConfig], for loading and managing configuration files.


```python
from finchge.utils.ge_config import FinchConfig

cfg = FinchConfig.from_file("config.ini")
```

The file format is detected automatically based on extension (`.ini`, `.yaml`, `.yml`).

Configuration sections are accessed as dictionaries:

```python
cfg.ge["codon_size"]
cfg.ge["init_type"]
cfg.ge["mutation_probability"]
```

Note: [`FinchConfig`][finchge.config.FinchConfig] is a convenience layer.
FinchGE classes also accept configs in the form of plain Python dictionaries.

### Using dictionary as config
For notebooks or scripts, we can bypass [`FinchConfig`][finchge.config.FinchConfig] entirely
by creating configs in the form of dictionaries, as shown below.

```python
config = {
  "experiment": {
    "random_seed": 42,
    "num_generations": 10,
    "verbose": True,
    "cache_type": "lru",
    "cache_size": 128,
  },
  "ge": {
    "population_size": 100,
    "grammar_file": "grammar.bnf",
    "codon_size": 127,
    "max_wraps": 6,
    "max_recursion_depth": 10,
    "genome_length": 100,
    "init_type": "random_genome",
    "mutation_probability": 0.01,
    "crossover_probability": 0.5,
    "elite_size": 3,
  },
  "parallel": {
    "parallel_enabled": True,
    "executor_type": "process",
    "max_workers": 4,
    "batch_size": 25,
  },
}

```
Constructors and class factories consume these dictionaries directly.


---

### Configuration Copying and Adaptation

`FinchConfig` supports copying with updates, which is useful for:

- parameter control
- self-adaptive strategies
- experiment branching


```python
new_cfg = cfg.copy(update={
    "ge": {
        "mutation_probability": 0.1
    }
})
```
The original configuration remains unchanged.

### Inspecting Configuration
For logging and debugging, configurations can be rendered in multiple formats:

```python
print(cfg.to_json())
print(cfg.to_table())
cfg.display()   # Jupyter notebook-friendly display
```
This makes experiments easy to reproduce and compare.



## Experiment Utilities

finchGE provides utilities for logging, result tracking and visualization of results.
Apart from the interactive notebook-based workflow finchGE provides a recommended experiment structure
designed for efficient experimentation for various use cases.
Read more about [finchGE Experiments](getting_started.md#project-based-workflow)


### Logging and Experiment Tracking

FinchGE provides a **two-level logging system** that clearly separates
runtime diagnostics from experiment artifacts.

This design avoids coupling logging, results, and checkpointing, while
supporting both lightweight monitoring and full experiment tracking.

---

#### Runtime logging (Python logging)

FinchGE uses Python’s built-in `logging` module for **runtime messages** such as:

- progress updates
- warnings and errors
- timing information
- debug output

Runtime logging is configured automatically when an experiment starts and
is compatible with progress bars (`tqdm`).

Typical messages include:

- experiment start and end
- generation progress
- total execution time
- configuration warnings

Runtime logs are written to a `.log` file and optionally displayed in the console.

---

#### Experiment loggers

Experiment loggers record **structured results** produced during evolution.
They are independent of Python logging and operate through lifecycle callbacks.

All experiment loggers implement the [`BaseLogger`][finchge.utils.logger.BaseLogger] interface.

---

#### Available experiment loggers

#### FileLogger (lightweight)

[`FileLogger`][finchge.utils.logger.FileLogger] records **compact, append-only summaries** suitable for monitoring
and post-hoc analysis.
Use `[`FileLogger`][finchge.utils.logger.FileLogger] when storage should be minimal,  only high-level metrics are needed, or long runs are monitored in real time.

For each generation it logs:

- best fitness (single-objective), or
- summary statistics of the Pareto front (multi-objective)

At the end of a run it writes a final summary file.

Example:

```python
from finchge.utils.logger import FileLogger

logger = FileLogger()

```


#### ExperimentLogger (full experiment tracking)

[`ExperimentLogger`][finchge.utils.logger.ExperimentLogger] records complete experiment artifacts, enabling full
reproducibility and detailed analysis. Use [`ExperimentLogger`][finchge.utils.logger.ExperimentLogger]  when
 detailed inspection is required,
 experiments are intended for publication, or
 derivation trees or full fronts must be preserved.

Depending on the optimization mode, it may store:

- phenotypes
- genotypes
- derivation trees
- per-generation CSV files
- full Pareto fronts (multi-objective)

Example:

```python
from finchge.utils.logger import ExperimentLogger

logger = ExperimentLogger(exclude=["trees"])
```


#### Using experiment loggers

When using [`GrammaticalEvolution`][finchge.core.GrammaticalEvolution] for running project, loggers are managed
automatically.

```python
ge = GrammaticalEvolution(
    grammar=grammar,
    fitness_evaluator=fitness_evaluator,
    config=config,
    logger=ExperimentLogger(),
)

result = ge.run()

```

However, for more advanced usage or custom logging Experiment Loggers provide callbacks such as `on_run_start()` ,
`on_generation_end()` and `on_run_end()`.


#### Single-objective vs multi-objective logging

- **Single-objective** runs log the best individual per generation.
- **Multi-objective** runs log Pareto front summaries ([`FileLogger`][finchge.utils.logger.FileLogger]) or
full fronts ([`ExperimentLogger`][finchge.utils.logger.ExperimentLogger]).

The logging behavior adapts automatically based on the algorithm used.


!!! info "Note"

    - Runtime logging and experiment logging are intentionally separate.
    - Experiment loggers never configure Python logging.
    - Checkpointing operates independently of logging.
    - Logging does not affect determinism or reproducibility.

      Use [`FileLogger`][finchge.utils.logger.FileLogger] with minimal logging for quicker development and [`ExperimentLogger`][finchge.utils.logger.ExperimentLogger] for
      final experiments for detailed results analysis.




### Checkpointing and Resuming experiments

FinchGE supports **deterministic checkpointing**, allowing long-running evolutionary
experiments to be paused and resumed without altering the evolutionary trajectory.

This is especially useful for:

- long experiments
- cluster or cloud environments
- interactive experimentation

#### Enabling checkpointing

To enable checkpointing, create a [`CheckpointManager`][finchge.utils.checkpoint.CheckpointManager] and pass it to
[`GrammaticalEvolution`][finchge.core.GrammaticalEvolution].

```python
from finchge.utils.checkpoint import FileCheckpointManager
from finchge.algorithm import GrammaticalEvolution

checkpoint_manager = FileCheckpointManager(
    directory="runs/exp1/checkpoints",
    every=10,  # save every 10 generations
    keep_last=5,  # keep only the latest 5 checkpoints
)

ge = GrammaticalEvolution(
    grammar=grammar,
    fitness_evaluator=fitness,
    config=config,
    checkpoint_manager=checkpoint_manager,
)

result = ge.run()
```


Checkpoints will be written as:



```yaml
runs/exp1/checkpoints/
├── checkpoint_gen_10.pkl
├── checkpoint_gen_20.pkl
├── checkpoint_gen_30.pkl
```

#### Resuming from a checkpoint

If a checkpoint exists in the specified directory, FinchGE will automatically
resume from the latest checkpoint.

No additional flags are required.

```python

# Re-create the same experiment
ge = GrammaticalEvolution(
    grammar=grammar,
    fitness_evaluator=fitness,
    config=config,
    checkpoint_manager=checkpoint_manager,
)

# Automatically resumes from the latest checkpoint
result = ge.run()


```

The resumed run is bit-for-bit deterministic with respect to a continuous run.


#### What is saved in a checkpoint?

Each checkpoint stores:

- the current generation index
- the full population
- algorithm state
- Python and NumPy RNG states
- a hash of the experiment configuration

This guarantees exact reproducibility when resuming.


!!! warning "IMPORTANT: Configuration Safety "

    If the experiment configuration changes between runs, FinchGE will refuse to
    resume from an incompatible checkpoint and raise an error.


    CAUTION: Although this configuration safety features prevents accidental misuse of checkpoints,
    **it may not be helpful in use cases where configuration itself is allowed to evolve.**
    Alternative ways of handling config is such situation is strongly recommended.


### Result Aggregation and Visualization

FinchGE includes [`ExperimentLogger`][finchge.utils.logger.ExperimentLogger] to log the results which can be aggregated using
 [`ResultHelper`][finchge.utils.results.ResultHelper] and [`StatsHelper`][finchge.utils.results.StatsHelper] utilities.

NOTE *At this point Result Aggregating and Visualization utils
provide limited functionality to work with GrammaticalEvolution,
for advanced result analysis, custom analysis and visualation is used based on the requirement.*

---

### Symbolic Regression

#### Symbolic Expression

[`SymbolicExpression`][finchge.symbolic.SymbolicExpression] class provides a lightweight interface for parsing,
validating, and evaluating symbolic mathematical expressions
used in symbolic regression workflows.

The class converts string-based mathematical expressions into *SymPy* symbolic objects and compiles them into efficient
*NumPy-based* numerical evaluators. It supports multi-variable expressions, custom function sets, and automatic detection of required input features.

##### Key Features

- Parse mathematical expressions from strings (e.g., `sin(x0) + x1**2`)
- Automatic variable detection (`x0`, `x1`, `x2`, ... OR in array slice format `x[: 0]`, `x[: 1]`, `x[: 2]`, ...)
- Support for user-defined symbolic functions
- Structural complexity metrics (node count and depth)
- Optional symbolic simplification
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



## Benchmarks

FinchGE provides a collection of benchmark problems for genetic programming research,
covering regression, logic, and symbolic regression tasks.



### Benchmark Components

Following are the key components used in FinchGE's benchmark suite.
Understanding these pieces will make it easy to run experiments and extend the library.

##### Benchmark Classes
Every problem in FinchGE is represented by a [`Benchmark`][finchge.benchmarks.Benchmark] class.
These classes know everything about a specific problem: its mathematical definition, data ranges,
and how to evaluate solutions. Benchmarks are organized by categories such as regression, control, logic

```python
from finchge.benchmarks.regression import NguyenBenchmark, KeijzerBenchmark
from finchge.benchmarks.control import SantaFeTrailBenchmark
from finchge.benchmarks.logic import MultiplexerBenchmark
```

Each benchmark provides:

- `metadata` - Information about the problem
- `grammar()` - The BNF grammar for valid solutions
- `create_runner()` - Creates a runner that can evaluate phenotypes eg. evaluating expressions, or training models etc.

##### Runners

A `PhenotypeRunner` is responsible for taking a program string and turning it into measurable results.
Different problem types use different runners:

- `SymbolicRegressionRunner` - Evaluates mathematical expressions on data
- `LogicRunner` - Tests logic circuits against truth tables

Runners always return an evaluation context that fitness functions can work with.

##### Environments

For control problems, Environment classes simulate the world where agents operate.
Each environment maintains state and provides observations:

- [```SantaFeEnvironment```][finchge.benchmarks.control.SantaFeEnvironment] - The ant trail grid with food
- [```MazeEnvironment```][finchge.benchmarks.control.MazeEnvironment]  - Navigation with walls and goals
- [```CartPoleEnvironment```][finchge.benchmarks.control.CartPoleEnvironment] - Physics simulation for balancing

Environments are created fresh for each evaluation using factory functions.

##### Fitness Functions

The [```GEFitnessFunction```][finchge.fitness.GEFitnessFunction] classes calculate how good a solution is:

- [```MAEFitness```][finchge.fitness.MAEFitness] - For regression (lower is better)
- [```RMSEFitness```][finchge.fitness.RMSEFitness] - For regression (lower is better)
- [```RewardFitness```][finchge.fitness.RewardFitness] - For control problems (higher is better)
- [```AccuracyFitness```][finchge.fitness.AccuracyFitness] - For logic problems (higher is better)

Each fitness function receives a context dictionary mainly with `y_pred` and `y_true`. However, the context may include
other problem-specific information required by the fitness function.

##### Random State Management

Reproducibility is handled through the [```RandomStateMixin```][finchge.utils.random_mixin.RandomStateMixin]. Any class that needs randomness accepts a `random_state` parameter:

```python
bench = NguyenBenchmark(version=6, random_state=42)
```

The mixin provides `self.rng` (Python random) and `self.np_rng` (NumPy RandomState) for all the random needs.
Any custom cumponent requiring randomnes should use these RNGs instead of global `random` or `np.random` to ensure
deterministic experiments.



A typical experiment flows like this:

```python
benchmark = SantaFeTrailBenchmark(random_state=42)
grammar = Grammar.from_string(benchmark.grammar())
runner = benchmark.create_runner()
fitness = RewardFitness(maximize=True)

ge = GrammaticalEvolution(runner, fitness, grammar)
result = ge.run()

```


??? Pro Tip  "Using Custom Benchmarks"

    To add a new problem, implement:

        1. An `Environment` class (for control problems)
        2. A `Runner` that knows how to evaluate phenotypes
        3. A `Benchmark` that brings it all together

    The base classes handle the rest, including random state management and parallelization support.


### Regression Benchmarks

#### Nguyen Benchmark Suite
10 symbolic regression problems of increasing complexity, from simple polynomials to trigonometric functions.

Each benchmark provides:

- `load_data()`: Returns (`X_train`, `y_train`, `X_test`, `y_test`)
- `metadata`: Information about the benchmark
- `grammar()`: Appropriate grammar for the problem type

??? info "Nguyen Benchmark Suite"

    | Benchmark Name | Target Function | Input Range |
    | :--- | :--- | :--- |
    | `"Nguyen-1"` | $x^3 + x^2 + x$ | $x \in [-1, 1]$ |
    | `"Nguyen-2"` | $x^4 + x^3 + x^2 + x$ | $x \in [-1, 1]$ |
    | `"Nguyen-3"` | $x^5 + x^4 + x^3 + x^2 + x$ | $x \in [-1, 1]$ |
    | `"Nguyen-4"` | $x^6 + x^5 + x^4 + x^3 + x^2 + x$ | $x \in [-1, 1]$ |
    | `"Nguyen-5"` | $\sin(x^2) \cos(x) - 1$ | $x \in [-1, 1]$ |
    | `"Nguyen-6"` | $\sin(x) + \sin(x + x^2)$ | $x \in [-1, 1]$ |
    | `"Nguyen-7"` | $\log(x + 1) + \log(x^2 + 1)$ | $x \in [0, 2]$ |
    | `"Nguyen-8"` | $\sqrt{x}$ | $x \in [0, 4]$ |
    | `"Nguyen-9"` | $\sin(x) + \sin(y^2)$ | $x, y \in [0, 1]$ |
    | `"Nguyen-10"`| $2\sin(x)\cos(y)$ | $x, y \in [0, 1]$ |

    Training data is generated using uniform random sampling (20 points), while testing uses a fixed grid (1000 points).

 **Reference:** Uy, N.Q., Hoai, N.X., O’Neill, M., McKay, R.I. and Galván-López, E., (2011).
*Semantically-based crossover in genetic programming: application to real-valued symbolic regression. Genetic Programming and Evolvable Machines*


```python
from finchge.benchmarks.regression import NguyenBenchmark

benchmark = NguyenBenchmark(version=1, random_state=42) # version 1 for Nguyen-1
X_train, y_train, X_test, y_test = benchmark.load_data()
print(f"Training samples: {len(X_train)}")  # 20 points
print(f"Test samples: {len(X_test)}")  # 1000 points

```
#### Keijzer Benchmark Suite

15 functions including rational functions, harmonic series, and multi-dimensional problems.

??? info "Keizjer Benchmark Suite"

    The Keijzer suite includes 15 functions of varying complexity. All implementations follow the exact specifications from Keijzer (2003).

    | Benchmark ID | Target Function | Input Range | Points (Train/Test) |
    | :--- | :--- | :--- | :--- |
    | `Keijzer-1` | $0.3x \sin(2\pi x)$ | $x \in [-1, 1]$ | 21 / 201 |
    | `Keijzer-2` | $0.3x \sin(2\pi x)$ | $x \in [-2, 2]$ | 41 / 401 |
    | `Keijzer-3` | $0.3x \sin(2\pi x)$ | $x \in [-3, 3]$ | 61 / 601 |
    | `Keijzer-4` | $x^3 e^{-x} \cos(x) \sin(x) (\sin^2(x) \cos(x) - 1)$ | $x \in [0, 10]$ | 201 / 201 |
    | `Keijzer-5` | $30 \frac{(x-1)(x-3)}{(x-2)^2}$ | $x \in [0.05, 2]$ | 40 / 40 |
    | `Keijzer-6` | $x + \sin(x)$ | $x \in [-1, 1]$ | 21 / 201 |
    | `Keijzer-7` | $\log(x)$ | $x \in [1, 100]$ | 100 / 991 |
    | `Keijzer-8` | $\sqrt{x}$ | $x \in [0, 100]$ | 101 / 1001 |
    | `Keijzer-9` | $\text{asinh}(x)$ | $x \in [0, 100]$ | 101 / 1001 |
    | `Keijzer-10`| $x^y$ | $x, y \in [0, 1]$ | 100 / 1000 |
    | `Keijzer-11`| $xy + \sin((x-1)(y-1))$ | $x, y \in [-3, 3]$ | 100 / 1000 |
    | `Keijzer-12`| $x^4 - x^3 + \frac{y^2}{2} - y$ | $x, y \in [-3, 3]$ | 100 / 1000 |
    | `Keijzer-13`| $6 \sin(x) \cos(y)$ | $x, y \in [-3, 3]$ | 100 / 1000 |
    | `Keijzer-14`| $8 / (2 + x^2 + y^2)$ | $x, y \in [-3, 3]$ | 100 / 1000 |
    | `Keijzer-15`| $x^3/5 + y^3/2 - y - x$ | $x, y \in [-3, 3]$ | 100 / 1000 |


    * **Keijzer 1-9:** Use a fixed **step-based** sampling for training and a finer step for testing.
    * **Keijzer 10-15:** Use **uniform random** sampling (100 pts) for training and a **regular grid** (1000 pts) for testing.


**Reference:** Keijzer, M. (2003). *Improving Symbolic Regression with Interval Arithmetic and Linear Scaling.*

```python
from finchge.benchmarks import KeijzerBenchmark

benchmark = KeijzerBenchmark(version = 5, random_state=42) # Keijzer-5
X_train, y_train, X_test, y_test = benchmark.load_data()

```

#### Koza Quartic

Classic quartic polynomial benchmark from Koza's 1992 GP book.

The Koza Quartic is a classic symbolic regression problem used to evaluate the symbolic modeling capabilities of GP systems.

??? info "Koza Quartic Benchmark"

    | Benchmark ID | Target Function | Input Range | Points (Train/Test) |
    | :--- | :--- | :--- | :--- |
    | `koza-quartic` | $x^4 + x^3 + x^2 + x$ | $x \in [-1, 1]$ | 20 / 1000 |

    * **Training Data:** 20 points sampled using uniform random distribution.
    * **Test Data:** 1000 points sampled on a regular grid to measure generalization accuracy.
    * **Fitness Metric:** Usually measured via Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

```python
from finchge.benchmarks.regression import KozaQuarticBenchmark
benchmark = KozaQuarticBenchmark()
```


#### Vladislavleva Benchmark Suite

The Vladislavleva benchmarks are designed to evaluate extrapolation and interpolation capabilities in multi-dimensional spaces.

??? info "Vladislavleva Benchmark Suite"

    | Benchmark ID | Target Function | Variables | Points (Train/Test) |
    | :--- | :--- | :---: | :--- |
    | `vla-1` | $\frac{e^{-(x-1)^2}}{1.2 + (y-2.5)^2}$ | 2 | 100 / 1024 |
    | `vla-2` | $e^{-x} x^3 \cos(x) \sin(x) (\sin^2(x) \cos(x) - 1)$ | 1 | 100 / 1000 |
    | `vla-3` | $e^{-x} x^3 \cos(x) \sin(x) (\sin^2(x) \cos(x) - 1)(y-5)$ | 2 | 300 / 1024 |
    | `vla-4` | $10 / (5 + \sum_{i=1}^{5} (x_i - 3)^2)$ | 5 | 1024 / 5000 |
    | `vla-5` | $30(x_1-1)(x_3-1) / (x_1^2(x_1-10)(x_2-20))$ | 3 | 300 / 1024 |
    | `vla-6` | $6 \sin(x_1) \cos(x_2)$ | 2 | 30 / 1000 |
    | `vla-7` | $(x_1-3)^4 + (x_2-3)^3 - (x_2-3)$ | 2 | 300 / 1024 |
    | `vla-8` | $((x_1-3)^4 + (x_2-3)^3 - (x_2-3)) / ((x_3-2)^2 + 1)$ | 3 | 300 / 1024 |


    * **Input Ranges:** These vary by problem but typically cover $x_i \in [0, 5]$ or $[0.05, 10]$ depending on the specific Urbina/Vladislavleva paper being referenced.
    * **Features:** These problems are specifically chosen to test "bloat" control and the ability to handle rational functions (fractions) and exponentials.

**Reference:** Vladislavleva, et al. (2008) Order of nonlinearity as a complexity measure for models generated
by symbolic regression via pareto genetic programming. IEEE Transactions on Evolutionary Computation, 13(2), pp.333-349.

### Logic Benchmarks

#### Boolean Multiplexer Benchmark

The Multiplexer problem requires the system to evolve a boolean function that uses $k$ address bits
to select one of $2^k$ data bits for the output.

- 6-bit Multiplexer: 2 address bits + 4 data bits (64 cases)
- 11-bit Multiplexer: 3 address bits + 8 data bits (2048 cases)

??? info "Multiplexer  Benchmark"

    | Benchmark ID | Address Bits | Data Bits | Total Input Bits | Truth Table Rows |
    | :--- | :---: | :---: | :---: | :--- |
    | `mux-6` | 2 | 4 | 6 | 64 |
    | `mux-11` | 3 | 8 | 11 | 2,048 |

    * **Function Set:** `AND`, `OR`, `NOT`, `IF` (or `Conditional`).
    * **Terminal Set:** The input bits $\{a_0...a_n, d_0...d_n\}$.
    * **Fitness:** Usually defined as the number of correct outputs across all possible $2^n$ bit combinations.

```python
from finchge.benchmarks.logic import MultiplexerBenchmark

benchmark = MultiplexerBenchmark(version=11) # 11-bit multiplexer
X, y, _, _ = benchmark.load_data()  # Complete truth table
```

### Control Benchmarks

#### Cartpole Benchmark

A classic control problem where a pole is attached to a cart moving along a frictionless track.
The goal is to apply forces to keep the pole upright as long as possible.
The environment is stochastic (random initial state) and rewards +1 per step the pole remains balanced.

```python
from finchge.benchmarks.control import CartPoleBenchmark

benchmark = CartPoleBenchmark(random_state=None, max_steps=500, n_episodes=1)
```

#### Maze Benchmark

A navigation benchmark where an agent must find its way from a start position to a goal in a grid-based maze.
Walls block movement, and the agent receives sensor inputs indicating whether walls are present ahead,
to the left, or to the right.

##### Available Mazes:

- `simple`: 5×5 grid (start=2, goal=3)
- `medium`: 8×8 grid (classic Koza maze)
- `hard`: 11×11 grid (more complex layout)

```python
# Simple Maze
from finchge.benchmarks.control import MazeBenchmark

benchmark1 = MazeBenchmark(version = "medium")
```




#### SantaFe Trail (Artificial Ant) Benchmark

The Santa Fe Trail is a classic genetic programming benchmark where an ant must navigate
a 32×32 toroidal grid to collect 89 food pellets arranged along a winding trail.
The ant starts at position (20, 0) facing east and has a maximum of 600 steps per episode.

SantaFeEnvironment environment provides the control simulation of the Santa Fe Trail Problem with,


```python
from finchge.benchmarks.control import SantaFeTrailBenchmark

benchmark = SantaFeTrailBenchmark(random_state=42)

```
