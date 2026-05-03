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
fully support every eBNF syntax. The syntax followed by finchGE is:

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

3. Character Ranges (ellipsis notation)

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
    with [`BNFGrammarParser`][finchge.grammar.parser.BNFGrammarParser] as follows:
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
in grammar as follows:

```bnf

<expr> ::= <expr> <op> <expr> | <var> | <const>
<op> ::= + | -
<var> ::= x[:, 4]
<const> ::= 1.0 | 2.0 | 3.0

```

Here, `x[:, 4]` represents the fourth column in the data `x`. Similarly, to allow multiple columns as
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
during parsing.

Similarly, for less common use cases, stepped range pattern is also supported for array slices. For example `
x[:, 0..10 step 2]` is expanded to `x[:, 0] | x[:, 2] | x[:, 4] | x[:, 6] | x[:, 8] | x[:, 10]`.

### Genotype Mapper

Grammatical Evolution is primarily driven by _genotype-phenotype mapping_.
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
