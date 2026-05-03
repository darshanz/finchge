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
