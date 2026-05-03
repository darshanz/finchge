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

    YAML example:

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

    INI example:

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
