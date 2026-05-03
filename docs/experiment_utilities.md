## Experiment Utilities

finchGE provides utilities for logging, result tracking, and visualization.
Apart from the interactive notebook-based workflow, finchGE provides a recommended experiment structure
designed for efficient experimentation across various use cases.
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



#### ExperimentLogger

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

#### Using experiment logger

When using [`GrammaticalEvolution`][finchge.core.GrammaticalEvolution] to run a project, the logger is managed
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

However, for more advanced usage or custom logging, experiment loggers provide callbacks such as `on_run_start()`,
`on_generation_end()` and `on_run_end()`.

The logging behavior records relevant information automatically based on whether a single-objective or multi-objective algorithm is used.


!!! info "Note"

    - Runtime logging and experiment logging are intentionally separate.
    - Experiment logger never configures Python logging.
    - Checkpointing operates independently of logging.
    - Logging does not affect determinism or reproducibility.



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


    CAUTION: Although this configuration safety feature prevents accidental misuse of checkpoints,
    **it may not be helpful in use cases where configuration itself is allowed to evolve.**
    Alternative ways of handling config in such situations are strongly recommended.


### Result Aggregation and Visualization

FinchGE includes [`ExperimentLogger`][finchge.utils.logger.ExperimentLogger] to log results that can be aggregated using
 [`ResultHelper`][finchge.utils.results.ResultHelper] and [`StatsHelper`][finchge.utils.results.StatsHelper] utilities.

NOTE *At this point, result aggregation and visualization utilities
provide limited functionality to work with GrammaticalEvolution,
for advanced result analysis, custom analysis and visualization are used based on the requirement.*

---
