# Getting Started

This page provides a short overview of how to start working with **finchGE**.

---

## Getting Started Is Simple

finchGE is designed to be flexible, but getting started does not require much setup.
You can begin working with finchGE in **two primary ways**, depending on your goals:

- **Interactive usage** using a Jupyter notebook
- **Project-based usage** using the recommended finchGE experiment structure

Both approaches use the same underlying components, however experiment-style workflow is designed to have better organized control and tooling for experiments.

To get started, simply create a finchGE project using following commands.
```bash
finchge new my_experiment --notebook
```

This will create a sample project with a starter notebook and a project for project-based workflow.

Use `--template` option for different project templates. Following templates are available.

- `--template basic`  (default template): StringMatch problem
- `--template control`: SantaFE Trail Problem
- `--template logic`  : Multiplexer problem
- `--template symbolic_regression`  : Symbolic Regression problem (Nguyen-6)


---


## Interactive Usage (Jupyter)

For quick experimentation, learning, or exploratory research, you can use finchGE
directly inside a Jupyter notebook. In the sample project, find the notebook (```basic.ipynb```) in ```my_experiment/notebooks``` folder and follow along the notebook cells.

This approach provides maximum control and visibility into the evolutionary process.

See: [Examples - Interactive String Matching Example](examples.md)

---

## Project-Based Workflow

For larger or reproducible experiments, finchGE provides a recommended **project structure**.

A basic finchGE project looks like this:

```yaml
my_experiment/
├── main.py
├── config.yaml
├── grammar.bnf
├── fitness.py
└── logs/
```
Once the project is created using ```finchge new NAME``` command, either open it on your favorite IDE or use ```finchge run``` to run from commandline. This will create experiment outputs in logs directory.

### Key Files

- **`main.py`**
  Entry point for running the experiment.

- **`config.yaml`** (or `config.ini`)
  Defines the experiment configuration and evolutionary parameters such as population size, number of generations,
  mutation probability, crossover probability, and elite size.
  Using a config file makes experiments easy to reproduce and modify without
  changing code. For configuration `.yaml` and  `.ini`  files are supported.
  For more information, check out [Configuration](introduction.md#configuration) section.

- **`grammar.bnf`**
  Defines the grammar used for genotype-phenotype mapping and constrains the search space.

- **`fitness.py`**
  Contains the problem-specific custom fitness function(s).

---

The basic project layout presented here is a minimal example intended to illustrate finchGE usage.
In practice, experiments may adopt more elaborate structures depending on scale, reproducibility requirements.

## Experiment Logs and Outputs

When using the project-based workflow, finchGE automatically creates a
timestamped directory inside the `logs/` folder for each run.

A typical log directory contains:
- evolved **genotypes** and **phenotypes**
- grammar **derivation trees**
- fitness statistics per generation
- plots (e.g. fitness vs. generation)
- CSV summaries and human-readable reports
- runtime logs for debugging and analysis

This logging system is designed to support **reproducibility**, **analysis**, and
**post-processing** of experimental results.

---

## Where to Go Next

- For hands-on usage examples, see the  [Examples](examples.md) section:
  - Interactive notebook-based examples
  - Project-based experiment examples

- For a deeper explanation of finchGE’s design and core components, see the
  **Introduction** section.

That’s all you need to start working with finchGE.
