# Changelog

All notable changes to this project will be documented in this file.

For changelog we follow  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format and for versioning use  [Semantic Versioning](https://semver.org/)

## [1.0.1-beta.15](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.15) - 2026-08-17

### Added
- Added `min_tail_ratio` parameter on `GenotypeMapper.reverse_map`, for genome tail configuration
- Documentation updated adding citations for the Evolution Strategies, Memetic GA, Island GA, and CLONALG algorithms.
- Test coverage for the GA algorithm, NSGA-II/NSGA-III utilities, cache manager, tree derivation, protected math, and experiment logger

### Changed
- `SubtreeMutation` no longer accepts a `mutation_probability` parameter.
- `NSGA2Replacement` (renamed from `NSGA2ElitistReplacement`) no longer applies an
  explicit elite-preservation step as NSGA-II's own non-dominated sorting and crowding
  distance already guarantee elitist behavior.

### Fixed
- `GenotypeMapper`'s genome padding no longer truncates genomes that need padding.


## [1.0.1-beta.14](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.14) - 2026-08-08

### Added
- Evolution strategies: OnePlusOneES, MuPlusLambdaES, MuCommaLambdaES
- MemeticGA algorithm
- CLONALG (clonal selection) algorithm
- Top-level package imports `from finchge import GrammaticalEvolution, FitnessEvaluator, Grammar, GenotypeMapper, FinchConfig, Keys`
- Test coverage for the new algorithms, NSGA2 utilities, Individual class, and fitness functions

### Changed
- `GEResult.best_individual` replaced with `best_in_generation` and `all_time_best`, tracked separately across the whole run instead of only the final generation (fixes #17)
- NSGA-II now pushes individuals to the end of the ranking instead of raising an error
- Removed unnecessary quote-stripping in the mapper, remained from initial versions of the parser.

### Fixed
- SwapMutation no longer silently skips the last position when an odd count is selected
- NSGA-II crowding distance now recalculated for all individuals; removed an incorrect `if None` condition that skipped some
- SubtreeMutation and SubtreeCrossover now respect `max_tree_depth`. previously mutation had separate depth constraint
- GenotypeMapper (genome-based initializers) non-terminals at the depth cap are now rejected before expansion instead of one step late, which caused tree size larger than `max_tree_depth`



## [1.0.1-beta.13](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.13) - 2026-07-05

### Added
- Algorithm support for IslandGA and SteadyStateGA
- Improved documentation coverage and unit tests

### Changed
- Refactored base Algorithm class to have different base classes for single and multi-objective algorithms
- Removed SteadyStateReplacement, which is now an algorithm SteadyStateGA
- Refactored algorithms to move RNG injection responsibility to base algorithms, writing custom algorithm is now easier
- Improved parallel experiments, manually shutdown process pool executor to avoid accumulation of orphaned workers.

### Fixed
- Generational replacement corrected elite handling error
- Mapper to avoid inconsistency with default params
- RNG injection error in MultipleMutation, ensuring deterministic experiment with multiple mutation.
- Gaussian mutation producing codons above codon_size
- Crossover issue, first element not being swapped.
- Removed redundant evaluation call in evolution loop


## [1.0.1-beta.12](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.12) - 2026-05-03

### Added
- Test cases for experiment logging
- Test cases for CLI
- Added docstrings to some multiple classes for API docs
- CLI tools for listing available templates and finchge doctor command for overall health check

### Changed
- Changed documentation by breaking down introduction page into multiple sub pages
- Updated CLI tools for improved validation of config and grammar files
- Refactored to separate the functions for ptc2 and ptc2d
- Refactoring in mapper for variable name mismatches

### Fixed
- Fixed experiment logging issue with multi-objective ge
- Fixed subtree crossover returning incomplete trees
- Default grammar changed to make same for Keijzer, Nguyen Vladislavleva benchmarks
- Fixed docstrings supporting strict build mode of mkdocs
- Fixed permutation counts (storing exact-depth)
- Fixed crossovers resulting in one extra individual when population size is odd number


## [1.0.1-beta.11](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.11) - 2026-04-13

### Fixed
- Fixed tail appending during reverse-mapping
- Fixed subtree crossover returning incomplete trees
- Default grammar changed to make same for Keijzer, Nguyen Vladislavleva benchmarks


## [1.0.1-beta.10](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.10) - 2026-04-13

### Fixed
- Fixed CLI-based project creation issues

## [1.0.1-beta.9](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.9) - 2026-04-13

### Added
- Test cases to verify population related properties of tree based initialisers
- Documentation added in API docs, comments and docstrings.

### Changed
- Refactored code for checking incompatible operators
- updated docs related to tree based initialization and operators
- updated experiment logger to support logging for multi-objective experiments
- Updated mutation and crossover to handle within_used
- Tree generator updated for tree-based population initialisers
- Invalid individuals handing updated.

### Fixed
-  Fixed toml file to include non-pyton files like yaml, bnf in project templates
-  Logging issues fixed by connecting global logging
-  Fixed RVD intializer to remove hard-coded codon size value


## [1.0.1-beta.8](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.8) - 2026-04-10

### Added
- Selection operators: LexicaseSelection, EpsilonLexicaseSelection
- Fitness Functions: CrossEntroypyFitness, HingeLossFitness, MSEFitness
- Added VladislavlevaBenchmark benchmark for regression

### Changed
- Updated FitnessEvaluator, Parallel Executor, Algorithms, Checkpointing to support Lexicase
- Updated Fitness functions to support Case-wise evaluation
- Added validation to ensure the evaluation context contract is valid among FitnessEvaluator, selction, and fitness.
- Refactored benchmark suites for more organized folder structure

### Fixed
- Fixed random seed leak fixed in parallel execution.
- Fixed silent failure cases when incompatible phenotype runners were used
- Fixed SymbolicExpression evaluation freezing for deeply nested trigonometric functions.

## [1.0.1-beta.7](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.7) - 2026-04-05

### Added
- Added API docs and documentation for benchmarks

### Changed
- Revised Introduction section of documentation to include links to API docs.

### Fixed
- Fixed grammar parser to support angular brackets in phenotype
- Updated Cartpole grammar to make more readable with comparison symbols.

## [1.0.1-beta.6](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.6) - 2026-04-04

### Added
- Tree Logging for multi-objective GE

### Changed
- Refactored visualization module to separate tree plots
- Logging updated for cleaner output directories for single and multi-objective ge
- Updated control problem benchmarks

### Fixed
- Fixed grammar and map of SantaFe Trail benchmark
- Updated grammar for Cartpole problem comparison symbols replaced by lt, gt, ge, le.
- Updated random state handling for deterministic experiments in control benchmarks

## [1.0.1-beta.5](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.5) - 2026-04-02

### Added
- Project templates for CLI project creation

### Changed
- Updated documentation Population, Grammar, benchmarks.
- Updated logger, saving trees as txt files

### Fixed
- Fixed SymbolicExpression Safe Eval issues
- Fixed Warnings for high values in RMSE fitness by clipping high pred values before squaring

## [1.0.1-beta.4](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.4) - 2026-03-25
### Changed
- Updated documentation
- Changed grammar method in benchmarks to return Grammar instance instead of string

## [1.0.1-beta.3](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.3) - 2026-03-05
### Added
- Documentation for SymbolicExpression, GERegressor
### Fixed
- Fixed psqrt, plog issues in SymbolicExpression
- Fixed CLI when param not passed

## [1.0.1-beta.2](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.2) - 2026-02-26
### Added
- Added MAEFitness function
- Documentation for multiple classes
### Fixed
- Fixed hyperlinks in documentation to make compatible with ReadTheDocs

## [1.0.1-beta.1](https://github.com/finchGE/finchge/releases/tag/v1.0.1-beta.1) - 2026-02-25
### Added
- Initial release of finchGE
- Core Grammatical Evolution functionality
### Changed
- Initial release
### Fixed
- Initial release
