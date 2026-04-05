# Changelog

All notable changes to this project will be documented in this file.

For changelog we follow  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format and for versioning use  [Semantic Versioning](https://semver.org/)

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
