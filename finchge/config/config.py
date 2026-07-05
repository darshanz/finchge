import ast
import configparser
import copy
import glob
import json
from typing import Any, Tuple

import yaml
from tabulate import tabulate

from finchge.utils.display_utils import display_html, is_jupyter


class Keys:
    # Experiment
    RANDOM_SEED = "random_seed"
    NUM_GENERATIONS = "num_generations"
    VERBOSE = "verbose"  # Enable verbose logging
    EXPT_LOGGER_ENABLED = "expt_logger_enabled"  # Enable full expt logging
    EXCLUDE_LOGS = "exclude_logs"  # List of log directories to be excluded

    # Core
    POPULATION_SIZE = "population_size"
    GRAMMAR_FILE = "grammar_file"
    CODON_SIZE = "codon_size"
    GENOME_LENGTH = "genome_length"
    MAX_WRAPS = "max_wraps"
    MAX_RECURSION_DEPTH = "max_recursion_depth"

    # Tree-related
    MAX_TREE_DEPTH = "max_tree_depth"  # limit on tree depth
    INIT_MIN_DEPTH = "init_min_depth"  # Minimum depth for Full and Grow initialisation
    INIT_MAX_DEPTH = "init_max_depth"  # Maximum depth for tree based initialisation
    PTC2_TARGET_SIZE = "ptc2_target_size"  # Target tree size for PCT2
    INIT_TREE_MIN_SIZE = "init_tree_min_size"  # for Ramped PTC2
    INIT_TREE_MAX_SIZE = "init_tree_max_size"  # for Ramped PTC2
    DETERMINISTIC_RAMPED_PTC2 = (
        "deterministic_ramped_ptc2"  # Is determnistic flag for ramped ptc2
    )
    INIT_TREE_STRICT_FULL = "init_tree_strict_full"  # no fallback if grammar does not support full expansion

    MUTATION_MAX_DEPTH = (
        "mutation_max_depth"  # Maximum depth of newly generated subtrees
    )
    INIT_TYPE = "init_type"  # random_genome | rvd | pi_grow

    # operators
    MUTATION_PROBABILITY = "mutation_probability"
    CROSSOVER_PROBABILITY = "crossover_probability"
    ELITE_SIZE = "elite_size"
    TOURNAMENT_SIZE = "tournament_size"

    # Caching
    CACHE_TYPE = "cache_type"  # none | lru | disk
    CACHE_SIZE = "cache_size"  # Cache capacity (entries or bytes)

    # For parallel Execution
    PARALLEL_ENABLED = "parallel_enabled"
    EXECUTOR_TYPE = "executor_type"  # process | thread
    MAX_WORKERS = "max_workers"
    BATCH_SIZE = "batch_size"

    # For Island GA
    NUM_ISLANDS = "num_islands"
    MIGRATION_INTERVAL = "migration_interval"
    MIGRATION_SIZE = "migration_size"


class FinchConfig:
    """
    Configuration manager for Grammar Evolution (GE) utils.

    FinchConfig provides an interface for loading, accessing, and modifying
    GE configuration from YAML and INI files.

    Can load conifg as dictionaries (dict[str, dict[str, Any]]) through a construtor of from_file.
    Supports automatic file discovery and typed access to configuration sections.
    When working in a single proplem project with one configuration, FinchConfig can detect the file automatiaally.
    Although automatic config detection is handy during quick tasks such as checking grammar and other components.
    For more organized utils it is recommended to provide the actual file. As this class may silently pick the
    config file from current working directory if available any.

    Attributes:
        _data: Internal storage for configuration data as nested dictionaries.

    """

    def __init__(self, data: dict[str, dict[str, Any]]) -> None:
        """Initializes a FinchConfig instance with configuration data.

        Args:
            data: Nested dictionary where outer keys are section names and inner
                dictionaries contain section key-value pairs.
                Example:
                ```
                    {"eperiment": {"start_symbol": "<expr>"}, ...}
                ```

        Note:
            Prefer using the class methods `from_file`, `from_yaml`, or `from_ini`
            to create instances rather than calling this constructor directly.
        """
        self._data = data

    @classmethod
    def from_file(cls, path: str | None = None) -> "FinchConfig":
        """Creates a FinchConfig instance from a configuration file.

        Automatically detects file format based on extension and loads the
        configuration. If no path is provided, searches for common config
        filenames in the current directory.

        If the config files follow expected name (ge_config) and are in current working directory,
        they can be  automatically detected. Othewise they have to be passed
        as path argument.

        Args:
            path: Optional path to configuration file. If None, searches for
                'ge_config.yaml', 'ge_config.yml', or 'ge_config.ini' in the
                current directory.

        Returns:
            FinchConfig instance loaded with configuration data.

        Raises:
            FileNotFoundError: If no configuration file is found.
            RuntimeError: If multiple configuration files are found (when path
                is None).
            ValueError: If the file format is unsupported or the file content
                is invalid.
        """
        if path is None:
            # Define patterns to search for
            patterns = ["*.yaml", "*.yml", "*.ini"]
            found = []
            for pattern in patterns:
                found.extend(glob.glob(pattern))

            if len(found) == 0:
                raise FileNotFoundError(
                    "No config files (.yaml/.yml/.ini) found in the working directory."
                )
            if len(found) > 1:
                raise RuntimeError(
                    f"Multiple config files found: {found}. Please specify filename."
                )
            path = found[0]

        if path.endswith((".yaml", ".yml")):
            return cls.from_yaml(path)
        if path.endswith(".ini"):
            return cls.from_ini(path)

        raise ValueError(f"Unsupported config format: {path}")

    @classmethod
    def from_ini(cls, path: str) -> "FinchConfig":
        """Creates a FinchConfig instance from an INI configuration file.

        Parses INI sections and converts string values to appropriate Python types
        using `ast.literal_eval`. Values that cannot be evaluated remain as strings.

        Args:
            path: Path to the INI configuration file.

        Returns:
            FinchConfig instance with parsed configuration data.

        Raises:
            FileNotFoundError: If the INI file does not exist.
            configparser.Error: If the INI file is malformed.

        Example INI format:
            ```
            [grammar]
            start_symbol = "<expr>"
            max_depth = 10

            [initialisation]
            population_size = 100
            method = "ramped_half_and_half"
            ```
        """
        parser = configparser.ConfigParser()
        parser.read(path)

        def safe_eval(v: str) -> Any:
            """Safely evaluates a string to a Python literal.

            Attempts to parse the string as a Python literal (string, number,
            list, dict, etc.). If parsing fails, returns the original string.

            Args:
                v: String value to evaluate.

            Returns:
                Evaluated Python object or the original string if evaluation fails.
            """
            try:
                return ast.literal_eval(v)
            except Exception:
                return v

        data: dict[str, dict[str, Any]] = {}

        for section in parser.sections():
            data[section] = {key: safe_eval(val) for key, val in parser.items(section)}

        return cls(data)

    @classmethod
    def from_yaml(cls, path: str) -> "FinchConfig":
        """Creates a FinchConfig instance from a YAML configuration file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            FinchConfig instance with parsed configuration data.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
            ValueError: If the top-level YAML structure is not a dictionary.

        Example YAML format:
        ```
            grammar:
                start_symbol: "<expr>"
                max_depth: 10
            initialisation:
                population_size: 100
                method: "ramped_half_and_half"
        ```
        """
        with open(path, "r") as f:
            data: dict[str, dict[str, Any]] = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML config must be a mapping at top level")

        return cls(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinchConfig":
        """
        Loads dict as FinchConfig. Internally finchGE uses FinchConfig class.
        Args:
            data (dict[str, Any]): configuration as a dictionary

        Returns: FinchConfig instance

        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict type.")
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the FinchConfig instance into a dict

        Returns: dictionary for the FinchConfig

        """
        return copy.deepcopy(self._data)

    def to_json(self, indent: int = 2) -> str:
        """
        Converts the GECOnfig instance to json
        Args:
            indent: indent for the json

        Returns: json string

        """
        return json.dumps(self._data, indent=indent)

    def to_table(self, tablefmt: str = "simple") -> str:
        rows: list[list[str]] = []
        for section, values in self._data.items():
            rows.append([f"[{section}]", "", ""])
            for key, val in values.items():
                rows.append(["", key, repr(val)])
        return tabulate(rows, headers=["Section", "Key", "Value"], tablefmt=tablefmt)

    def display(self) -> None:
        in_notebook: bool = is_jupyter()
        if in_notebook:
            html_tbl = self.to_table(tablefmt="html")
            display_html(html_tbl)
        else:
            print(self.to_table())

    def copy(self, update: dict[str, dict[str, Any]] | None = None) -> "FinchConfig":
        """Creates a deep copy of the configuration with optional updates.

        Args:
            update: Optional dictionary of updates to apply to the copy.
                Format: {section_name: {key: value, ...}, ...}
                If a section doesn't exist, it will be created.

        Returns:
            A new FinchConfig instance with the copied (and optionally updated) data.

        Example:
            ```
            >>> new_config = config.copy({
            ...     "experiment": {"num_generations": 500},
            ...     "new_section": {"key": "value"}
            ... })
            ```
        """
        data = copy.deepcopy(self._data)
        if update:
            for section, values in update.items():
                data.setdefault(section, {}).update(values)
        return FinchConfig(data)

    def section(self, name: str) -> dict[str, Any]:
        """Retrieves a configuration section by name.

        Args:
            name: Name of the configuration section.

        Returns:
            Dictionary containing the section's key-value pairs.
            Returns an empty dictionary if the section does not exist.
        """
        return self._data.get(name, {})

    @property
    def ge(self) -> dict[str, Any]:
        """Retrieves the grammatical evolution configuration section.

        Typically contains:
            - start_symbol: The starting non-terminal symbol
            - rules: Grammar production rules
            - max_depth: Maximum derivation depth
            - crossover_rate: Probability of applying crossover
            - mutation_rate: Probability of applying mutation
            - selection_method: Selection strategy (tournament, roulette, etc.)
            - tournament_size: If using tournament selection
            - other related parameters

        Returns:
            Dictionary of grammar configuration parameters.
        """
        return self.section("ge")

    @property
    def experiment(self) -> dict[str, Any]:
        """Retrieves the experiment configuration section.
        Returns:
            Dictionary of experiment configuration parameters.
        """
        return self.section("experiment")

    @property
    def parallel(self) -> dict[str, Any]:
        """Retrieves the parallel configuration section.

        Returns:
            Dictionary of experiment configuration parameters.
        """
        return self.section("parallel")

    @classmethod
    def default(self) -> dict[str, Any]:
        default_config = {
            "experiment": {
                "random_seed": 42,
                "num_generations": 100,
                "verbose": False,
                "cache_type": "lru",
                "cache_size": 128,
            },
            "ge": {
                "population_size": 100,
                "codon_size": 127,
                "max_wraps": 6,
                "max_recursion_depth": 20,
                "genome_length": 100,
                "init_type": "random_genome",
                "mutation_probability": 0.01,
                "crossover_probability": 0.5,
                "elite_size": 3,
                "tournament_size": 3,
            },
            "parallel": {
                "parallel_enabled": False,
            },
        }
        return default_config


class ConfigError(Exception):
    """Raised when a FinchGE configuration is invalid."""


_INIT_TYPES = {
    "random_genome",
    "rvd",
    "full",
    "grow",
    "pi_grow",
    "rhh",
    "ptc2",
    "ramped_ptc2",
}
_CACHE_TYPES = {"none", "lru", "disk"}
_EXECUTOR_TYPES = {"process", "thread"}


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any, *, min_value: int | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return min_value is None or value >= min_value


def _is_probability(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 1.0


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_one_of(value: Any, allowed: set[str]) -> bool:
    return isinstance(value, str) and value.lower() in allowed


class ConfigValidator:
    """
    Defines validation rules for FinchGE configuration files.

    This class contains the schema and validation logic used to verify
    the structure and values of a FinchGE configuration.

    Attributes:
        MANDATORY_SECTIONS (set[str]):
            Sections that must be present for a valid configuration.

        RECOMMENDED_SECTIONS (set[str]):
            Sections that are optional but strongly recommended.

        OPTIONAL_SECTIONS (set[str]):
            Fully optional sections.

        REQUIRED_FIELDS (dict[str, set[str]]):
            Required fields for each configuration section.

        FIELD_VALIDATORS (dict[str, dict[str, Callable]]):
            Field-level validators keyed by section and field name.
    """

    MANDATORY_SECTIONS: set[str] = {"experiment", "ge"}
    OPTIONAL_SECTIONS: set[str] = {"parallel"}

    INIT_TYPES: set[str] = _INIT_TYPES
    CACHE_TYPES: set[str] = _CACHE_TYPES
    EXECUTOR_TYPES: set[str] = _EXECUTOR_TYPES

    KNOWN_FIELDS: dict[str, set[str]] = {
        "experiment": {
            Keys.RANDOM_SEED,
            Keys.NUM_GENERATIONS,
            Keys.VERBOSE,
            Keys.EXPT_LOGGER_ENABLED,
            Keys.EXCLUDE_LOGS,
            Keys.CACHE_TYPE,
            Keys.CACHE_SIZE,
        },
        "ge": {
            Keys.POPULATION_SIZE,
            Keys.GRAMMAR_FILE,
            Keys.CODON_SIZE,
            Keys.GENOME_LENGTH,
            Keys.MAX_WRAPS,
            Keys.MAX_RECURSION_DEPTH,
            Keys.MAX_TREE_DEPTH,
            Keys.INIT_MIN_DEPTH,
            Keys.INIT_MAX_DEPTH,
            Keys.PTC2_TARGET_SIZE,
            Keys.INIT_TREE_MIN_SIZE,
            Keys.INIT_TREE_MAX_SIZE,
            Keys.DETERMINISTIC_RAMPED_PTC2,
            Keys.INIT_TREE_STRICT_FULL,
            Keys.MUTATION_MAX_DEPTH,
            Keys.INIT_TYPE,
            Keys.MUTATION_PROBABILITY,
            Keys.CROSSOVER_PROBABILITY,
            Keys.ELITE_SIZE,
            Keys.TOURNAMENT_SIZE,
            Keys.NUM_ISLANDS,
            Keys.MIGRATION_INTERVAL,
            Keys.MIGRATION_SIZE,
        },
        "parallel": {
            Keys.PARALLEL_ENABLED,
            Keys.EXECUTOR_TYPE,
            Keys.MAX_WORKERS,
            Keys.BATCH_SIZE,
        },
    }

    REQUIRED_FIELDS = {
        "experiment": {
            Keys.RANDOM_SEED,
            Keys.NUM_GENERATIONS,
        },
        "ge": {
            Keys.POPULATION_SIZE,
            Keys.INIT_TYPE,
            Keys.MUTATION_PROBABILITY,
            Keys.CROSSOVER_PROBABILITY,
            Keys.ELITE_SIZE,
        },
    }

    FIELD_VALIDATORS: dict[str, dict[str, Any]] = {
        "experiment": {
            Keys.NUM_GENERATIONS: lambda v: _is_int(v, min_value=1),
            Keys.RANDOM_SEED: _is_int,
            Keys.VERBOSE: _is_bool,
            Keys.EXPT_LOGGER_ENABLED: _is_bool,
            Keys.EXCLUDE_LOGS: _is_string_list,
            Keys.CACHE_TYPE: lambda v: _is_one_of(v, _CACHE_TYPES),
            Keys.CACHE_SIZE: lambda v: _is_int(v, min_value=1),
        },
        "ge": {
            Keys.POPULATION_SIZE: lambda v: _is_int(v, min_value=1),
            Keys.GRAMMAR_FILE: _is_non_empty_string,
            Keys.GENOME_LENGTH: lambda v: _is_int(v, min_value=1),
            Keys.CODON_SIZE: lambda v: _is_int(v, min_value=1),
            Keys.MAX_WRAPS: lambda v: _is_int(v, min_value=0),
            Keys.MAX_RECURSION_DEPTH: lambda v: _is_int(v, min_value=1),
            Keys.MAX_TREE_DEPTH: lambda v: _is_int(v, min_value=1),
            Keys.INIT_MIN_DEPTH: lambda v: _is_int(v, min_value=1),
            Keys.INIT_MAX_DEPTH: lambda v: _is_int(v, min_value=1),
            Keys.PTC2_TARGET_SIZE: lambda v: _is_int(v, min_value=1),
            Keys.INIT_TREE_MIN_SIZE: lambda v: _is_int(v, min_value=2),
            Keys.INIT_TREE_MAX_SIZE: lambda v: _is_int(v, min_value=2),
            Keys.DETERMINISTIC_RAMPED_PTC2: _is_bool,
            Keys.INIT_TREE_STRICT_FULL: _is_bool,
            Keys.MUTATION_MAX_DEPTH: lambda v: _is_int(v, min_value=1),
            Keys.INIT_TYPE: lambda v: _is_one_of(v, _INIT_TYPES),
            Keys.MUTATION_PROBABILITY: _is_probability,
            Keys.CROSSOVER_PROBABILITY: _is_probability,
            Keys.ELITE_SIZE: lambda v: _is_int(v, min_value=0),
            Keys.TOURNAMENT_SIZE: lambda v: _is_int(v, min_value=1),
            Keys.NUM_ISLANDS: lambda v: _is_int(v, min_value=2),
            Keys.MIGRATION_INTERVAL: lambda v: _is_int(v, min_value=1),
            Keys.MIGRATION_SIZE: lambda v: _is_int(v, min_value=1),
        },
        "parallel": {
            Keys.PARALLEL_ENABLED: _is_bool,
            Keys.EXECUTOR_TYPE: lambda v: _is_one_of(v, _EXECUTOR_TYPES),
            Keys.MAX_WORKERS: lambda v: _is_int(v, min_value=1),
            Keys.BATCH_SIZE: lambda v: _is_int(v, min_value=1),
        },
    }


def validate_parallel_config(config: dict[str, Any]) -> Tuple[list[str], list[str]]:
    """

      Args:
        config (dict): dictionary of config in parallel secion

    Returns:
        Tuple[list[str], list[str]]:
            - issues: A list of validation error messages. If non-empty, the
              configuration should be considered invalid and execution should stop.
            - warnings: A list of warning messages indicating potential problems
              or non-standard configuration usage.

    """
    issues: list[str] = []
    warnings: list[str] = []

    if not config:
        return issues, warnings

    enabled = config.get(Keys.PARALLEL_ENABLED, False)
    if not isinstance(enabled, bool):
        issues.append(
            f"Invalid value for parallel.{Keys.PARALLEL_ENABLED}: {enabled!r}"
        )
        return issues, warnings

    executor_type = config.get(Keys.EXECUTOR_TYPE, "process")
    if not _is_one_of(executor_type, _EXECUTOR_TYPES):
        issues.append(
            f"Invalid value for parallel.{Keys.EXECUTOR_TYPE}: {executor_type!r}"
        )

    max_workers = config.get(Keys.MAX_WORKERS)
    if max_workers is not None and not _is_int(max_workers, min_value=1):
        issues.append(f"Invalid value for parallel.{Keys.MAX_WORKERS}: {max_workers!r}")

    batch_size = config.get(Keys.BATCH_SIZE)
    if batch_size is not None and not _is_int(batch_size, min_value=1):
        issues.append(f"Invalid value for parallel.{Keys.BATCH_SIZE}: {batch_size!r}")

    if enabled and max_workers is None:
        warnings.append(
            "parallel.max_workers is not set; FinchGE will use the backend default"
        )
    if enabled and batch_size is None:
        warnings.append("parallel.batch_size is not set; FinchGE will use 10")

    return issues, warnings


def _section_is_mapping(
    section: str,
    config: FinchConfig,
    issues: list[str],
) -> bool:
    value = config._data.get(section)
    if isinstance(value, dict):
        return True
    issues.append(f"Section '{section}' must be a mapping")
    return False


def _missing_fields(section_data: dict[str, Any], required: set[str]) -> set[str]:
    return required - set(section_data.keys())


def _validate_known_fields(
    section: str,
    section_data: dict[str, Any],
    warnings: list[str],
) -> None:
    known_fields = ConfigValidator.KNOWN_FIELDS.get(section)
    if not known_fields:
        return

    unknown_fields = set(section_data.keys()) - known_fields
    if unknown_fields:
        warnings.append(
            f"Section '{section}' contains unknown fields: "
            f"{', '.join(sorted(unknown_fields))}"
        )


def _validate_field_values(
    section: str,
    section_data: dict[str, Any],
    issues: list[str],
) -> None:
    validators = ConfigValidator.FIELD_VALIDATORS.get(section, {})
    for field, value in section_data.items():
        validator = validators.get(field)
        if validator is None:
            continue

        try:
            valid = validator(value)
        except Exception as e:
            issues.append(f"Validation error for {section}.{field}: {e}")
            continue

        if not valid:
            issues.append(f"Invalid value for {section}.{field}: {value!r}")


def _validate_initialiser_config(
    ge: dict[str, Any],
    issues: list[str],
    warnings: list[str],
) -> None:
    init_type_value = ge.get(Keys.INIT_TYPE, "random_genome")
    if not isinstance(init_type_value, str):
        return

    init_type = init_type_value.lower()
    required_by_init = {
        "random_genome": {Keys.GENOME_LENGTH, Keys.CODON_SIZE},
        "rvd": {Keys.GENOME_LENGTH, Keys.CODON_SIZE, Keys.POPULATION_SIZE},
        "full": {Keys.INIT_MIN_DEPTH, Keys.INIT_MAX_DEPTH},
        "grow": {Keys.INIT_MIN_DEPTH, Keys.INIT_MAX_DEPTH},
        "rhh": {Keys.INIT_MAX_DEPTH, Keys.POPULATION_SIZE},
        "pi_grow": {Keys.INIT_MAX_DEPTH, Keys.POPULATION_SIZE},
        "ptc2": {Keys.PTC2_TARGET_SIZE},
        "ramped_ptc2": {
            Keys.INIT_TREE_MIN_SIZE,
            Keys.INIT_TREE_MAX_SIZE,
            Keys.POPULATION_SIZE,
        },
    }

    missing = _missing_fields(ge, required_by_init.get(init_type, set()))
    if missing:
        warnings.append(
            f"ge.init_type '{init_type}' usually needs fields: "
            f"{', '.join(sorted(missing))}. "
            "This is valid only if they are supplied programmatically."
        )

    min_depth = ge.get(Keys.INIT_MIN_DEPTH)
    max_depth = ge.get(Keys.INIT_MAX_DEPTH)
    if (
        isinstance(min_depth, int)
        and isinstance(max_depth, int)
        and min_depth > max_depth
    ):
        issues.append("ge.init_min_depth must be <= ge.init_max_depth")

    min_size = ge.get(Keys.INIT_TREE_MIN_SIZE)
    max_size = ge.get(Keys.INIT_TREE_MAX_SIZE)
    if isinstance(min_size, int) and isinstance(max_size, int) and min_size > max_size:
        issues.append("ge.init_tree_min_size must be <= ge.init_tree_max_size")

    if init_type in {"full", "grow", "rhh", "pi_grow", "ptc2", "ramped_ptc2"}:
        if Keys.MAX_TREE_DEPTH not in ge:
            warnings.append(
                "ge.max_tree_depth is not set for tree-based initialisation; "
                "the engine may require it when it builds a TreeGenerator"
            )

    if Keys.GRAMMAR_FILE not in ge:
        warnings.append(
            "ge.grammar_file is not set; this is valid only when a Grammar object "
            "is supplied directly to the engine or benchmark template"
        )

    runtime_fields = {
        Keys.GENOME_LENGTH,
        Keys.CODON_SIZE,
        Keys.MAX_WRAPS,
        Keys.MAX_RECURSION_DEPTH,
    }
    missing_runtime_fields = _missing_fields(ge, runtime_fields) - missing
    if missing_runtime_fields:
        warnings.append(
            "Some mapping/genome fields are not set in ge: "
            f"{', '.join(sorted(missing_runtime_fields))}. "
            "This is valid only if the relevant components are configured "
            "programmatically."
        )


def _validate_cross_field_config(
    config: FinchConfig,
    issues: list[str],
    warnings: list[str],
) -> None:
    ge = config.ge if isinstance(config.ge, dict) else {}
    experiment = config.experiment if isinstance(config.experiment, dict) else {}

    pop_size = ge.get(Keys.POPULATION_SIZE)
    elite_size = ge.get(Keys.ELITE_SIZE)
    if (
        isinstance(pop_size, int)
        and not isinstance(pop_size, bool)
        and isinstance(elite_size, int)
        and not isinstance(elite_size, bool)
        and elite_size >= pop_size
    ):
        issues.append(
            f"ge.elite_size ({elite_size}) must be smaller than "
            f"ge.population_size ({pop_size})"
        )

    tournament_size = ge.get(Keys.TOURNAMENT_SIZE)
    if (
        isinstance(pop_size, int)
        and not isinstance(pop_size, bool)
        and isinstance(tournament_size, int)
        and not isinstance(tournament_size, bool)
        and tournament_size > pop_size
    ):
        warnings.append(
            f"ge.tournament_size ({tournament_size}) is larger than "
            f"ge.population_size ({pop_size})"
        )

    genome_length = ge.get(Keys.GENOME_LENGTH)
    if isinstance(genome_length, int) and not isinstance(genome_length, bool):
        if genome_length < 2:
            warnings.append(
                "ge.genome_length is very small; this may prevent deep derivations"
            )

    cache_type = experiment.get(Keys.CACHE_TYPE)
    cache_size = experiment.get(Keys.CACHE_SIZE)
    if isinstance(cache_type, str) and cache_type.lower() == "none" and cache_size:
        warnings.append("experiment.cache_size is ignored when cache_type is 'none'")

    if ge:
        _validate_initialiser_config(ge, issues, warnings)


def validate_config(config: "FinchConfig") -> Tuple[list[str], list[str]]:
    """
    Validate a FinchGE configuration object.

    This function performs validation of the FinchGE configuration. The configuration is expected to follow the
    new two-section layout:

        - experiment: controls experiment orchestration (population size,
          generations, random seed, logging, etc.)
        - ge: controls Grammatical Evolution behavior (grammar, genome,
          initialisation, operators, and limits)

    Validation includes:
        1. Presence of mandatory top-level sections.
        2. Presence of required fields within each section.
        3. Type and range validation for known configuration fields.
        4. Cross-field consistency checks (e.g., elite size vs population size).
        5. Detection of unknown or custom sections (reported as warnings).

    The function is intentionally conservative: missing or invalid required
    fields are reported as hard errors, while unknown sections or potentially
    problematic values are reported as warnings.

    Args:
        config (FinchConfig): Parsed FinchGE configuration object.

    Returns:
        Tuple[list[str], list[str]]:
            - issues: A list of validation error messages. If non-empty, the
              configuration should be considered invalid and execution should stop.
            - warnings: A list of warning messages indicating potential problems
              or non-standard configuration usage.

    Raises:
        None. All validation problems are returned as messages instead of raising
        exceptions, allowing the caller to decide how to handle them.

    """

    issues: list[str] = []
    warnings: list[str] = []

    if not isinstance(config, FinchConfig):
        return [
            f"config must be a FinchConfig instance, got {type(config).__name__}"
        ], []

    if not isinstance(config._data, dict):
        return ["Config data must be a mapping"], []

    config_sections = set(config._data.keys())

    # Required sections
    missing_sections = ConfigValidator.MANDATORY_SECTIONS - config_sections
    if missing_sections:
        issues.append(
            f"Missing mandatory sections: {', '.join(sorted(missing_sections))}"
        )

    # Unknown sections
    known_sections = (
        ConfigValidator.MANDATORY_SECTIONS | ConfigValidator.OPTIONAL_SECTIONS
    )
    unknown_sections = config_sections - known_sections
    if unknown_sections:
        warnings.append(
            f"Unknown / custom sections found: {', '.join(sorted(unknown_sections))}"
        )

    valid_sections: set[str] = set()
    for section in config_sections & known_sections:
        if _section_is_mapping(section, config, issues):
            valid_sections.add(section)

    # Required fields per section
    for section, required_fields in ConfigValidator.REQUIRED_FIELDS.items():
        if section not in valid_sections:
            continue

        section_data = config.section(section)
        missing_fields = required_fields - set(section_data.keys())
        if missing_fields:
            issues.append(
                f"Section '{section}' missing required fields: "
                f"{', '.join(sorted(missing_fields))}"
            )

    for section in valid_sections:
        section_data = config.section(section)
        _validate_known_fields(section, section_data, warnings)
        if section != "parallel":
            _validate_field_values(section, section_data, issues)

    if "parallel" in valid_sections:
        parallel_issues, parallel_warnings = validate_parallel_config(config.parallel)
        issues.extend(parallel_issues)
        warnings.extend(parallel_warnings)

    _validate_cross_field_config(config, issues, warnings)

    return issues, warnings
