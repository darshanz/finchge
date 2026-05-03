import asyncio
import hashlib
import logging
from typing import Any, Union

from cloudpickle import cloudpickle

from finchge.config import Keys
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.fitness.fitness_functions import GEFitnessFunction
from finchge.fitness.fitness_types import EvaluationRecord, merge_fitness_results
from finchge.grammar import GenotypeMapper
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.mapper import MappingResult
from finchge.parallel.base import BaseParallelBackend
from finchge.runners.base import PhenotypeRunner
from finchge.utils.cache import CacheManager


class FitnessEvaluator:
    """
    Fitness evaluator is used to evaluate the fitness of the individuals.
    It is a wrapper for fitness function to allow convenient fitness evaluation
    especially for use cases where data and models are involved.
    For example in hyperparameter optimization or neural architecture search.
    The phenotype is converted to model using the 'phenotype_to_model' parameter.

    Supports both model-based evaluation (e.g., using train/validation data)
    and phenotype-only evaluation (e.g., string matching or symbolic tasks).

    Args:
        fitness_functions: One or more fitness function instances.
        mapper: Genotype mapper.
        runner: Runner for evaluating phenotypes.
        encode_trees: Whether to encode trees to integer genotypes. This is
            needed if genome-based operators are used after tree-based initialization.
        parallel_config: Parallel section of the config.
        require_case_data: Whether case-based evaluation is required, for example
            when using lexicase selection.
    """

    def __init__(
        self,
        fitness_functions: Union[GEFitnessFunction, list[GEFitnessFunction]],
        mapper: GenotypeMapper,
        runner: PhenotypeRunner | None = None,
        encode_trees: bool = False,
        parallel_config: dict[str, Any] | None = None,
        require_case_data: bool = False,
    ):
        if not isinstance(fitness_functions, list):
            fitness_functions = [fitness_functions]
        self.fitness_functions = fitness_functions
        self.mapper = mapper
        self.cache_manager: CacheManager[Any] | None = None
        self.encode_trees = encode_trees
        self._configured_genome_length: int | None = None
        self._configured_codon_size: int = 127
        self.runner = runner
        # require_case_data flag determines whether case wise evaluation is required. It will be set True for lexicase
        self.require_case_data = require_case_data

        # included to avoid collision in cache from different utils or if the data  environment changes
        # Not sure if this is the most efficient . More research required to make how to make caching more efficient
        # In later updates , this may also be provided by the runner (PhenotypeRunner)
        # as runner is the component owning the data.
        self._env_version: str = "static"

        # parallelization
        self.parallel_config = parallel_config
        self._parallel_enabled = (
            parallel_config.get(Keys.PARALLEL_ENABLED, False)
            if parallel_config
            else False
        )
        self._parallel_backend: BaseParallelBackend | None = None

        self._validate_case_data_support()
        self._validate_context_compatibility()

    def is_multi_objective(self) -> bool:
        """
        Determines if the evaluation is multi-objective.

        Returns:
            bool: True if more than one fitness function is provided, False otherwise.
        """
        return len(self.fitness_functions) > 1

    def set_cache_manager(self, cache_manager: CacheManager[Any]) -> None:
        self.cache_manager = cache_manager

    def get_env_version(self) -> str:
        """
        Return current environment version.
        To distinguish environments in adaptive environments.
        """
        return self._env_version

    def set_env_version(self, version: str) -> None:
        """
        Update environment version (e.g. generation, dataset update).
        """
        self._env_version = version

    def get_objective_names(self) -> list[str]:
        """
        Retrieves human-readable names of the objectives.

        Returns:
            list: List of objective names derived from fitness function class names.
        """
        return [
            type(fitness).__name__.replace("Fitness", "")
            for fitness in self.fitness_functions
        ]

    def get_fitness_functions(self) -> list[GEFitnessFunction]:
        """
        Returns the list of fitness function instances.

        Returns:
            list: List of fitness function objects.
        """
        return self.fitness_functions

    def count_objectives(self) -> int:
        """
        Counts the number of objectives.

        Returns:
            int: Number of fitness functions (objectives).
        """
        return len(self.fitness_functions)

    def get_maximize_flags(self) -> list[bool]:
        """
        Returns the maximize flags for each objective
        :return:
        """
        return [fn.maximize for fn in self.fitness_functions]

    def _get_required_keys(self) -> set[str]:
        """
        Set of all required keys from fitness functions.
        """
        keys = set()
        for fn in self.fitness_functions:
            keys.update(fn.required_context_keys)
        return keys

    def _evaluate_population_sequential(self, population: "Population") -> None:
        """
        Evaluate all individuals in a population,
        using cache if provided.
        """
        for ind in population.individuals:
            self.evaluate_individual(ind)

    def evaluate_population(self, population: "Population") -> None:
        """
        NEW: Evaluates entire population.
        Uses parallel execution if enabled, sequential otherwise.

        Users should call this instead of manually calling evaluate().
        """
        if self._parallel_enabled:
            asyncio.run(self._evaluate_population_parallel(population))
        else:
            self._evaluate_population_sequential(population)

    async def _evaluate_population_parallel(self, population: Population) -> None:
        """
        Evaluate population in parallel using executor pattern.

        Groups duplicate phenotypes, evaluates only unique individuals,
        and propagates results to all individuals with same phenotype.
        """
        if self._parallel_backend is None:
            self._parallel_backend = self._create_parallel_backend()

        # Collect unevaluated individuals and group by phenotype
        phenotype_to_indices: dict[str, list[int]] = {}
        phenotypes: list[str] = []
        for i, ind in enumerate(population.individuals):
            self.refesh_mapping(ind)

            # skip un mapped or invalids from evaluation
            if not ind.is_evaluable():
                continue
            if ind.has_fitness():
                continue

            phenotype = ind.require_phenotype()
            #  cache check
            if self.cache_manager is not None:
                cached = self.cache_manager.get_fitness(
                    phenotype=phenotype,
                    env_version=self.get_env_version(),
                )
                if cached is not None:
                    self._apply_evaluation_record(ind, cached)
                    continue

            # Group by phenotype for duplicate detection
            if phenotype not in phenotype_to_indices:
                phenotype_to_indices[phenotype] = []
                phenotypes.append(phenotype)
            phenotype_to_indices[phenotype].append(i)

        if not phenotypes:
            return

        # Keys required by the fitness function
        required_keys = self._get_required_keys()

        # Create minimal contexts for unique phenotypes
        contexts = []
        for phenotype in phenotypes:
            # Get seed from mapper for reproducibility
            mapper_seed_info = self.mapper.get_seed_info()
            seed: int | None = mapper_seed_info.get("effective_seed")
            if seed is None:
                seed = 42
                logging.info("GenotypeMapper: Seed not set. Using default seed: 42")
            eval_seed = self.derive_eval_seed(
                base_seed=seed, phenotype=phenotype, env_version=self.get_env_version()
            )

            context = {
                "phenotype": phenotype,
                "seed": eval_seed,
                "required_keys": required_keys,
                "require_case_data": self.require_case_data,
            }

            if self.runner:
                context["runner"] = self.runner
            contexts.append(context)

        # Evaluate unique phenotypes in parallel
        from finchge.parallel.thread_pool_backend import (  # importing here to avoid circular dependency
            ThreadPoolBackend,
        )

        if isinstance(self._parallel_backend, ThreadPoolBackend):
            unique_records = await self._parallel_backend.evaluate_batch(
                contexts=contexts,  # List of dicts with runner, phenotype, seed
                fitness_functions=self.fitness_functions,  # Direct list
            )
        else:
            # For process backend - still need pickling
            unique_records = await self._parallel_backend.evaluate_batch(
                cloudpickle.dumps(contexts), cloudpickle.dumps(self.fitness_functions)
            )

        # Map results back to phenotypes
        phenotype_to_record = dict(zip(phenotypes, unique_records))
        # Propagate fitness to all individuals (including duplicates)
        for phenotype, idxs in phenotype_to_indices.items():
            record = phenotype_to_record[phenotype]

            for idx in idxs:
                ind = population.individuals[idx]
                self._apply_evaluation_record(ind, record)

                if self.cache_manager is not None:
                    self.cache_manager.set_fitness(
                        phenotype=phenotype,
                        env_version=self.get_env_version(),
                        fitness=record,
                    )

    def evaluate_individual(self, individual: "Individual") -> None:
        """
        Evaluates a given phenotype by training the corresponding model and applying fitness functions.

        Args:
            individual (Individual): An individual configuration or representation used to generate a model.

        Returns:
            list: A list of fitness scores corresponding to each fitness function.
        """
        self.refesh_mapping(individual)
        if not individual.is_evaluable():
            return

        phenotype = individual.require_phenotype()

        # Build cache key
        if self.cache_manager is not None:
            # env version will protect against using wrong fitness from cache in dynamic systems
            # if data changes env should change
            cached = self.cache_manager.get_fitness(
                phenotype=phenotype,
                env_version=self.get_env_version(),
            )
            if cached is not None:
                self._apply_evaluation_record(individual, cached)
                return

        record = self._evaluate_in_context(individual)
        self._apply_evaluation_record(individual, record)

        # Store in cache
        if self.cache_manager is not None:
            self.cache_manager.set_fitness(
                phenotype=phenotype,
                env_version=self.get_env_version(),
                fitness=record,
            )

    def _evaluate_in_context(self, individual: "Individual") -> EvaluationRecord:
        # Keys required by the fitness function
        required_keys = self._get_required_keys()

        phenotype = individual.require_phenotype()

        # prepare context for evaluation
        eval_context: dict[str, Any] = {
            "phenotype": phenotype,
            "require_case_data": self.require_case_data,
        }
        if self.runner:
            result_context = self.runner.run(
                phenotype=phenotype, context_hints=required_keys
            )
            eval_context.update(result_context)
            # Compute fitness and return

        self._validate_context_keys(eval_context, required_keys, individual)

        results = [fn.evaluate(eval_context) for fn in self.fitness_functions]
        return merge_fitness_results(results)

    def _create_parallel_backend(self) -> BaseParallelBackend:
        """Create parallel backend from config dictionary"""
        from finchge.parallel.factory import ParallelBackendFactory

        if not self.parallel_config:
            raise ValueError("Parallel Config cannot be null.")
        return ParallelBackendFactory.create_backend(self.parallel_config)

    async def shutdown(self) -> None:
        """Cleanup parallel resources"""
        if self._parallel_backend is not None:
            await self._parallel_backend.shutdown()

    def refesh_mapping(self, ind: "Individual") -> None:
        """
        Ensure that genotype, phenotype, and tree are consistent.

        Rules:
            - If already mapped, do nothing.
            - If tree exists and genotype is absent:
                - reverse-map genotype when encode_trees=True
                - otherwise derive phenotype directly from tree
            - If genotype exists, map through the mapper

        """
        if ind.genotype is None and ind.tree is None:
            raise RuntimeError("Individual has neither genotype nor tree.")

        if not self.mapper:
            raise RuntimeError(
                "Mapper is required. No mapper was passed to the constructor of FitnessEvaluator"
            )

        # Already mapped: either valid phenotype exists or the individual is known invalid.
        if ind.is_mapped():
            return

        # When there is tree and no genotype :
        # genotypes are not needed , however encode_trees flag can be used to reverse map to genotype anyway
        if ind.genotype is None and ind.tree is not None:
            if self.encode_trees:
                genome_length = getattr(self, "_configured_genome_length", None)
                codon_size = getattr(self, "_configured_codon_size", 127)

                try:
                    ind.genotype = self.mapper.reverse_map(
                        tree=ind.tree,
                        codon_size=codon_size,
                        pad_to_length=genome_length,
                        pad_mode="random",
                    )
                except ValueError:
                    ind.genotype = None
                    ind.phenotype = None
                    ind.invalid = True
                    return

            else:
                ind.phenotype = TreeNode.from_string(ind.tree).to_phenotype()
                ind.invalid = False  # trees are always valid

        # whether there is tree or not we have to map it to phenotype
        if ind.genotype is not None:
            mapping_result: MappingResult = self.mapper.map(ind.genotype)
            ind.phenotype = mapping_result.phenotype
            ind.used_genome = mapping_result.used_genome
            ind.used_codon_count = mapping_result.used_codon_count
            ind.invalid = mapping_result.invalid
            ind.tree = mapping_result.tree_str

            if mapping_result.invalid:
                ind.mark_invalid()

    def refresh_mapping_all(self, individuals: list[Individual]) -> None:
        for ind in individuals:
            self.refesh_mapping(ind)

    def clear_cache(self) -> None:
        if self.cache_manager:
            self.cache_manager.clear()

    def derive_eval_seed(self, base_seed: int, phenotype: str, env_version: str) -> int:
        """
        Although we share same random state using private RNG shared in different components,
        It will not work for parallelized evaluation.
        Each process, or even each machine, will run evaluations in a different order, so random state will fail.

        So, we must use some deterministic way to evaluate. One way is to use different seeds in deterministic way
        Same phenotype and base seed combination should result in same seed in each run.

        So we derive a 32-bit seed for evaluation/training in a worker process.
        Same (base_seed, phenotype, env_version) will give same evaluation seed.
        """
        h = hashlib.blake2b(digest_size=8)  # 64-bit digest
        h.update(str(base_seed).encode("utf-8"))
        h.update(b"|")
        h.update(str(phenotype).encode("utf-8"))
        if env_version is not None:
            h.update(b"|")
            h.update(env_version.encode("utf-8"))

        # Convert digest to an integer seed in [0, 2**32-1]
        seed64 = int.from_bytes(h.digest(), byteorder="big", signed=False)
        return seed64 % (2**32)

    def _validate_case_data_support(self) -> None:
        if not self.require_case_data:
            return

        for fn in self.fitness_functions:
            if getattr(fn, "case_data_key", None) is None:
                raise ValueError(
                    f"{type(fn).__name__} does not support case data required for lexicase selection."
                )

    def _apply_evaluation_record(
        self,
        individual: Individual,
        record: EvaluationRecord,
    ) -> None:
        # set fitness and case metadata to the individual
        individual.fitness = record.fitness

        if record.case_data:
            individual.set_meta(Individual.CASE_DATA_META_KEY, record.case_data)
        else:
            individual.remove_meta(Individual.CASE_DATA_META_KEY)

        for key, value in record.meta.items():
            individual.set_meta(key, value)

    def _validate_context_compatibility(self) -> None:
        # Validate the runner-fitness combination to make sure runner can
        # provide the keys required by fitness function

        # if there is no runner we don't need to validate. for example in stringmatch problem the runner is not needed
        if not self.runner:
            return

        required_keys = self._get_required_keys()
        builtin_keys = {"phenotype", "require_case_data"}

        runner_keys: set[str] = set()
        if self.runner is not None:
            runner_keys = getattr(self.runner, "provided_context_keys", set())

        available_keys = builtin_keys | runner_keys
        missing_keys = required_keys - available_keys

        if missing_keys:
            runner_name = (
                type(self.runner).__name__ if self.runner is not None else "None"
            )
            raise ValueError(
                f"FitnessEvaluator context contract is invalid."
                f"Required context keys {sorted(missing_keys)} are not provided by "
                f"runner {runner_name}"
            )

    def _validate_context_keys(
        self,
        eval_context: dict[str, Any],
        required_keys: set[str],
        individual: "Individual",
    ) -> None:
        # if there is no runner we don't need to validate. for example in stringmatch problem the runner is not needed
        if not self.runner:
            return

        # validate context keys to avoid failing when wrong runner is provided.
        missing = sorted(key for key in required_keys if key not in eval_context)
        if missing:
            raise ValueError(
                f"Evaluation context for phenotype {individual.phenotype!r} is missing "
                f"required keys: {missing}. "
                f"Ensure the required keys are provided by the runner: ({type(self.runner).__name__}) "
            )
