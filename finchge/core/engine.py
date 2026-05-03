import logging
import timeit
import warnings
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from finchge.algorithm.ga import GeneticAlgorithm
from finchge.algorithm.utils import validate_algorithm_fitness_match
from finchge.config.config import FinchConfig, Keys
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.core.result import GEResult
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.grammar import Grammar
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.base import GEInitialiser, GETreeInitialiser
from finchge.initialisation.factory import make_initialiser
from finchge.operators.crossover import OnePointCrossover
from finchge.operators.mutation import IntFlipMutation
from finchge.operators.replacement import GenerationalReplacement
from finchge.operators.selection import TournamentSelection
from finchge.utils.cache import CacheManager
from finchge.utils.checkpoint import CheckpointManager, stable_config_hash
from finchge.utils.logger import BaseLogger, ExperimentLogger, get_logger, setup_logging
from finchge.utils.random_mixin import RandomStateMixin
from finchge.utils.results import ResultHelper, StatsHelper


class GrammaticalEvolution(RandomStateMixin):
    """
    Grammatical evolution class for running the evolution. This is the Evolution controller class responsible for runing
    FinchGE utils. This class is responsible for wiring up different components. This includes loading experiment configurations,
    Loading grammar, initializing initial population, setting up fitness evaluator, and using gentic algorithm to
    run the evolution.  GrammaticalEvolution class is also responsible for running the evolution loop.

    Args:
        fitness_evaluator (FitnessEvaluator):
            Evaluator to evaluate the fitness of individuals.
        grammar (Optional[Grammar]):
            BNF Grammar to be used. If not provided config must be available and must contain grammar_file value
        config (Any | None = None):
            Configuration settings for the GE algorithm.
        initialiser (Optional[GEInitialiser]) :
            Initializer class either integer based or tree based initializer
        algorithm (BaseAlgorithm):
            Evolutionary algorithm to be used e.g., GA, NSGA.

    Note:
    If agorithm is not provided, GA will be used (provided that config is available)

    """

    def __init__(
        self,
        fitness_evaluator: FitnessEvaluator,
        grammar: Optional[Grammar] = None,
        config: Optional[FinchConfig | dict[str, Any]] = None,
        initialiser: GEInitialiser | GETreeInitialiser | None = None,
        algorithm: Optional[Any] = None,
        expt_logger: BaseLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        random_state: Optional[int] = None,
    ):
        # if config is missing in the constructor, load from file
        # If is provided in the constructor, just use that
        if config is None:
            self.config = FinchConfig.from_file()
        elif isinstance(config, FinchConfig):
            self.config = config
        else:
            self.config = FinchConfig.from_dict(
                config
            )  # internally FinchConfig is used for convenience

        # Get seed from config
        config_seed = self.config.experiment.get(Keys.RANDOM_SEED)
        # Priority: parameter > config
        final_random_state = random_state if random_state is not None else config_seed
        # Call super with final seed
        super().__init__(random_state=final_random_state)

        # Setup logging
        log_dir = setup_logging(verbose=self.config.experiment.get("verbose", True))
        self.logger = get_logger()

        # Log seed information
        seed_info = self.get_seed_info()
        if seed_info["effective_seed"] is not None:
            self.logger.info(f"Run seed: {seed_info['effective_seed']}")
            self.logger.info(
                f"Seed source: {self._get_seed_source(random_state, config_seed)}"
            )
        else:
            self.logger.warning("No random seed set - results will not be reproducible")

        self.logger.info("GE Parameters: \n" + self.config.to_json())

        # halt signal with minimum generations allowed
        self._halt = False
        self._halt_min_gen = 0

        # Grammar setup
        if not grammar:
            no_grammar_error = (
                "Grammar Undefined. "
                "Please supply the grammar by either passing the 'Grammar' instance as the 'grammar' argument during class instantiation,"
                " or defining the file path using the 'grammar_file' key within the configuration settings."
            )
            # no grammar , no config
            if not self.config:
                raise ValueError(no_grammar_error)

            # has config and see if grammar is mentioned in config
            conf_grammar_file = self.config.ge.get(Keys.GRAMMAR_FILE)
            if conf_grammar_file:
                self.grammar: Grammar = Grammar.from_file(filename=conf_grammar_file)
                if self.grammar:
                    self.logger.info(self.grammar.describe())
            else:
                raise ValueError(no_grammar_error)
        else:
            self.grammar = grammar

        # fitness evaluator
        self.fitness_evaluator = fitness_evaluator

        # use inititalizer passed by Grammatical Evolution or make one if can be created from config.
        # initialiser passed through GrammaticalEvolution should always get priority and config will be ignored..
        # if neither of them are provided make_initialiser takes empty dict
        # and returns default initialiser(random_genome) with default parameters.

        if initialiser is not None:
            warnings.warn(
                "Initializer provided explicitly; "
                "initialisation settings in config, if any, will be ignored.",
                UserWarning,
            )
            self.initialiser: GEInitialiser | GETreeInitialiser = initialiser
        else:
            self.logger.info(
                f"Making Initializer from config with initialisation type: {self.config.ge[Keys.INIT_TYPE]}"
            )
            self.initialiser = make_initialiser(self.config)

            if isinstance(self.initialiser, GETreeInitialiser):
                self.tree_generator = TreeGenerator(
                    grammar=self.grammar,
                    max_tree_depth=self.config.ge[Keys.MAX_TREE_DEPTH],
                )
                self.initialiser.set_tree_generator(self.tree_generator)

            if hasattr(self.initialiser, "set_mapper"):
                self.initialiser.set_mapper(self.fitness_evaluator.mapper)

        self.objective_names = self.fitness_evaluator.get_objective_names()
        self.multi_obj = self.fitness_evaluator.is_multi_objective()

        if algorithm is not None:
            self.algorithm = algorithm
            self.logger.info(f"GE Using {algorithm.__class__.__name__}")
        else:
            # If algorithm is not defined ,
            # GeneticAlgorithm is used by default,
            # But it requires GE parameters to be stored in config.yaml
            if not self.config:
                raise ValueError("Algorithim not specified.")

            self.logger.info(
                "Default GeneticAlgorithm setup: Selection = TournamentSelection, "
                "Crossover = OnePointCrossover, Mutation = IntflipMutation, "
                "Replacement = GenerationalReplacement. To customize, pass an 'algorithm' argument."
            )

            # Default algorithm is single-objective
            max_best = self.fitness_evaluator.fitness_functions[0].maximize

            self.algorithm = GeneticAlgorithm(
                selection=TournamentSelection(max_best=max_best),
                crossover=OnePointCrossover(
                    codon_size=self.config.ge[Keys.CODON_SIZE],
                    crossover_proba=self.config.ge[Keys.CROSSOVER_PROBABILITY],
                    random_state=random_state,
                ),
                mutation=IntFlipMutation(
                    self.config.ge[Keys.MUTATION_PROBABILITY],
                    codon_size=self.config.ge[Keys.CODON_SIZE],
                    random_state=random_state,
                ),
                replacement=GenerationalReplacement(max_best=max_best),
                elite_size=self.config.ge[Keys.ELITE_SIZE],
                fitness_evaluator=fitness_evaluator,
                random_state=random_state,
            )

        # experiment logging and checkpoint
        # If custom experiment logging is not passed in the constructor,
        # if enabled in config we use default Experiment Logger
        self.expt_logger = expt_logger
        if not self.expt_logger:
            expt_config = self.config.experiment.get(Keys.EXPT_LOGGER_ENABLED, False)
            exclude_log_config = self.config.experiment.get(Keys.EXCLUDE_LOGS, [])
            if expt_config:  # logger is configured
                self.expt_logger = ExperimentLogger(exclude=exclude_log_config)
            else:  # logger not configured
                if exclude_log_config:  # but exclude dirs setup
                    logging.warning(
                        "[GrammaticalEvolutions]: exclude_logs config ignored. ExperimentLogger is not enabled"
                    )

        # if available setup the callbacks
        if self.expt_logger:
            self.expt_logger.on_run_start(
                log_dir, self.objective_names, self.config.to_dict()
            )

        self.checkpoint_manager = checkpoint_manager

        # cache
        self.cache_manager: CacheManager[Any] = CacheManager.from_config(self.config)
        self.fitness_evaluator.set_cache_manager(self.cache_manager)

        # tail related settings during reverse map
        # TODO Needs better way of handling this later
        self.fitness_evaluator._configured_genome_length = self.config.ge[
            Keys.GENOME_LENGTH
        ]
        self.fitness_evaluator._configured_codon_size = self.config.ge[Keys.CODON_SIZE]

        self.result_helper = ResultHelper()

        self._inject_rng()  # inject rng into all components that use randomness

    def initialize_population(self) -> Population:
        # Initialize population
        population = Population(
            population_size=self.config.ge[Keys.POPULATION_SIZE],
            initialiser=self.initialiser,
        )
        return population

    def run(self) -> GEResult:
        """
        Finds the fittest individual ( or a pareto front for multi-objective) in the population
        and logs the results.

        """
        # check algorithm - fitness compatibility
        validate_algorithm_fitness_match(
            algorithm=self.algorithm, fitness_evaluator=self.fitness_evaluator
        )

        start = timeit.default_timer()
        start_generation = 0  # generation where to start evolution (can be non-zero if started from saved checkpoint)
        population = self.initialize_population()

        # Initial evaluation
        self.fitness_evaluator.evaluate_population(population)
        self.algorithm.sort_population(
            population
        )  # Not used by NSGA3 but won't be any problem calling this

        # Logging Generation 0
        initial_fitness_stats: list[dict[str, Any]] = StatsHelper.compute_fitness_stats(
            individuals=population.individuals
        )
        # Logging initial population info
        if not self.multi_obj:
            initial_best = self.algorithm.get_best_individual(population)
            if self.expt_logger:
                self.expt_logger.on_generation_end(
                    generation=0,
                    population=population,
                    best=initial_best,
                    fitness_stats=initial_fitness_stats,
                )
        else:
            # multi objective
            initial_front = self.algorithm.get_pareto_front(population)

            if self.expt_logger:
                self.expt_logger.on_generation_end(
                    generation=0,
                    population=population,
                    pareto_front=initial_front,
                    fitness_stats=initial_fitness_stats,
                )

        # Resume if checkpoint exists
        if self.checkpoint_manager and self.checkpoint_manager.exists():
            expected_hash = stable_config_hash(self.config.to_dict())
            state = self.checkpoint_manager.load_latest(
                expected_config_hash=expected_hash
            )

            # restore rng
            self._rng.setstate(state.rng_state.py_state)
            self.np_rng.set_state(state.rng_state.np_state)

            start_generation = state.generation  # checkpoint
            population = state.population
            algorithm = state.algorithm

            if hasattr(algorithm, "selection") and getattr(
                algorithm.selection, "requires_case_data", False
            ):
                for ind in population.individuals:
                    if ind.fitness and not ind.has_meta(Individual.CASE_DATA_META_KEY):
                        raise RuntimeError(
                            "Checkpoint contains evaluated individuals without required casewise metadata "
                            "for lexicase selection."
                        )

            self.algorithm = algorithm

            self._inject_rng()  # share it with all components

            self.logger.info(
                f"Resuming from checkpoint at generation {start_generation}"
            )

        # handle single-objective and multi-objective
        if not self.multi_obj:
            # Run evolution
            fittest_individual, population = self.__find_best_individual(
                start_generation, population
            )
            # Round fitness to 4 decimal values
            rounded_fitness = [round(val, 4) for val in fittest_individual.fitness]
            self.logger.info(
                f"Best Phenotype (Fitness: {rounded_fitness}): {fittest_individual.phenotype}"
            )
            # If the logger is set up log run_end.
            if self.expt_logger:
                self.expt_logger.on_run_end(
                    GEResult(
                        best_individual=fittest_individual,
                        population=population,
                        pareto_front=None,  # NA: for MO only
                    )
                )
        else:
            pareto_front, population = self.__find_best_front(
                start_generation, population
            )

            if self.expt_logger:
                self.expt_logger.on_run_end(
                    GEResult(
                        pareto_front=pareto_front,
                        population=population,
                        best_individual=None,  # NA: for single objective only
                    )
                )
        # Save Summary and plots
        self.result_helper.generate_summary(self.objective_names)

        stop = timeit.default_timer()
        self.logger.info(f"Total time taken: {stop - start :.4f} seconds")

        return GEResult(
            best_individual=fittest_individual if not self.multi_obj else None,
            pareto_front=pareto_front if self.multi_obj else None,
            population=population,
        )

    def __find_best_individual(
        self, start_generation: int, population: Population
    ) -> tuple[Individual, Population]:
        fittest: Individual = self.algorithm.get_best_individual(population)

        generation_progress = tqdm(
            range(
                start_generation + 1, (self.config.experiment[Keys.NUM_GENERATIONS] + 1)
            )
        )

        for generation in generation_progress:
            # If halt signal is sent stop after min generations allowed
            # Created for helping automated testing for checkpoints,
            # but can have other applications too
            if self._halt and generation > self._halt_min_gen:
                self.logger.info(f"Halting at generation {generation}")
                break

            population = self.algorithm.evolve_one_generation(population)
            fitness_stats: list[dict[str, Any]] = StatsHelper.compute_fitness_stats(
                individuals=population.individuals
            )
            fittest = self.algorithm.get_best_individual(population)

            # Log experiment
            if self.expt_logger:
                self.expt_logger.on_generation_end(
                    generation=generation,
                    population=population,
                    best=fittest,
                    fitness_stats=fitness_stats,
                )

            # Save checkpoint if applies to this generation
            self._checkpoint_generation(generation, population)

        # Clear cache
        self.cache_manager.clear()
        return fittest, population

    def __find_best_front(
        self, start_generation: int, population: Population
    ) -> tuple[list[Individual], Population]:
        # Create log directories if they are not excluded
        # exclude_log_config = self.config.experiment.get("exclude_log", [])

        best_front: list[Individual] = self.algorithm.get_pareto_front(population)

        # Evolution Loop
        generation_progress = tqdm(
            range(
                start_generation + 1, (self.config.experiment[Keys.NUM_GENERATIONS] + 1)
            )
        )
        for generation in generation_progress:
            population = self.algorithm.evolve_one_generation(population)
            best_front = self.algorithm.get_pareto_front(population)
            # Update progress tqdm
            avg_fitness = np.mean([ind.fitness for ind in best_front], axis=0)
            generation_progress.set_description(
                f"Generation: {generation} Avg Front Fitness: {avg_fitness}"
            )

            # Log experiment
            if self.expt_logger:
                self.expt_logger.on_generation_end(
                    generation=generation,
                    pareto_front=best_front,
                    population=population,
                )

            # Save checkpoint if applies to this generation
            self._checkpoint_generation(generation, population)

        # Clear cache before returning
        self.cache_manager.clear()
        return best_front, population

    def _checkpoint_generation(self, generation: int, population: Population) -> None:
        # If checkpointing is enablled and current generation needs checkpointing
        if self.checkpoint_manager and self.checkpoint_manager.should_save(generation):
            self.checkpoint_manager.save(
                generation=generation,
                population=population,
                algorithm=self.algorithm,
                config=self.config.to_dict(),
                py_rng_state=self._rng.getstate(),
                np_rng_state=self._np_rng.get_state(),
            )

    def halt(self, min_gens_allowed: int = 0) -> None:
        """
        halt signal to stop generation loop : can specify minimum allowed generations,
        if halt signal received it will wait if min_generations are not satisfied yet,
        if halt signal is received after minimum generations allowed, will halt immediately

        Args:
            min_gens_allowed: Minimum number of generations before halt
        """
        self._halt_min_gen = min_gens_allowed
        self._halt = True

    def _get_seed_source(self, param_seed: Any | None, config_seed: Any | None) -> str:
        # for logging purpose. to know where the seed is coming from config or constructor
        if param_seed is not None:
            return "constructor parameter"
        elif config_seed is not None:
            return "config file"
        return "none"

    def _inject_rng(self) -> None:
        for component in [
            self.fitness_evaluator.mapper,
            self.grammar,
            self.initialiser,
            self.algorithm,
        ]:
            if hasattr(component, "_rng"):
                component._rng = self._rng
                component._np_rng = self._np_rng

        # also inject to operators
        self.algorithm.inject_operator_rng()
