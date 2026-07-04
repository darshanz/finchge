import warnings
from typing import Any, Optional

from finchge.algorithm.base import BaseAlgorithmSO
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.operators.base import (
    GECrossoverStrategy,
    GEMutationStrategy,
    GEReplacementStrategy,
    GESelectionStrategy,
)


class GeneticAlgorithm(BaseAlgorithmSO):
    """
    Genetic Algorithm

    Args:
        selection: Selection strategy instance or function.
        crossover: Crossover strategy instance or function.
        mutation: Mutation strategy instance or function.
        replacement: Replacement strategy instance or function.
        elite_size (int): Number of elite individuals to carry over.
        random_state (Optional[int]): random state
    """

    def __init__(
        self,
        selection: GESelectionStrategy,
        crossover: GECrossoverStrategy,
        mutation: GEMutationStrategy,
        replacement: GEReplacementStrategy,
        elite_size: int,
        fitness_evaluator: FitnessEvaluator,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.elite_size = elite_size
        self.fitness_evaluator = fitness_evaluator

        self.inject_operator_rng()

        # Quick check for evaluation and selection compatibility.
        # Fitness evaluator must have require_case_data=False
        if (
            self.selection.requires_case_data
            and not self.fitness_evaluator.require_case_data
        ):
            raise ValueError(
                f"{type(self.selection).__name__} requires case data, "
                f"but FitnessEvaluator is not configured to produce it. require_case_data must be set to True for {type(self.selection).__name__} "
            )

        if (
            self.fitness_evaluator.require_case_data
            and not self.selection.requires_case_data
        ):
            warnings.warn(
                "FitnessEvaluator is configured to compute case data, but the selected "
                "selection strategy does not use it."
            )

        # Get max_best flag
        flags = fitness_evaluator.get_maximize_flags()
        if len(flags) != 1:
            raise ValueError(
                "GeneticAlgorithm only supports single-objective optimization. "
                f"Provided {len(flags)} fitness functions."
            )
        self.max_best = flags[0]

    def evolve_one_generation(self, population: Population) -> Population:
        """
        Perform one generation of evolution on the given population.

        Args:
            population (Population): The population to evolve.

        Returns:
            Population: The evolved population.
        """

        selectable_individuals = self._get_selectable_individuals(population)
        if len(selectable_individuals) < 2:
            raise Exception(
                f"Not enough valid individuals. Valid count: {len(selectable_individuals)}"
            )

        # before selection check if case data is required. if individual has case-wise data then appropriate selection
        # selection method like Lexicase Selection must be provided
        self._validate_selection_requirements(selectable_individuals)

        # Selection
        selected_individuals = self.selection.select(
            population_size=population.population_size,
            individuals=selectable_individuals,
        )

        # CROSSOVER
        offsprings = self.apply_crossover(
            crossover_strategy=self.crossover,
            selected_individuals=selected_individuals,
            new_pop_size=population.population_size,
        )

        # remap
        self.fitness_evaluator.refresh_mapping_all(offsprings)

        # Mutation
        offspring_population = self.apply_mutation(
            mutation_strategy=self.mutation, individuals=offsprings
        )

        self.fitness_evaluator.evaluate_population(offspring_population)
        self.sort_population(offspring_population)

        # Replacement
        new_individuals = self.replacement.replace(
            new_population=offspring_population.individuals,
            old_population=population.individuals,
            elite_size=self.elite_size,
            population_size=len(population),
        )

        new_population = Population.from_individuals(
            individuals=new_individuals,
            population_size=len(population),
        )

        self.sort_population(new_population)
        return new_population

    def _validate_selection_requirements(self, individuals: list[Individual]) -> None:
        if not getattr(self.selection, "requires_case_data", False):
            return

        required_keys = getattr(self.selection, "required_case_keys", ())
        for ind in individuals:
            if not ind.has_meta(Individual.CASE_DATA_META_KEY):
                raise ValueError(
                    f"{type(self.selection).__name__} requires casewise evaluation data, "
                    "but an evaluated individual is missing it."
                )
            case_store = ind.get_meta(Individual.CASE_DATA_META_KEY, dict)
            for key in required_keys:
                if key not in case_store:
                    raise ValueError(
                        f"{type(self.selection).__name__} requires case key '{key}', "
                        f"but it was not present on the individual."
                    )

    def _get_operators(self) -> list[Any]:
        return [
            self.selection,
            self.crossover,
            self.mutation,
            self.replacement,
            self.fitness_evaluator,
        ]
