from typing import Optional

from finchge.algorithm.base import BaseAlgorithm
from finchge.algorithm.utils import (
    calculate_crowding_distance,
    environmental_selection_nsga3,
    fast_non_dominated_sort,
    generate_reference_points,
)
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.operators.base import (
    GECrossoverStrategy,
    GEMutationStrategy,
    GEReplacementStrategy,
    GESelectionStrategy,
)


class NSGA2(BaseAlgorithm):
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
        """
        NSGA2  class.

        Args:
            selection: Selection strategy instance or function.
            crossover: Crossover strategy instance or function.
            mutation: Mutation strategy instance or function.
            replacement: Replacement strategy instance or function.
            elite_size (int): Number of elite individuals to carry over.
            random_state (int) : random state
        """
        super().__init__(random_state=random_state)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.elite_size = elite_size
        self.fitness_evaluator = fitness_evaluator

        self.inject_operator_rng()

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

        self.fitness_evaluator.evaluate_population(new_population)
        self.sort_population(new_population)
        return new_population

    def sort_population(self, population: Population) -> None:
        """
        Sort population using NSGA-II criteria

        Args:
            population (Population): The population to sort.

        Returns:
            Population: The sorted population.
        """
        maximize_flags = self.fitness_evaluator.get_maximize_flags()
        fronts = fast_non_dominated_sort(population.individuals, maximize_flags)
        # Calculate crowding distance for each front
        for front in fronts:
            calculate_crowding_distance(front)
        # Sort population based on rank and crowding distance
        population.individuals.sort(
            key=lambda ind: (
                ind.get_meta("rank", int),
                -ind.get_meta("crowding_distance", float),
            )
        )

    def get_pareto_front(self, population: Population) -> list[Individual]:
        """Return the Pareto front (rank 0 individuals)."""
        fronts = fast_non_dominated_sort(
            population.individuals,
            self.fitness_evaluator.get_maximize_flags(),
        )
        return fronts[0] if fronts else []

    def get_best_individual(self, population: Population) -> Individual:
        raise NotImplementedError(
            "Best individual is not defined for multi-objective algorithms"
        )

    def inject_operator_rng(self) -> None:
        operators = [
            self.selection,
            self.crossover,
            self.mutation,
            self.replacement,
            self.fitness_evaluator,
        ]

        for op in operators:
            if hasattr(op, "_rng"):
                op._rng = self._rng  # Share same RNG object
            if hasattr(op, "_np_rng") and hasattr(self, "_np_rng"):
                op._np_rng = self._np_rng


class NSGA3(BaseAlgorithm):
    def __init__(
        self,
        selection: GESelectionStrategy,
        crossover: GECrossoverStrategy,
        mutation: GEMutationStrategy,
        fitness_evaluator: FitnessEvaluator,
        num_divisions: int = 12,
        random_state: Optional[int] = None,
    ) -> None:
        """
        NSGA-III class


        Args:
            selection (GESelectionStrategy: Selection strategy instance or function.
            crossover (GECrossoverStrategy): Crossover strategy instance or function.
            mutation (GEMutationStrategy): Mutation strategy instance or function.
            fitness_evaluator (FitnessEvaluator): Fitness Evaluator for performing Evaluation
            num_divisions (int): Number of divisions for reference points.
            random_state (Optional[int]) : Random state
        """
        super().__init__(random_state=random_state)
        self.num_divisions = num_divisions
        self.epsilon = 1e-12

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.fitness_evaluator = fitness_evaluator

        self.inject_operator_rng()

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

        # Selection
        selected_individuals = self.selection.select(
            population.population_size, selectable_individuals
        )

        # CROSSOVER
        offsprings = self.apply_crossover(
            crossover_strategy=self.crossover,
            selected_individuals=selected_individuals,
            new_pop_size=population.population_size,
        )

        # Mutation
        offspring_population = self.apply_mutation(
            mutation_strategy=self.mutation, individuals=offsprings
        )

        self.fitness_evaluator.evaluate_population(offspring_population)

        # Replacement NSGA-III environmental selection from combined population
        combined: list[Individual] = (
            population.individuals + offspring_population.individuals
        )
        maximize_flags = self.fitness_evaluator.get_maximize_flags()

        # reference points exist
        num_objectives = len(combined[0].fitness)  # assumes list[float]
        reference_points = generate_reference_points(num_objectives, self.num_divisions)

        for ind in combined:
            ind.meta.clear()

        # just for calculating crowding distance once.
        fast_non_dominated_sort(combined, maximize_flags)

        next_inds = environmental_selection_nsga3(
            combined,
            population_size=population.population_size,
            reference_points=reference_points,
            maximize_flags=maximize_flags,
            epsilon=self.epsilon,
            rng=self.rng,
        )

        # Create next population object
        next_pop = Population.from_individuals(
            population_size=population.population_size,
            individuals=next_inds,
        )

        next_pop.individuals = next_inds
        return next_pop

    def sort_population(self, population: Population) -> None:
        pass  # Skipped in NSGA3

    def get_best_individual(self, population: Population) -> Individual:
        raise NotImplementedError(
            "Best individual is not defined for multi-objective algorithms"
        )

    def get_pareto_front(self, population: Population) -> list[Individual]:
        """
        Return the Pareto front (individuals with rank 0)
        Args:
            population (Population): The population to extract the Pareto front from.
        Returns:
            list[Individual]: List of individuals in the Pareto front.

        """
        fronts = fast_non_dominated_sort(
            population.individuals,
            self.fitness_evaluator.get_maximize_flags(),
        )

        return fronts[0] if fronts else []

    def inject_operator_rng(self) -> None:
        operators = [
            self.selection,
            self.crossover,
            self.mutation,
            self.fitness_evaluator,
        ]

        for op in operators:
            if hasattr(op, "_rng"):
                op._rng = self._rng  # Share same RNG object
            if hasattr(op, "_np_rng") and hasattr(self, "_np_rng"):
                op._np_rng = self._np_rng
