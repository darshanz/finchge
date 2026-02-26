from typing import Optional

from finchge.algorithm.base import BaseAlgorithm
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.operators.base import (
    GECrossoverStrategy,
    GEMutationStrategy,
    GEReplacementStrategy,
    GESelectionStrategy,
)


class GeneticAlgorithm(BaseAlgorithm):
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

    def evolve_one_generation(self, population: Population) -> Population:
        """
        Perform one generation of evolution on the given population.

        Args:
            population (Population): The population to evolve.

        Returns:
            Population: The evolved population.
        """
        valid_individuals = [ind for ind in population.individuals if not ind.invalid]
        if len(valid_individuals) < 2:
            raise Exception(
                f"Not enough valid individuals. Valid count: {len(valid_individuals)}"
            )

        # Selection
        selected_individuals = self.selection.select(
            population_size=population.population_size, individuals=valid_individuals
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

    def get_best_individual(self, population: Population) -> Individual:
        fitness_functions = self.fitness_evaluator.get_fitness_functions()
        if len(fitness_functions) != 1:
            raise ValueError(
                "GeneticAlgorithm only supports single-objective optimization. "
                f"Provided {len(fitness_functions)} fitness functions."
            )

        if fitness_functions[0].maximize:
            return max(population.individuals, key=lambda ind: ind.fitness[0])
        return min(population.individuals, key=lambda ind: ind.fitness[0])

    def get_pareto_front(self, population: Population) -> list[Individual]:
        raise NotImplementedError("Pareto front is not defined for single-objective GA")

    def sort_population(self, population: Population) -> None:
        """
        Sorts the population based on fitness values.
        Supports the fitness_evaluator with single fitness function.

        Args:
            population (Population): The population to sort.
        """
        fitness_functions = self.fitness_evaluator.get_fitness_functions()
        if not isinstance(fitness_functions, list):
            fitness_functions = [fitness_functions]

        if len(fitness_functions) != 1:
            raise ValueError(
                "GeneticAlgorithm only supports single-objective optimization. "
                f"Provided {len(fitness_functions)} fitness functions."
            )

        fitness_fn = fitness_functions[0]

        reverse = fitness_fn.maximize if hasattr(fitness_fn, "maximize") else False
        population.individuals.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else float("-inf"),
            reverse=reverse,
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
