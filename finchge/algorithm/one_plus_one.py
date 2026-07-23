from typing import Optional

from finchge.algorithm.base import BaseAlgorithmSO
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.operators.base import GEMutationStrategy


class OnePlusOneES(BaseAlgorithmSO):
    """
    (1+1) Evolution Strategy for single-objective GE.

    The simplest Evolution Strategy: a single parent produces a single offspring
    each generation via mutation. The better of the two survives (greedy accept).
    If the offspring is invalid or has no usable fitness, the parent is retained.

    This is a special case of (mu+lambda)-ES with mu=1 and lambda=1. Requires
    population_size=1 in the run configuration.

    Args:
        mutation: Mutation operator applied to produce the offspring.
        fitness_evaluator: Evaluator for phenotype mapping and fitness.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        mutation: GEMutationStrategy,
        fitness_evaluator: FitnessEvaluator,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.mutation = mutation
        self.fitness_evaluator = fitness_evaluator
        self.max_best = fitness_evaluator.get_maximize_flags()[0]

    def _get_operators(self) -> list:
        return [self.mutation, self.fitness_evaluator]

    def evolve_one_generation(self, population: Population) -> Population:
        if population.population_size != 1:
            raise ValueError(
                f"OnePlusOneES requires population_size=1, "
                f"got {population.population_size}. Set population_size: 1 in ge_config.yaml."
            )

        parent = population.individuals[0]

        offspring = self.mutation.mutate(parent)
        self.fitness_evaluator.refresh_mapping_all([offspring])

        offspring_pop = Population.from_individuals([offspring], population_size=1)
        self.fitness_evaluator.evaluate_population(offspring_pop)

        survivor = offspring if self._offspring_wins(parent, offspring) else parent

        next_pop = Population.from_individuals([survivor], population_size=1)
        self.sort_population(next_pop)
        return next_pop

    def _offspring_wins(self, parent: Individual, offspring: Individual) -> bool:
        if not offspring.has_usable_fitness():
            return False
        if not parent.has_usable_fitness():
            return True
        if self.max_best:
            return offspring.fitness[0] > parent.fitness[0]
        return offspring.fitness[0] < parent.fitness[0]
