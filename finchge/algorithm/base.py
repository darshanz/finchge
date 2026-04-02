from abc import ABC, abstractmethod
from typing import Any, Optional

from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.operators.base import GECrossoverStrategy, GEMutationStrategy
from finchge.utils.random_mixin import RandomStateMixin


class BaseAlgorithm(RandomStateMixin, ABC):
    """Abstract base class for Search Algorithm"""

    def __init__(self, random_state: Optional[Any] = None) -> None:
        super().__init__(random_state=random_state)

    @abstractmethod
    def sort_population(self, population: Population) -> None: ...

    @abstractmethod
    def get_best_individual(self, population: Population) -> Individual: ...

    @abstractmethod
    def get_pareto_front(self, population: Population) -> list[Individual]: ...

    @abstractmethod
    def evolve_one_generation(self, population: Population) -> Population: ...

    def apply_crossover(
        self,
        crossover_strategy: GECrossoverStrategy,
        selected_individuals: list[Individual],
        new_pop_size: int,
    ) -> list[Individual]:
        offsprings: list[Individual] = []
        # create offsprings (population.population_size : total required population)
        while len(offsprings) < new_pop_size:
            i = self.rng.randrange(len(selected_individuals))
            j = self.rng.randrange(len(selected_individuals))

            offspring1, offspring2 = crossover_strategy.cross(
                selected_individuals[i],
                selected_individuals[j],
            )
            offsprings.extend([offspring1, offspring2])
        return offsprings

    def apply_mutation(
        self, mutation_strategy: GEMutationStrategy, individuals: list[Individual]
    ) -> Population:
        mutated_offsprings = [mutation_strategy.mutate(ind) for ind in individuals]

        # Create new population from offspring
        offspring_population = Population.from_individuals(
            individuals=mutated_offsprings,
            population_size=len(individuals),
        )
        return offspring_population
