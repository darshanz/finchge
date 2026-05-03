from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.operators.base import GECrossoverStrategy, GEMutationStrategy
from finchge.utils.random_mixin import RandomStateMixin


class BaseAlgorithm(RandomStateMixin, ABC):
    """Abstract base class for Search Algorithm"""

    def __init__(self, random_state: Optional[Any] = None) -> None:
        super().__init__(random_state=random_state)

    @abstractmethod
    def sort_population(self, population: Population) -> None:
        ...

    @abstractmethod
    def get_best_individual(self, population: Population) -> Individual:
        ...

    @abstractmethod
    def get_pareto_front(self, population: Population) -> list[Individual]:
        ...

    @abstractmethod
    def evolve_one_generation(self, population: Population) -> Population:
        ...

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
            # offspring 1 is first added, second is added only if the population size is not met
            # just to avoid extra  individual
            # which would be removed during replacement but may affect benchmarking comparisons
            offsprings.append(offspring1)
            if len(offsprings) < new_pop_size:
                offsprings.append(offspring2)
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

    def _is_mapping_valid(self, ind: Individual) -> bool:
        return not ind.invalid

    def _has_usable_fitness(self, ind: Individual) -> bool:
        return bool(ind.fitness) and all(np.isfinite(v) for v in ind.fitness)

    def _is_selectable(self, ind: Individual) -> bool:
        return self._is_mapping_valid(ind) and self._has_usable_fitness(ind)

    def _get_selectable_individuals(self, population: Population) -> list[Individual]:
        return [ind for ind in population.individuals if self._is_selectable(ind)]
