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


class IslandGA(BaseAlgorithmSO):
    """
    Island Model Genetic Algorithm for single-objective GE.

    The population is partitioned into num_islands independent sub-populations
    (islands), each evolving in isolation using standard GA operators. Every
    migration_interval generations, the best migration_size individuals from
    each island migrate to the next island in a ring topology, replacing that
    island's worst individuals.

    Island isolation promotes diversity by allowing different islands to explore
    different regions of the search space. Migration periodically shares good
    solutions to prevent any island from stagnating in a local optimum.

    Requires population_size to be divisible by num_islands.

    Args:
        num_islands: Number of independent sub-populations. Must be >= 2.
        migration_interval: Generations between migration events.
        migration_size: Number of individuals exchanged per migration event.
            Must be < population_size // num_islands.
        selection: Selection strategy for within-island evolution.
        crossover: Crossover strategy for within-island evolution.
        mutation: Mutation strategy for within-island evolution.
        replacement: Replacement strategy for within-island evolution.
        elite_size: Number of elites preserved per island per generation.
        fitness_evaluator: Evaluator for phenotype mapping and fitness.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        num_islands: int,
        migration_interval: int,
        migration_size: int,
        selection: GESelectionStrategy,
        crossover: GECrossoverStrategy,
        mutation: GEMutationStrategy,
        replacement: GEReplacementStrategy,
        elite_size: int,
        fitness_evaluator: FitnessEvaluator,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        if num_islands < 2:
            raise ValueError("num_islands must be >= 2")
        if migration_size < 1:
            raise ValueError("migration_size must be >= 1")
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.elite_size = elite_size
        self.fitness_evaluator = fitness_evaluator
        self.max_best = fitness_evaluator.get_maximize_flags()[0]
        self._islands: Optional[list[Population]] = None
        self._generation: int = 0

    def _get_operators(self) -> list[Any]:
        return [
            self.selection,
            self.crossover,
            self.mutation,
            self.replacement,
            self.fitness_evaluator,
        ]

    def evolve_one_generation(self, population: Population) -> Population:
        if self._islands is None:
            self._islands = self._partition(population)

        for i in range(self.num_islands):
            self._islands[i] = self._evolve_island(self._islands[i])

        self._generation += 1

        if self._generation % self.migration_interval == 0:
            self._migrate()

        return self._merge(population.population_size)

    def _partition(self, population: Population) -> list[Population]:
        n = population.population_size
        if n % self.num_islands != 0:
            raise ValueError(
                f"population_size ({n}) must be divisible by num_islands ({self.num_islands})."
            )
        island_size = n // self.num_islands
        if self.migration_size >= island_size:
            raise ValueError(
                f"migration_size ({self.migration_size}) must be < island_size ({island_size})."
            )
        inds = population.individuals[:]
        return [
            Population.from_individuals(
                inds[i * island_size : (i + 1) * island_size],
                population_size=island_size,
            )
            for i in range(self.num_islands)
        ]

    def _evolve_island(self, island: Population) -> Population:
        selectable = self._get_selectable_individuals(island)
        if len(selectable) < 2:
            return island

        selected = self.selection.select(
            population_size=island.population_size,
            individuals=selectable,
        )
        offspring = self.apply_crossover(
            crossover_strategy=self.crossover,
            selected_individuals=selected,
            new_pop_size=island.population_size,
        )
        self.fitness_evaluator.refresh_mapping_all(offspring)
        offspring_pop = self.apply_mutation(
            mutation_strategy=self.mutation,
            individuals=offspring,
        )
        self.fitness_evaluator.evaluate_population(offspring_pop)
        self.sort_population(offspring_pop)

        new_inds = self.replacement.replace(
            new_population=offspring_pop.individuals,
            old_population=island.individuals,
            elite_size=self.elite_size,
            population_size=island.population_size,
        )
        new_island = Population.from_individuals(
            new_inds, population_size=island.population_size
        )
        self.sort_population(new_island)
        return new_island

    def _migrate(self) -> None:
        # Ring topology: best migration_size individuals from island i
        # replace worst migration_size individuals in island (i+1) % num_islands
        assert self._islands is not None
        emigrants: list[list[Individual]] = []
        for island in self._islands:
            ranked = sorted(
                island.individuals,
                key=lambda ind: ind.sort_key(self.max_best),
                reverse=self.max_best,
            )
            emigrants.append(ranked[: self.migration_size])

        for i, island in enumerate(self._islands):
            ranked = sorted(
                island.individuals,
                key=lambda ind: ind.sort_key(self.max_best),
                reverse=self.max_best,
            )
            survivors = ranked[: island.population_size - self.migration_size]
            arrivals = emigrants[(i - 1) % self.num_islands]
            self._islands[i] = Population.from_individuals(
                survivors + arrivals,
                population_size=island.population_size,
            )

    def _merge(self, total_size: int) -> Population:
        assert self._islands is not None
        all_inds: list[Individual] = []
        for island in self._islands:
            all_inds.extend(island.individuals)
        merged = Population.from_individuals(all_inds, population_size=total_size)
        self.sort_population(merged)
        return merged
