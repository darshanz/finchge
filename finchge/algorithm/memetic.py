from typing import Optional

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


class MemeticGA(BaseAlgorithmSO):
    """
    Memetic Algorithm for single-objective GE.

    Combines a standard GA with a local search phase applied to each offspring
    after mutation. The GA provides global exploration (crossover + mutation);
    local search provides local exploitation by hill-climbing each offspring for
    a fixed number of steps before replacement.

    Local search applies the mutation operator repeatedly to each offspring and
    keeps the result if it improves fitness. A probability parameter controls
    what fraction of offspring undergo local search each generation, allowing
    the compute budget to be traded against solution quality.

    Args:
        selection: Selection strategy for parent selection.
        crossover: Crossover strategy.
        mutation: Mutation strategy used for both GA mutation and local search.
        replacement: Replacement strategy.
        elite_size: Number of elites preserved per generation.
        fitness_evaluator: Evaluator for phenotype mapping and fitness.
        local_search_steps: Number of mutation trials per individual in local search.
        local_search_probability: Fraction of offspring that undergo local search.
            1.0 applies local search to every offspring; 0.0 disables it entirely.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        selection: GESelectionStrategy,
        crossover: GECrossoverStrategy,
        mutation: GEMutationStrategy,
        replacement: GEReplacementStrategy,
        elite_size: int,
        fitness_evaluator: FitnessEvaluator,
        local_search_steps: int = 5,
        local_search_probability: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        if local_search_steps < 1:
            raise ValueError("local_search_steps must be >= 1")
        if not 0.0 <= local_search_probability <= 1.0:
            raise ValueError("local_search_probability must be in [0.0, 1.0]")
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement

        self.elite_size = elite_size
        self.fitness_evaluator = fitness_evaluator
        self.local_search_steps = local_search_steps

        self.local_search_probability = local_search_probability
        self.max_best = fitness_evaluator.get_maximize_flags()[0]

    def _get_operators(self) -> list:
        return [
            self.selection,
            self.crossover,
            self.mutation,
            self.replacement,
            self.fitness_evaluator,
        ]

    def evolve_one_generation(self, population: Population) -> Population:
        selectable = self._get_selectable_individuals(population)
        if len(selectable) < 2:
            raise ValueError(
                f"Not enough valid individuals for selection. Got {len(selectable)}."
            )

        selected = self.selection.select(
            population_size=population.population_size,
            individuals=selectable,
        )
        offspring = self.apply_crossover(
            crossover_strategy=self.crossover,
            selected_individuals=selected,
            new_pop_size=population.population_size,
        )
        self.fitness_evaluator.refresh_mapping_all(offspring)
        offspring_pop = self.apply_mutation(
            mutation_strategy=self.mutation,
            individuals=offspring,
        )
        self.fitness_evaluator.evaluate_population(offspring_pop)

        # Local search: hill-climb each offspring that is selected for refinement
        refined = [
            self._local_search(ind)
            if self.rng.random() < self.local_search_probability
            else ind
            for ind in offspring_pop.individuals
        ]

        offspring_pop = Population.from_individuals(
            refined, population_size=len(refined)
        )
        self.sort_population(offspring_pop)

        new_inds = self.replacement.replace(
            new_population=offspring_pop.individuals,
            old_population=population.individuals,
            elite_size=self.elite_size,
            population_size=population.population_size,
        )
        new_population = Population.from_individuals(
            new_inds, population_size=population.population_size
        )
        self.sort_population(new_population)
        return new_population

    def _local_search(self, ind: Individual) -> Individual:
        current = ind
        for _ in range(self.local_search_steps):
            candidate = self.mutation.mutate(current)
            self.fitness_evaluator.refresh_mapping_all([candidate])
            candidate_pop = Population.from_individuals([candidate], population_size=1)
            self.fitness_evaluator.evaluate_population(candidate_pop)
            if self._is_improvement(candidate, current):
                current = candidate
        return current

    def _is_improvement(self, candidate: Individual, incumbent: Individual) -> bool:
        if not candidate.has_usable_fitness():
            return False
        if not incumbent.has_usable_fitness():
            return True
        if self.max_best:
            return candidate.fitness[0] > incumbent.fitness[0]
        return candidate.fitness[0] < incumbent.fitness[0]
