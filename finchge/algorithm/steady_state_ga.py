from typing import Any, Optional

from finchge.algorithm.base import BaseAlgorithmSO
from finchge.core.population import Population
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.operators.base import (
    GECrossoverStrategy,
    GEMutationStrategy,
    GESelectionStrategy,
)


class SteadyStateGA(BaseAlgorithmSO):
    """
    Steady-State Genetic Algorithm for single-objective GE.

    Implements the Genitor model (Whitley, 1989): each iteration selects two
    parents, produces two offspring via crossover and mutation, evaluates them,
    then replaces the two worst individuals in the current population. This
    repeats until population_size offspring have been processed.

    Unlike generational GAs, offspring enter the live population immediately.
    Newly inserted individuals are eligible for both parent selection and
    replacement in subsequent iterations of the same generation. No explicit
    elitism is needed: the best individuals survive naturally because only the
    worst are displaced each iteration.

    Args:
        selection: Parent selection strategy.
        crossover: Crossover operator producing two offspring per call.
        mutation: Mutation operator applied to each offspring.
        fitness_evaluator: Evaluator for phenotype mapping and fitness scoring.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        selection: GESelectionStrategy,
        crossover: GECrossoverStrategy,
        mutation: GEMutationStrategy,
        fitness_evaluator: FitnessEvaluator,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.fitness_evaluator = fitness_evaluator
        self.max_best = fitness_evaluator.get_maximize_flags()[0]

    def _validate_population_size(self, pop_size: int) -> None:
        if pop_size < 4:
            raise ValueError(
                f"SteadyStateGA requires population_size >= 4 "
                f"(need at least 2 survivors after replacing the 2 worst), got {pop_size}."
            )

    def _get_operators(self) -> list[Any]:
        return [self.selection, self.crossover, self.mutation, self.fitness_evaluator]

    def evolve_one_generation(self, population: Population) -> Population:
        pop_size = population.population_size
        self._validate_population_size(pop_size)
        individuals = population.individuals[:]
        ind_counter = 0

        while ind_counter < pop_size:
            selectable = [ind for ind in individuals if self._is_selectable(ind)]
            if len(selectable) < 2:
                break

            parents = self.selection.select(population_size=2, individuals=selectable)
            offspring1, offspring2 = self.crossover.cross(parents[0], parents[1])
            new_pair = [
                self.mutation.mutate(offspring1),
                self.mutation.mutate(offspring2),
            ]

            self.fitness_evaluator.refresh_mapping_all(new_pair)
            pair_pop = Population.from_individuals(
                new_pair, population_size=len(new_pair)
            )
            self.fitness_evaluator.evaluate_population(pair_pop)

            # Replace the two worst in the current live population
            individuals.sort(
                key=lambda ind: ind.sort_key(self.max_best), reverse=self.max_best
            )
            individuals = individuals[: pop_size - len(new_pair)] + new_pair

            ind_counter += len(new_pair)

        next_pop = Population.from_individuals(individuals, population_size=pop_size)
        self.sort_population(next_pop)
        return next_pop
