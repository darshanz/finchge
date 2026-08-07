from typing import Optional

from finchge.algorithm.base import BaseAlgorithmSO
from finchge.core.individual import Individual
from finchge.core.population import Population
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.operators.base import GEMutationStrategy


class CLONALG(BaseAlgorithmSO):
    """
    Clonal Selection Algorithm (CLONALG) for single-objective GE.

    Inspired by the biological clonal selection principle of the immune system.
    Each generation, the top num_select individuals are selected, cloned
    num_clones times each, and subjected to hypermutation. Hypermutation rate
    is proportional to fitness rank: the best individuals mutate least (to
    preserve good solutions) and the worst selected individuals mutate most
    (to explore new regions). The best clone from each group replaces its
    original in the population.

    Hypermutation is approximated by applying the mutation operator
    n = max(1, round(hyper_factor * (rank + 1))) times per clone, where
    rank 0 is the best individual. This produces a linear scaling of
    mutation intensity with rank.

    Args:
        num_select: Number of top individuals selected for cloning each generation.
        num_clones: Number of clones produced per selected individual.
        hyper_factor: Controls hypermutation intensity. Higher values increase
            the mutation rate gap between best and worst selected individuals.
        mutation: Mutation operator applied during hypermutation.
        fitness_evaluator: Evaluator for phenotype mapping and fitness.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        num_select: int,
        num_clones: int,
        hyper_factor: float,
        mutation: GEMutationStrategy,
        fitness_evaluator: FitnessEvaluator,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        if num_select < 1:
            raise ValueError("num_select must be >= 1")
        if num_clones < 1:
            raise ValueError("num_clones must be >= 1")
        if hyper_factor <= 0:
            raise ValueError("hyper_factor must be > 0")
        self.num_select = num_select
        self.num_clones = num_clones
        self.hyper_factor = hyper_factor
        self.mutation = mutation
        self.fitness_evaluator = fitness_evaluator
        self.max_best = fitness_evaluator.get_maximize_flags()[0]

    def _get_operators(self) -> list:
        return [self.mutation, self.fitness_evaluator]

    def evolve_one_generation(self, population: Population) -> Population:
        self.sort_population(population)

        selectable = self._get_selectable_individuals(population)
        if not selectable:
            raise ValueError("No valid individuals available for clonal selection.")

        # Select top num_select individuals (population is already sorted best-first)
        selected = selectable[: self.num_select]

        # Clone and hypermutate: rank 0 (best) mutates least, higher rank mutates more
        all_clones: list[Individual] = []
        for rank, ind in enumerate(selected):
            n_mutations = max(1, round(self.hyper_factor * (rank + 1)))
            for _ in range(self.num_clones):
                clone = ind
                for _ in range(n_mutations):
                    clone = self.mutation.mutate(clone)
                all_clones.append(clone)

        self.fitness_evaluator.refresh_mapping_all(all_clones)
        clone_pop = Population.from_individuals(
            all_clones, population_size=len(all_clones)
        )
        self.fitness_evaluator.evaluate_population(clone_pop)

        # For each selected individual, keep the best of its clones vs itself
        best_clones: list[Individual] = []
        for rank, original in enumerate(selected):
            group = all_clones[rank * self.num_clones : (rank + 1) * self.num_clones]
            valid = [c for c in group if c.has_usable_fitness()]
            if valid:
                best_clone = (
                    max(valid, key=lambda c: c.fitness[0])
                    if self.max_best
                    else min(valid, key=lambda c: c.fitness[0])
                )
                # Retain original if it is at least as good as the best clone
                if original.has_usable_fitness() and not self._is_improvement(
                    best_clone, original
                ):
                    best_clones.append(original)
                else:
                    best_clones.append(best_clone)
            else:
                best_clones.append(original)

        # Build new population: best clones occupy the top slots;
        # remaining individuals from old population fill the rest
        non_selected = [ind for ind in population.individuals if ind not in selected]
        new_inds = (best_clones + non_selected)[: population.population_size]

        new_pop = Population.from_individuals(
            new_inds, population_size=population.population_size
        )
        self.sort_population(new_pop)
        return new_pop

    def _is_improvement(self, candidate: Individual, incumbent: Individual) -> bool:
        if not candidate.has_usable_fitness():
            return False
        if not incumbent.has_usable_fitness():
            return True
        if self.max_best:
            return candidate.fitness[0] > incumbent.fitness[0]
        return candidate.fitness[0] < incumbent.fitness[0]
