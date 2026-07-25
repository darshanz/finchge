from typing import Optional

from finchge.algorithm.base import BaseAlgorithmSO
from finchge.core.population import Population
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.operators.base import GEMutationStrategy


class MuPlusLambdaES(BaseAlgorithmSO):
    """
    (mu+lambda) Evolution Strategy for single-objective GE.

    Each generation, lambda_ offspring are produced from the current population
    using mutation only. The next generation is selected as the best mu
    individuals from the combined pool of mu parents and lambda_ offspring.

    This is the classic (mu+lambda)-ES survivor selection rule. Unlike a
    standard GA, there is no crossover and no explicit selection operator:
    every parent is equally eligible to produce offspring, and the combined
    pool is ranked purely by fitness.

    Args:
        lambda_: Number of offspring generated per generation.
            A common starting point is lambda_ = population_size.
        mutation: Mutation operator applied to each offspring.
        fitness_evaluator: Evaluator for phenotype mapping and fitness.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        lambda_: int,
        mutation: GEMutationStrategy,
        fitness_evaluator: FitnessEvaluator,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        if lambda_ < 1:
            raise ValueError("lambda_ must be >= 1")
        self.lambda_ = lambda_
        self.mutation = mutation
        self.fitness_evaluator = fitness_evaluator
        self.max_best = fitness_evaluator.get_maximize_flags()[0]

    def _get_operators(self) -> list:
        return [self.mutation, self.fitness_evaluator]

    def evolve_one_generation(self, population: Population) -> Population:
        mu = population.population_size
        parents = self._get_selectable_individuals(population)
        if not parents:
            raise ValueError("No valid parents available for offspring generation.")

        # Each offspring is a mutated copy of a randomly chosen parent
        offspring = [
            self.mutation.mutate(self.rng.choice(parents)) for _ in range(self.lambda_)
        ]

        # Map genotype to phenotype for offspring
        self.fitness_evaluator.refresh_mapping_all(offspring)

        # Evaluate offspring fitness
        offspring_pop = Population.from_individuals(
            offspring, population_size=len(offspring)
        )
        self.fitness_evaluator.evaluate_population(offspring_pop)

        # (mu+lambda) survivor selection: rank combined pool, keep best mu
        combined = population.individuals + offspring
        combined.sort(
            key=lambda ind: ind.sort_key(self.max_best), reverse=self.max_best
        )

        next_pop = Population.from_individuals(combined[:mu], population_size=mu)
        self.sort_population(next_pop)
        return next_pop
