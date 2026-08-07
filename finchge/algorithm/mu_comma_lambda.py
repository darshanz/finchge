from typing import Optional

from finchge.algorithm.base import BaseAlgorithmSO
from finchge.core.population import Population
from finchge.fitness.fitness_evaluator import FitnessEvaluator
from finchge.operators.base import GEMutationStrategy


class MuCommaLambdaES(BaseAlgorithmSO):
    """
    (mu,lambda) Evolution Strategy for single-objective GE.

    Each generation, lambda_ offspring are produced from the current population
    using mutation only. The next generation is selected as the best mu
    individuals from the lambda_ offspring only — parents do NOT survive.

    This is the classic (mu,lambda)-ES survivor selection rule. Unlike (mu+lambda)-ES,
    parents are discarded every generation regardless of fitness. This makes the
    algorithm more explorative and able to escape local optima, at the cost of
    potentially discarding good solutions. Requires lambda_ >= mu so the next
    generation can always be filled from offspring alone.

    Args:
        lambda_: Number of offspring generated per generation. Must be >= population_size.
            A common starting point is lambda_ = 2 * population_size.
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
        if self.lambda_ < mu:
            raise ValueError(
                f"lambda_ ({self.lambda_}) must be >= mu ({mu}): "
                "offspring-only selection cannot fill the next generation."
            )

        parents = self._get_selectable_individuals(population)
        if not parents:
            raise ValueError("No valid parents available for offspring generation.")

        offspring = [
            self.mutation.mutate(self.rng.choice(parents)) for _ in range(self.lambda_)
        ]

        self.fitness_evaluator.refresh_mapping_all(offspring)

        offspring_pop = Population.from_individuals(
            offspring, population_size=len(offspring)
        )
        self.fitness_evaluator.evaluate_population(offspring_pop)

        # (mu,lambda) survivor selection: best mu from offspring ONLY — parents discarded
        offspring.sort(
            key=lambda ind: ind.sort_key(self.max_best), reverse=self.max_best
        )

        next_pop = Population.from_individuals(offspring[:mu], population_size=mu)
        self.sort_population(next_pop)
        return next_pop
