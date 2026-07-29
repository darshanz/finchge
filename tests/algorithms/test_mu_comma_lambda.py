import pytest

from finchge.algorithm.mu_comma_lambda import MuCommaLambdaES


class _FakeEvaluator:
    def get_maximize_flags(self):
        return [False]

    def refresh_mapping_all(self, individuals):
        pass

    def evaluate_population(self, population):
        for ind in population.individuals:
            if ind.fitness is None:
                ind.fitness = [0.0]


def _make_algorithm(lambda_: int = 20) -> MuCommaLambdaES:
    from unittest.mock import MagicMock

    from finchge.core.individual import Individual

    mutation = MagicMock()

    def _offspring():
        o = Individual.from_genotype([1, 2, 3])
        o.phenotype = "x"
        o.fitness = [0.0]
        return o

    mutation.mutate.side_effect = lambda ind: _offspring()
    return MuCommaLambdaES(
        lambda_=lambda_,
        mutation=mutation,
        fitness_evaluator=_FakeEvaluator(),
        random_state=42,
    )


def test_lambda_zero_raises():
    with pytest.raises(ValueError, match="lambda_"):
        _make_algorithm(lambda_=0)


def test_lambda_less_than_mu_raises_at_evolve():
    """lambda_ < mu is valid at construction but raises at evolve time."""
    from finchge.core.individual import Individual
    from finchge.core.population import Population

    algo = _make_algorithm(lambda_=3)
    mu = 10
    individuals = [Individual.from_genotype([idx]) for idx in range(mu)]
    for idx, ind in enumerate(individuals):
        ind.phenotype = "x"
        ind.fitness = [float(idx)]
    pop = Population.from_individuals(individuals, population_size=mu)
    with pytest.raises(ValueError, match="lambda_"):
        algo.evolve_one_generation(pop)


def test_next_generation_has_correct_size():
    from finchge.core.individual import Individual
    from finchge.core.population import Population

    mu = 5
    algo = _make_algorithm(lambda_=20)
    individuals = [Individual.from_genotype([idx]) for idx in range(mu)]
    for idx, ind in enumerate(individuals):
        ind.phenotype = "x"
        ind.fitness = [float(idx)]
    pop = Population.from_individuals(individuals, population_size=mu)
    next_pop = algo.evolve_one_generation(pop)
    assert next_pop.population_size == mu
