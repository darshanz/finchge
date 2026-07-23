import pytest

from finchge.algorithm.one_plus_one import OnePlusOneES
from finchge.core.individual import Individual
from finchge.core.population import Population


# minimal dummy evaluator for testing
class _FakeEvaluator:
    def __init__(self, maximize: bool = False):
        self._maximize = maximize

    def get_maximize_flags(self):
        return [self._maximize]

    def refresh_mapping_all(self, individuals):
        pass

    def evaluate_population(self, population):
        for ind in population.individuals:
            if ind.fitness is None:
                ind.fitness = [0.0]


def _make_individual(fitness_value: float) -> Individual:
    ind = Individual.from_genotype([1, 2, 3])
    ind.phenotype = "x"
    ind.fitness = [fitness_value]
    return ind


def _make_algorithm(maximize: bool = False) -> OnePlusOneES:
    evaluator = _FakeEvaluator(maximize=maximize)
    from unittest.mock import MagicMock

    mutation = MagicMock()
    mutation.mutate.side_effect = lambda ind: _make_individual(ind.fitness[0])
    return OnePlusOneES(
        mutation=mutation,
        fitness_evaluator=evaluator,
        random_state=42,
    )


def test_max_best_reflects_evaluator_minimize():
    algo = _make_algorithm(maximize=False)
    assert algo.max_best is False


def test_max_best_reflects_evaluator_maximize():
    algo = _make_algorithm(maximize=True)
    assert algo.max_best is True


def test_offspring_wins_when_lower_fitness_and_minimize():
    algo = _make_algorithm(maximize=False)
    parent = _make_individual(5.0)
    offspring = _make_individual(3.0)
    assert algo._offspring_wins(parent, offspring) is True


def test_offspring_loses_when_higher_fitness_and_minimize():
    algo = _make_algorithm(maximize=False)
    parent = _make_individual(3.0)
    offspring = _make_individual(5.0)
    assert algo._offspring_wins(parent, offspring) is False


def test_offspring_wins_when_higher_fitness_and_maximize():
    algo = _make_algorithm(maximize=True)
    parent = _make_individual(3.0)
    offspring = _make_individual(5.0)
    assert algo._offspring_wins(parent, offspring) is True


def test_offspring_loses_if_no_usable_fitness():
    algo = _make_algorithm(maximize=False)
    parent = _make_individual(3.0)
    offspring = Individual.from_genotype([1])
    offspring.fitness = None
    assert algo._offspring_wins(parent, offspring) is False


def test_raises_if_population_size_not_one():
    algo = _make_algorithm()
    pop = Population.from_individuals(
        [_make_individual(1.0), _make_individual(2.0)], population_size=2
    )
    with pytest.raises(ValueError, match="population_size=1"):
        algo.evolve_one_generation(pop)
