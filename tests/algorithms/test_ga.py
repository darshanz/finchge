from unittest.mock import MagicMock

import pytest

from finchge.algorithm.ga import GeneticAlgorithm
from finchge.core.individual import Individual
from finchge.core.population import Population


class _FakeEvaluator:
    def __init__(self, maximize: bool = False):
        self._maximize = maximize

    def get_maximize_flags(self):
        return [self._maximize]

    def refresh_mapping_all(self, individuals):
        pass

    def evaluate_population(self, population):
        for ind in population.individuals:
            if not ind.fitness:
                ind.fitness = [0.0]

    @property
    def require_case_data(self):
        return False


def _make_individual(fitness_value: float) -> Individual:
    ind = Individual.from_genotype([1, 2, 3, 4, 5])
    ind.phenotype = "x"
    ind.fitness = [fitness_value]
    return ind


def _make_algorithm(maximize: bool = False, elite_size: int = 1) -> GeneticAlgorithm:
    evaluator = _FakeEvaluator(maximize=maximize)
    crossover = MagicMock()
    crossover.cross.side_effect = lambda a, b: (a.clone(), b.clone())
    mutation = MagicMock()
    mutation.mutate.side_effect = lambda ind: ind.clone()
    selection = MagicMock()
    selection.requires_case_data = False
    selection.select.side_effect = lambda population_size, individuals: individuals
    replacement = MagicMock()
    replacement.replace.side_effect = (
        lambda new_population, old_population, elite_size, population_size: (
            old_population[:elite_size] + new_population[: population_size - elite_size]
        )
    )

    return GeneticAlgorithm(
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        replacement=replacement,
        elite_size=elite_size,
        fitness_evaluator=evaluator,
        random_state=42,
    )


def test_max_best_reflects_evaluator_minimize():
    algo = _make_algorithm(maximize=False)
    assert algo.max_best is False


def test_max_best_reflects_evaluator_maximize():
    algo = _make_algorithm(maximize=True)
    assert algo.max_best is True


def test_multi_objective_evaluator_raises():
    evaluator = MagicMock()
    evaluator.get_maximize_flags.return_value = [True, True]
    evaluator.require_case_data = False
    with pytest.raises(ValueError, match="single-objective"):
        GeneticAlgorithm(
            selection=MagicMock(requires_case_data=False),
            crossover=MagicMock(),
            mutation=MagicMock(),
            replacement=MagicMock(),
            elite_size=1,
            fitness_evaluator=evaluator,
        )


def test_sort_population_minimize_worst_last():
    algo = _make_algorithm(maximize=False)
    pop = Population.from_individuals(
        [_make_individual(3.0), _make_individual(1.0), _make_individual(2.0)],
        population_size=3,
    )
    algo.sort_population(pop)
    fitnesses = [ind.fitness[0] for ind in pop.individuals]
    assert fitnesses == sorted(fitnesses)


def test_sort_population_maximize_best_first():
    algo = _make_algorithm(maximize=True)
    pop = Population.from_individuals(
        [_make_individual(1.0), _make_individual(3.0), _make_individual(2.0)],
        population_size=3,
    )
    algo.sort_population(pop)
    fitnesses = [ind.fitness[0] for ind in pop.individuals]
    assert fitnesses == sorted(fitnesses, reverse=True)


def test_get_best_individual_minimize():
    algo = _make_algorithm(maximize=False)
    pop = Population.from_individuals(
        [_make_individual(5.0), _make_individual(1.0), _make_individual(3.0)],
        population_size=3,
    )
    best = algo.get_best_individual(pop)
    assert best.fitness[0] == 1.0


def test_get_best_individual_maximize():
    algo = _make_algorithm(maximize=True)
    pop = Population.from_individuals(
        [_make_individual(5.0), _make_individual(1.0), _make_individual(3.0)],
        population_size=3,
    )
    best = algo.get_best_individual(pop)
    assert best.fitness[0] == 5.0


def test_get_best_individual_raises_when_no_evaluated():
    algo = _make_algorithm()
    unevaluated = Individual.from_genotype([1, 2])
    unevaluated.phenotype = "x"
    pop = Population.from_individuals([unevaluated], population_size=1)
    with pytest.raises(ValueError, match="No evaluated"):
        algo.get_best_individual(pop)


def test_evolve_one_generation_preserves_population_size():
    algo = _make_algorithm(maximize=False, elite_size=1)
    individuals = [_make_individual(float(i)) for i in range(1, 5)]
    pop = Population.from_individuals(individuals, population_size=4)
    new_pop = algo.evolve_one_generation(pop)
    assert len(new_pop) == 4


def test_evolve_raises_when_too_few_valid():
    algo = _make_algorithm()
    unevaluated = Individual.from_genotype([1])
    unevaluated.phenotype = "x"
    pop = Population.from_individuals([unevaluated], population_size=1)
    with pytest.raises(Exception, match="Not enough valid"):
        algo.evolve_one_generation(pop)
