from unittest.mock import MagicMock

import pytest

from finchge.algorithm.steady_state_ga import SteadyStateGA
from finchge.core.individual import Individual
from finchge.core.population import Population


def _make_individual(fitness_val: float) -> Individual:
    ind = Individual.from_genotype([1, 2, 3])
    ind.phenotype = "x"
    ind.fitness = [fitness_val]
    return ind


def _make_algorithm(maximize: bool = False) -> SteadyStateGA:
    class _FakeEvaluator:
        def get_maximize_flags(self):
            return [maximize]

        def refresh_mapping_all(self, individuals):
            pass

        def evaluate_population(self, population):
            for ind in population.individuals:
                if ind.fitness is None:
                    ind.fitness = [0.5]

    selection = MagicMock()
    selection.select.side_effect = lambda population_size, individuals: individuals[:2]

    crossover = MagicMock()
    crossover.cross.side_effect = lambda p1, p2: (
        _make_individual(0.5),
        _make_individual(0.5),
    )

    mutation = MagicMock()
    mutation.mutate.side_effect = lambda ind: ind

    return SteadyStateGA(
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        fitness_evaluator=_FakeEvaluator(),
        random_state=42,
    )


def test_max_best_reflects_evaluator_minimize():
    assert _make_algorithm(maximize=False).max_best is False


def test_max_best_reflects_evaluator_maximize():
    assert _make_algorithm(maximize=True).max_best is True


def test_population_size_below_four_raises():
    algo = _make_algorithm()
    individuals = [_make_individual(float(i)) for i in range(3)]
    pop = Population.from_individuals(individuals, population_size=3)
    with pytest.raises(ValueError, match="population_size >= 4"):
        algo.evolve_one_generation(pop)


def test_population_size_preserved_after_generation():
    algo = _make_algorithm()
    individuals = [_make_individual(float(i)) for i in range(8)]
    pop = Population.from_individuals(individuals, population_size=8)
    next_pop = algo.evolve_one_generation(pop)
    assert next_pop.population_size == 8
    assert len(next_pop.individuals) == 8


def test_only_worst_individuals_displaced():
    """After one generation the two best should survive; only the worst are replaced."""
    algo = _make_algorithm(maximize=False)  # lower fitness = better
    # Give the two best individuals fitness 0.0 and 0.1 — they should survive
    individuals = [_make_individual(0.0), _make_individual(0.1)] + [
        _make_individual(float(i) + 5.0) for i in range(6)
    ]
    pop = Population.from_individuals(individuals, population_size=8)
    next_pop = algo.evolve_one_generation(pop)
    fitnesses = [
        ind.fitness[0] for ind in next_pop.individuals if ind.has_usable_fitness()
    ]
    assert 0.0 in fitnesses, "Best individual should survive steady-state replacement"
    assert 0.1 in fitnesses, "Second-best should survive steady-state replacement"
