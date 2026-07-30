from unittest.mock import MagicMock

import pytest

from finchge.algorithm.island_ga import IslandGA
from finchge.core.individual import Individual
from finchge.core.population import Population


def _make_individual(fitness_val: float) -> Individual:
    ind = Individual.from_genotype([1, 2, 3])
    ind.phenotype = "x"
    ind.fitness = [fitness_val]
    return ind


def _make_algorithm(
    num_islands: int = 2,
    migration_interval: int = 5,
    migration_size: int = 1,
    maximize: bool = False,
) -> IslandGA:
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

    replacement = MagicMock()
    replacement.replace.side_effect = lambda new_population, old_population, elite_size, population_size: new_population[
        :population_size
    ]

    return IslandGA(
        num_islands=num_islands,
        migration_interval=migration_interval,
        migration_size=migration_size,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        replacement=replacement,
        elite_size=0,
        fitness_evaluator=_FakeEvaluator(),
        random_state=42,
    )


def test_num_islands_one_raises():
    with pytest.raises(ValueError, match="num_islands"):
        _make_algorithm(num_islands=1)


def test_migration_size_zero_raises():
    with pytest.raises(ValueError, match="migration_size"):
        _make_algorithm(migration_size=0)


def test_population_not_divisible_by_islands_raises():
    algo = _make_algorithm(num_islands=3)
    individuals = [_make_individual(float(i)) for i in range(10)]  # 10 % 3 != 0
    pop = Population.from_individuals(individuals, population_size=10)
    with pytest.raises(ValueError, match="divisible"):
        algo.evolve_one_generation(pop)


def test_migration_size_too_large_raises():
    algo = _make_algorithm(num_islands=2, migration_size=5)
    individuals = [
        _make_individual(float(i)) for i in range(8)
    ]  # island_size=4, 5 >= 4
    pop = Population.from_individuals(individuals, population_size=8)
    with pytest.raises(ValueError, match="migration_size"):
        algo.evolve_one_generation(pop)


def test_population_size_preserved():
    algo = _make_algorithm(num_islands=2, migration_interval=10, migration_size=1)
    individuals = [_make_individual(float(i)) for i in range(8)]
    pop = Population.from_individuals(individuals, population_size=8)
    next_pop = algo.evolve_one_generation(pop)
    assert next_pop.population_size == 8
    assert len(next_pop.individuals) == 8


def test_migration_moves_best_to_next_island():
    """After migration, the best from island 0 should appear in island 1."""
    algo = _make_algorithm(num_islands=2, migration_interval=1, migration_size=1)
    # Island 0 gets fitness 0.0–0.3 (very good); island 1 gets 5.0–5.3 (poor)
    individuals = [_make_individual(float(i) * 0.1) for i in range(4)] + [
        _make_individual(float(i) + 5.0) for i in range(4)
    ]
    pop = Population.from_individuals(individuals, population_size=8)
    algo.evolve_one_generation(pop)
    assert algo._islands is not None
    island_1_fitnesses = [
        ind.fitness[0]
        for ind in algo._islands[1].individuals
        if ind.has_usable_fitness()
    ]
    assert any(
        f < 1.0 for f in island_1_fitnesses
    ), "Best individual from island 0 should have migrated to island 1"
