import pytest

from finchge.core.individual import Individual
from finchge.operators.selection import RankSelection


def make_population(fitness_values):
    population = []
    for v in fitness_values:
        ind = Individual()
        ind.fitness = [v]
        population.append(ind)
    return population


def test_selection_pressure_below_one_raises():
    with pytest.raises(ValueError):
        RankSelection(max_best=True, selection_pressure=0.9)


def test_selection_pressure_above_two_raises():
    with pytest.raises(ValueError):
        RankSelection(max_best=True, selection_pressure=2.1)


def test_boundary_pressures_are_valid():
    RankSelection(max_best=True, selection_pressure=1.0)
    RankSelection(max_best=True, selection_pressure=2.0)


def test_returns_correct_population_size():
    pop = make_population([1, 2, 3, 4, 5])
    sel = RankSelection(max_best=True, random_state=42)
    assert len(sel.select(10, pop)) == 10


def test_selected_individuals_belong_to_population():
    pop = make_population([1, 2, 3, 4])
    sel = RankSelection(max_best=True, random_state=42)
    assert all(ind in pop for ind in sel.select(20, pop))


def test_maximum_pressure_never_selects_worst():
    # Baker's formula with SP=2.0: worst individual (rank n-1) gets weight = 2 - 2 + 0 = 0.
    pop = make_population([1.0, 100.0])
    sel = RankSelection(max_best=True, selection_pressure=2.0, random_state=42)
    selected = sel.select(200, pop)
    assert all(ind.fitness == [100.0] for ind in selected)


def test_minimum_pressure_gives_uniform_selection():
    # Baker's formula with SP=1.0: all weights equal 1.0 regardless of rank.
    pop = make_population([1.0, 100.0])
    sel = RankSelection(max_best=True, selection_pressure=1.0, random_state=42)
    selected = sel.select(1000, pop)
    low_count = sum(1 for ind in selected if ind.fitness == [1.0])
    high_count = sum(1 for ind in selected if ind.fitness == [100.0])
    assert 400 < low_count < 600
    assert 400 < high_count < 600


def test_higher_pressure_selects_best_more_often():
    pop = make_population([1.0, 10.0, 100.0])
    low_pressure = RankSelection(max_best=True, selection_pressure=1.1, random_state=42)
    high_pressure = RankSelection(
        max_best=True, selection_pressure=1.9, random_state=42
    )
    low_best = sum(1 for ind in low_pressure.select(500, pop) if ind.fitness == [100.0])
    high_best = sum(
        1 for ind in high_pressure.select(500, pop) if ind.fitness == [100.0]
    )
    assert high_best > low_best


def test_maximization_best_ranked_selected_most():
    pop = make_population([1.0, 5.0, 10.0])
    sel = RankSelection(max_best=True, selection_pressure=1.9, random_state=42)
    selected = sel.select(300, pop)
    count_best = sum(1 for ind in selected if ind.fitness == [10.0])
    count_worst = sum(1 for ind in selected if ind.fitness == [1.0])
    assert count_best > count_worst


def test_minimization_lowest_fitness_ranked_first():
    pop = make_population([1.0, 5.0, 10.0])
    sel = RankSelection(max_best=False, selection_pressure=1.9, random_state=42)
    selected = sel.select(300, pop)
    count_best_for_min = sum(1 for ind in selected if ind.fitness == [1.0])
    count_worst_for_min = sum(1 for ind in selected if ind.fitness == [10.0])
    assert count_best_for_min > count_worst_for_min


def test_deterministic_with_fixed_seed():
    pop = make_population([1, 2, 3, 4, 5])
    s1 = RankSelection(max_best=True, random_state=42)
    s2 = RankSelection(max_best=True, random_state=42)
    r1 = [id(ind) for ind in s1.select(20, pop)]
    r2 = [id(ind) for ind in s2.select(20, pop)]
    assert r1 == r2


def test_selection_varies_across_seeds():
    pop = make_population([1, 2, 3, 4, 5])
    results = set()
    for seed in range(20):
        sel = RankSelection(max_best=True, selection_pressure=1.5, random_state=seed)
        results.add(tuple(id(ind) for ind in sel.select(5, pop)))
    assert len(results) > 1
