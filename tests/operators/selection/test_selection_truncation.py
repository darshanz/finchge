from collections import Counter

import pytest

from finchge.core.individual import Individual
from finchge.operators.selection import TruncationSelection


def make_population(values):
    population = []
    for v in values:
        ind = Individual()
        ind.fitness = v
        population.append(ind)

    return population


def test_constructor_accepts_valid_threshold_values():
    # the constructor allows valid truncation thresholds
    # within the allowed range (0.0, 1.0]. These values should not raise errors.
    TruncationSelection(max_best=True, truncation_threshold=0.5)
    TruncationSelection(max_best=True, truncation_threshold=1.0)
    TruncationSelection(max_best=True, truncation_threshold=0.01)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, 5])
def test_constructor_rejects_invalid_threshold_values(threshold):
    # the constructor raises a ValueError when the
    # truncation threshold is outside the valid range. 0.0 -1.0
    with pytest.raises(ValueError):
        TruncationSelection(max_best=True, truncation_threshold=threshold)


def test_selection_returns_requested_population_size():
    #  the selection method returns exactly the number
    # of individuals requested by population_size, regardless of threshold or sorting.
    population = make_population([i for i in range(10)])
    selector = TruncationSelection(max_best=True, random_state=42)
    selected = selector.select(5, population)
    assert len(selected) == 5


def test_selection_only_returns_members_from_original_population():
    # the selection process never creates new individuals
    # and only returns individuals that already exist in the input population.
    population = make_population([i for i in range(10)])
    selector = TruncationSelection(max_best=True, random_state=42)
    selected = selector.select(20, population)
    assert all(ind in population for ind in selected)


def test_selection_uses_top_fraction_for_maximization():
    # when performing maximization, the selector sorts
    # individuals in descending fitness order and only selects from the top
    # fraction defined by the truncation threshold.
    population = make_population([i for i in range(10)])
    selector = TruncationSelection(
        max_best=True, truncation_threshold=0.3, random_state=42
    )
    selected = selector.select(50, population)
    elite_fitness = {7, 8, 9}
    assert all(ind.fitness in elite_fitness for ind in selected)


def test_selection_uses_top_fraction_for_minimization():
    # when performing minimization, the selector sorts
    # individuals in ascending fitness order and selects only from the lowest
    # fitness values within the truncation threshold.
    population = make_population([i for i in range(10)])
    selector = TruncationSelection(
        max_best=False, truncation_threshold=0.3, random_state=42
    )
    selected = selector.select(50, population)
    elite_fitness = {0, 1, 2}
    assert all(ind.fitness in elite_fitness for ind in selected)


def test_selection_enforces_minimum_elite_pool_size_of_two():
    # the truncation selection always keeps at least two
    # individuals in the elite pool, even if the threshold calculation would
    # normally produce a smaller number.
    population = make_population([i for i in range(3)])
    selector = TruncationSelection(
        max_best=True, truncation_threshold=0.01, random_state=42
    )
    selected = selector.select(50, population)
    elite = {1, 2}
    assert all(ind.fitness in elite for ind in selected)


def test_threshold_of_one_allows_selection_from_entire_population():
    #  when the truncation threshold is set to 1.0,
    # the entire population becomes eligible for selection.
    population = make_population([i for i in range(5)])
    selector = TruncationSelection(
        max_best=True, truncation_threshold=1.0, random_state=42
    )

    selected = selector.select(100, population)

    assert set(ind.fitness for ind in selected).issubset({0, 1, 2, 3, 4})


def test_selection_is_deterministic_when_random_state_is_fixed():
    # providing the same random_state produces identical
    # selection results. for reproducibility.
    population = make_population([i for i in range(10)])

    selector1 = TruncationSelection(True, 0.5, random_state=123)
    selector2 = TruncationSelection(True, 0.5, random_state=123)

    result1 = selector1.select(30, population)
    result2 = selector2.select(30, population)

    assert [id(i) for i in result1] == [id(i) for i in result2]


def test_random_distribution_within_elite_pool():
    # individuals are randomly selected from within the
    # elite pool rather than always selecting the same individual.
    # a statistical check to confirm approximate uniform randomness among elites.
    population = make_population([i for i in range(6)])
    selector = TruncationSelection(True, 0.5, random_state=42)
    selected = selector.select(3000, population)
    counts = Counter(ind.fitness for ind in selected)
    assert set(counts.keys()) == {3, 4, 5}
    for count in counts.values():
        assert 800 < count < 1200


def test_selection_handles_population_of_two_individuals():
    # selection works correctly when the population
    # size is extremely small. The minimum elite pool size rule ensures both
    # individuals remain eligible.
    population = make_population([i for i in range(2)])
    selector = TruncationSelection(True, 0.2, random_state=42)

    selected = selector.select(20, population)

    assert set(ind.fitness for ind in selected) == {0, 1}
