import warnings
from collections import Counter

import pytest

from finchge.core.individual import Individual
from finchge.operators.selection import RouletteWheelSelection


def make_population(values):
    population = []
    for v in values:
        ind = Individual()
        ind.fitness = v
        population.append(ind)

    return population


def test_returns_correct_population_size():
    """Basic behaviour , correct ppulation size"""
    population = make_population([i for i in range(5)])
    selector = RouletteWheelSelection(max_best=True, random_state=42)

    selected = selector.select(3, population)

    assert len(selected) == 3


def test_returns_individuals_from_population():
    """Basic behaviour , from population"""
    population = make_population([i for i in range(5)])
    selector = RouletteWheelSelection(max_best=True, random_state=42)

    selected = selector.select(10, population)

    assert all(ind in population for ind in selected)


def test_deterministic_with_random_state():
    """Deterministic behaviour"""
    population = make_population([i for i in range(5)])

    selector1 = RouletteWheelSelection(max_best=True, random_state=123)
    selector2 = RouletteWheelSelection(max_best=True, random_state=123)

    result1 = selector1.select(20, population)
    result2 = selector2.select(20, population)

    assert [id(i) for i in result1] == [id(i) for i in result2]


def test_higher_fitness_selected_more_in_maximization():
    """Maximization behaviour"""
    population = make_population([1, 10])

    selector = RouletteWheelSelection(max_best=True, random_state=42)

    selected = selector.select(1000, population)

    counts = Counter(ind.fitness for ind in selected)

    assert counts[10] > counts[1]


def test_lower_fitness_selected_more_in_minimization():
    """Minimization behaviour"""
    population = make_population([1, 10])

    selector = RouletteWheelSelection(max_best=False, random_state=42)

    selected = selector.select(1000, population)

    counts = Counter(ind.fitness for ind in selected)

    assert counts[1] > counts[10]


def test_handles_negative_fitness():
    """Negative fitness handling"""
    population = make_population([-5, -1, -3])
    selector = RouletteWheelSelection(max_best=True, random_state=42)
    selected = selector.select(20, population)
    assert len(selected) == 20


def test_fallback_to_uniform_selection_when_all_zero():
    """Zero-weight fallback"""
    population = make_population([0, 0, 0])
    selector = RouletteWheelSelection(max_best=False, random_state=42)
    with pytest.warns(UserWarning):
        selected = selector.select(100, population)
    # Ensure selection still happens
    assert len(selected) == 100


def test_uniform_selection_distribution_on_fallback():
    """Uniform fallback statistical check"""
    population = make_population([0, 0, 0])

    selector = RouletteWheelSelection(max_best=False, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        selected = selector.select(3000, population)

    counts = Counter(id(ind) for ind in selected)

    # Rough uniform distribution check
    for count in counts.values():
        assert 800 < count < 1200  # wide statistical tolerance


def test_mixed_positive_and_negative_fitness():
    """Mixed positive/negative fitness"""
    population = make_population([-5, 0, 10])
    selector = RouletteWheelSelection(max_best=True, random_state=42)
    selected = selector.select(50, population)
    assert len(selected) == 50
