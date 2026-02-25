import pytest

from finchge.core.individual import Individual
from finchge.operators.selection import TournamentSelection


def make_population(values):
    population = []
    for v in values:
        ind = Individual()
        ind.fitness = v
        population.append(ind)

    return population


def test_tournament_selection_size():
    """Correct number of individuals selected"""
    pop = make_population([1, 2, 3, 4, 5])
    sel = TournamentSelection(max_best=True, tournament_size=3, random_state=42)
    selected = sel.select(4, pop)
    assert len(selected) == 4


def test_selection_membership():
    """Selected individuals belong to original population"""
    pop = make_population([1, 2, 3])
    sel = TournamentSelection(max_best=True, random_state=42)

    selected = sel.select(5, pop)

    for ind in selected:
        assert ind in pop


def test_maximization_full_tournament():
    """Maximization selects best individual with tournament_size == population"""
    pop = make_population([1, 2, 3, 10])
    sel = TournamentSelection(max_best=True, tournament_size=len(pop), random_state=1)

    selected = sel.select(5, pop)

    assert all(ind.fitness == 10 for ind in selected)


def test_minimization_full_tournament():
    """Minimization selects lowest fitness when full tournament"""
    pop = make_population([5, 3, 9, 1])
    sel = TournamentSelection(max_best=False, tournament_size=len(pop), random_state=1)
    selected = sel.select(5, pop)
    assert all(ind.fitness == 1 for ind in selected)


def test_tournament_reproducible():
    """Deterministic reproducibility"""
    pop = make_population([1, 2, 3, 4, 5])
    sel1 = TournamentSelection(max_best=True, random_state=42)
    sel2 = TournamentSelection(max_best=True, random_state=42)
    out1 = [ind.fitness for ind in sel1.select(10, pop)]
    out2 = [ind.fitness for ind in sel2.select(10, pop)]
    assert out1 == out2


def test_population_smaller_than_tournament():
    """Raises error when population smaller than tournament"""
    pop = make_population([1, 2])
    sel = TournamentSelection(max_best=True, tournament_size=3)
    with pytest.raises(ValueError):
        sel.select(2, pop)


def test_larger_tournament_biases_best():
    """Tournament size increases selection pressure"""
    pop = make_population([1, 2, 3, 4, 100])

    small = TournamentSelection(max_best=True, tournament_size=2, random_state=1)
    large = TournamentSelection(max_best=True, tournament_size=5, random_state=1)

    small_best = sum(ind.fitness == 100 for ind in small.select(200, pop))
    large_best = sum(ind.fitness == 100 for ind in large.select(200, pop))

    assert large_best >= small_best


def test_selection_with_replacement():
    """Works when population_size > original population"""
    pop = make_population([1, 2, 3])
    sel = TournamentSelection(max_best=True, random_state=2)
    selected = sel.select(20, pop)
    assert len(selected) == 20


def test_identical_fitness():
    """Works with identical fitness individuals"""
    pop = make_population([5, 5, 5, 5])
    sel = TournamentSelection(max_best=True, random_state=5)
    selected = sel.select(10, pop)
    assert len(selected) == 10


def test_binary_tournament_valid():
    """Tournament size = 2 behaves correctly"""
    pop = make_population([1, 2, 3, 4])
    sel = TournamentSelection(max_best=True, tournament_size=2, random_state=3)
    selected = sel.select(10, pop)
    assert len(selected) == 10
