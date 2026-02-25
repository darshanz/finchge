from collections import Counter

from finchge.core.individual import Individual
from finchge.operators.selection import NSGA3TournamentSelection


def make_population(values):
    population = []
    for rank in values:
        ind = Individual()
        ind.meta = {"rank": rank}
        population.append(ind)

    return population


def make_individual(rank):
    ind = Individual()
    ind.meta = {"rank": rank}
    return ind


def test_selection_returns_requested_population_size():
    # the selection method returns exactly the number
    # of individuals specified by population_size regardless of tournament logic.
    population = make_population(i % 3 for i in range(10))
    selector = NSGA3TournamentSelection(random_state=42)
    selected = selector.select(5, population)
    assert len(selected) == 5


def test_selection_only_returns_members_from_population():
    # selection only returns individuals from the
    # provided population and never creates new individuals.
    population = make_population(i % 3 for i in range(10))
    selector = NSGA3TournamentSelection(random_state=42)
    selected = selector.select(20, population)
    assert all(ind in population for ind in selected)


def test_rank_only_operator_selects_lowest_rank():
    # the rank-only operator always selects an
    # individual with the lowest Pareto rank when ranks differ.
    better = make_individual(rank=0)
    worse = make_individual(rank=2)
    selector = NSGA3TournamentSelection(random_state=42)
    winner = selector.rank_only_operator([worse, better])
    assert winner is better


def test_rank_only_operator_randomly_selects_among_equal_ranks():
    # when multiple individuals share the same best rank,
    # the rank-only operator randomly selects among them rather than always
    # selecting the same individual.
    ind1 = make_individual(rank=0)
    ind2 = make_individual(rank=0)
    ind3 = make_individual(rank=1)
    selector = NSGA3TournamentSelection(random_state=42)
    results = [selector.rank_only_operator([ind1, ind2, ind3]) for _ in range(200)]
    counts = Counter(id(ind) for ind in results)
    assert len(counts) == 2


def test_selection_uses_rank_operator_when_exploration_probability_zero():
    # This test verifies that when exploration_prob is set to zero, the selection
    # always relies on the rank-only operator and never randomly selects individuals
    # outside of the best-ranked group.
    best = make_individual(rank=0)
    worst = make_individual(rank=1)
    population = [best, worst]
    selector = NSGA3TournamentSelection(
        tournament_size=2, exploration_prob=0.0, random_state=42
    )
    selected = selector.select(50, population)
    assert all(ind is best for ind in selected)


def test_selection_uses_random_choice_when_exploration_probability_one():
    # when exploration_prob is set to one, selection
    # ignores rank-based comparison and randomly selects from the tournament.
    best = make_individual(rank=0)
    worst = make_individual(rank=1)
    population = [best, worst]
    selector = NSGA3TournamentSelection(
        tournament_size=2, exploration_prob=1.0, random_state=42
    )
    selected = selector.select(200, population)
    counts = Counter(id(ind) for ind in selected)
    assert len(counts) == 2


def test_selection_respects_tournament_size_limit():
    # if tournament_size is larger than the population,
    # the selection safely uses all available individuals instead of failing.
    population = make_population([i for i in range(3)])
    selector = NSGA3TournamentSelection(tournament_size=10, random_state=42)
    selected = selector.select(10, population)
    assert len(selected) == 10
    assert all(ind in population for ind in selected)


def test_selection_is_deterministic_with_fixed_random_state():
    # using the same random_state produces identical
    # selection results, ensuring reproducibility.
    population = make_population([i % 3 for i in range(10)])
    selector1 = NSGA3TournamentSelection(random_state=123)
    selector2 = NSGA3TournamentSelection(random_state=123)

    result1 = selector1.select(30, population)
    result2 = selector2.select(30, population)

    assert [id(i) for i in result1] == [id(i) for i in result2]


def test_selection_handles_population_of_one():
    # selection behaves correctly when only one
    # individual exists in the population. The same individual should be
    # returned repeatedly.
    individual = make_individual(rank=0)
    population = [individual]
    selector = NSGA3TournamentSelection(random_state=42)
    selected = selector.select(20, population)
    assert all(ind is individual for ind in selected)
