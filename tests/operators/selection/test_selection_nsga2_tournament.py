from collections import Counter

from finchge.core.individual import Individual
from finchge.operators.selection import NSGA2TournamentSelection


def make_population(values):
    population = []
    for rank, crowding_distance in values:
        ind = Individual()
        ind.meta = {"rank": rank, "crowding_distance": crowding_distance}
        population.append(ind)

    return population


def make_individual(rank, crowding_distance):
    ind = Individual()
    ind.meta = {"rank": rank, "crowding_distance": crowding_distance}
    return ind


def test_selection_returns_requested_population_size():
    #  the selection method returns exactly the number
    # of individuals specified by population_size, regardless of tournament logic.
    population = make_population((0, 1.0) for _ in range(10))
    selector = NSGA2TournamentSelection(random_state=42)
    selected = selector.select(5, population)
    assert len(selected) == 5


def test_selection_only_returns_members_from_population():
    # the selection process only returns individuals
    # from the provided population and does not create new objects.
    population = make_population((0, 1.0) for _ in range(10))
    selector = NSGA2TournamentSelection(random_state=42)
    selected = selector.select(20, population)
    assert all(ind in population for ind in selected)


def test_crowded_operator_prefers_lower_rank():
    # the crowded comparison operator selects individuals
    # with lower Pareto rank when ranks differ, regardless of crowding distance.
    better = make_individual(rank=0, crowding_distance=0.1)
    worse = make_individual(rank=1, crowding_distance=100.0)
    selector = NSGA2TournamentSelection(random_state=42)
    winner = selector.crowded_comparison_operator([worse, better])
    assert winner is better


def test_crowded_operator_prefers_higher_crowding_distance_when_ranks_equal():
    # when individuals have the same Pareto rank,
    # the crowded comparison operator chooses the individual with the
    # higher crowding distance.
    ind1 = make_individual(rank=0, crowding_distance=0.1)
    ind2 = make_individual(rank=0, crowding_distance=5.0)
    selector = NSGA2TournamentSelection(random_state=42)
    winner = selector.crowded_comparison_operator([ind1, ind2])
    assert winner is ind2


def test_crowded_operator_handles_single_individual_tournament():
    # if the tournament contains only one individual,
    # the crowded comparison operator simply returns that individual.
    individual = make_individual(rank=0, crowding_distance=1.0)
    selector = NSGA2TournamentSelection(random_state=42)
    winner = selector.crowded_comparison_operator([individual])
    assert winner is individual


def test_selection_respects_tournament_size_limit():
    # if the configured tournament size is larger
    # than the population size, the selection logic safely uses the entire
    # population instead of failing.
    population = make_population((i, 1.0) for i in range(3))
    selector = NSGA2TournamentSelection(tournament_size=10, random_state=42)
    selected = selector.select(10, population)
    assert len(selected) == 10
    assert all(ind in population for ind in selected)


def test_selection_uses_crowded_operator_when_exploration_probability_zero():
    # when exploration_prob is set to zero, the selection
    # always relies on the crowded comparison operator and never randomly selects
    # individuals from the tournament.
    best = make_individual(rank=0, crowding_distance=10.0)
    worst = make_individual(rank=1, crowding_distance=100.0)
    population = [best, worst]
    selector = NSGA2TournamentSelection(
        tournament_size=2, exploration_prob=0.0, random_state=42
    )
    selected = selector.select(50, population)
    assert all(ind is best for ind in selected)


def test_selection_uses_random_choice_when_exploration_probability_one():
    # when exploration_prob is set to one, the selection
    # ignores the crowded comparison operator and always randomly selects from
    # tournament participants.
    best = make_individual(rank=0, crowding_distance=10.0)
    worst = make_individual(rank=1, crowding_distance=100.0)
    population = [best, worst]
    selector = NSGA2TournamentSelection(
        tournament_size=2, exploration_prob=1.0, random_state=42
    )
    selected = selector.select(200, population)
    counts = Counter(id(ind) for ind in selected)
    assert len(counts) == 2


def test_selection_is_deterministic_with_fixed_random_state():
    # using the same random_state produces identical
    # selection sequences, ensuring reproducibility.
    population = make_population((i % 2, 1.0) for i in range(10))
    selector1 = NSGA2TournamentSelection(random_state=123)
    selector2 = NSGA2TournamentSelection(random_state=123)

    result1 = selector1.select(30, population)
    result2 = selector2.select(30, population)

    assert [id(i) for i in result1] == [id(i) for i in result2]


def test_selection_handles_population_of_one():
    # selection behaves correctly when only one
    # individual exists in the population. The same individual should be
    # returned repeatedly.
    individual = make_individual(rank=0, crowding_distance=1.0)
    population = [individual]
    selector = NSGA2TournamentSelection(random_state=42)
    selected = selector.select(20, population)
    assert all(ind is individual for ind in selected)
