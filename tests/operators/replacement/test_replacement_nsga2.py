from finchge.core.individual import Individual
from finchge.operators.replacement import NSGA2Replacement


def make_individual(fitness, rank=None, crowding_distance=0.0):
    ind = Individual()
    ind.fitness = fitness
    ind.set_meta("crowding_distance", crowding_distance)
    if rank is not None:
        ind.set_meta("rank", rank)
    return ind


def test_nsga2_replacement_elite_size_has_no_effect():
    """
    Regression test: elite_size is accepted for interface consistency only.
    Two calls differing only in elite_size must return identical results.
    """
    unranked_old = make_individual([3.0, 3.0])
    ranked_old = make_individual([2.0, 2.0], rank=0)
    child = make_individual([1.0, 1.0], rank=0)

    replacement = NSGA2Replacement(maximize_flags=[False, False])

    selected_with_elites = replacement.replace(
        old_population=[unranked_old, ranked_old],
        new_population=[child],
        elite_size=1,
        population_size=2,
    )
    selected_without_elites = replacement.replace(
        old_population=[unranked_old, ranked_old],
        new_population=[child],
        elite_size=0,
        population_size=2,
    )

    assert len(selected_with_elites) == 2
    assert selected_with_elites == selected_without_elites
