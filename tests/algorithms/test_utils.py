from finchge.algorithm.utils import (
    calculate_crowding_distance,
    dominates,
    fast_non_dominated_sort,
)
from finchge.core.individual import Individual


def _ind(fitness: list[float]) -> Individual:
    ind = Individual.from_genotype([1, 2, 3])
    ind.phenotype = "x"
    ind.fitness = fitness
    return ind


def _invalid_ind() -> Individual:
    ind = Individual.from_genotype([1])
    ind.mark_invalid()
    return ind


# --- dominates ---


def test_dominates_maximise_a_better_on_all():
    a = _ind([3.0, 4.0])
    b = _ind([2.0, 3.0])
    assert dominates(a, b, [True, True]) is True


def test_dominates_maximise_b_not_dominated_when_equal():
    a = _ind([3.0, 3.0])
    b = _ind([3.0, 3.0])
    assert dominates(a, b, [True, True]) is False


def test_dominates_minimise_a_better_on_all():
    a = _ind([1.0, 1.0])
    b = _ind([2.0, 3.0])
    assert dominates(a, b, [False, False]) is True


def test_dominates_mixed_directions():
    # obj0 maximise, obj1 minimise; a is no worse on both and better on obj0
    a = _ind([5.0, 1.0])
    b = _ind([3.0, 1.0])
    assert dominates(a, b, [True, False]) is True


def test_dominates_invalid_individual_never_dominates():
    bad = _invalid_ind()
    good = _ind([1.0, 2.0])
    assert dominates(bad, good, [True, True]) is False


def test_dominates_valid_always_dominates_invalid():
    good = _ind([1.0, 2.0])
    bad = _invalid_ind()
    assert dominates(good, bad, [True, True]) is True


def test_dominates_two_invalids_neither_dominates():
    a = _invalid_ind()
    b = _invalid_ind()
    assert dominates(a, b, [True, True]) is False
    assert dominates(b, a, [True, True]) is False


# --- fast_non_dominated_sort ---


def test_fast_non_dominated_sort_front0_non_dominated_only():
    a = _ind([3.0, 1.0])
    b = _ind([1.0, 3.0])
    c = _ind([2.0, 2.0])  # dominated by neither a nor b in maximise context
    fronts = fast_non_dominated_sort([a, b, c], [True, True])
    front0 = fronts[0]
    # a, b, c are all mutually non-dominating: a better on obj0, b better on obj1
    assert a in front0
    assert b in front0


def test_fast_non_dominated_sort_dominated_not_in_front0():
    a = _ind([3.0, 3.0])
    b = _ind([1.0, 1.0])  # dominated by a under maximise
    fronts = fast_non_dominated_sort([a, b], [True, True])
    assert a in fronts[0]
    assert b not in fronts[0]
    assert b in fronts[1]


def test_fast_non_dominated_sort_invalid_never_in_front0():
    good = _ind([5.0, 5.0])
    bad = _invalid_ind()
    fronts = fast_non_dominated_sort([good, bad], [True, True])
    assert good in fronts[0]
    assert bad not in fronts[0]


def test_fast_non_dominated_sort_all_invalids_land_in_front0():
    # With all invalids, dominates() returns False in both directions so neither
    # accumulates domination_count > 0
    # both end up in front 0 with rank 0.
    a = _invalid_ind()
    b = _invalid_ind()
    fronts = fast_non_dominated_sort([a, b], [True, True])
    assert a in fronts[0]
    assert b in fronts[0]


def test_fast_non_dominated_sort_assigns_rank_meta():
    a = _ind([3.0, 3.0])
    b = _ind([1.0, 1.0])
    fast_non_dominated_sort([a, b], [True, True])
    assert a.meta["rank"] == 0
    assert b.meta["rank"] == 1


# --- calculate_crowding_distance ---


def test_calculate_crowding_distance_boundary_get_inf():
    front = [_ind([1.0, 4.0]), _ind([2.0, 3.0]), _ind([3.0, 1.0])]
    calculate_crowding_distance(front)
    sorted_by_obj0 = sorted(front, key=lambda ind: ind.fitness[0])
    assert sorted_by_obj0[0].meta["crowding_distance"] == float("inf")
    assert sorted_by_obj0[-1].meta["crowding_distance"] == float("inf")


def test_calculate_crowding_distance_interior_positive():
    front = [_ind([1.0, 4.0]), _ind([2.0, 3.0]), _ind([3.0, 1.0])]
    calculate_crowding_distance(front)
    distances = [ind.meta["crowding_distance"] for ind in front]
    assert all(d >= 0 for d in distances)
    assert any(0 < d < float("inf") for d in distances)


def test_calculate_crowding_distance_invalid_gets_zero():
    bad = _invalid_ind()
    good1 = _ind([1.0, 3.0])
    good2 = _ind([3.0, 1.0])
    calculate_crowding_distance([good1, bad, good2])
    assert bad.meta["crowding_distance"] == 0.0


def test_calculate_crowding_distance_two_valid_both_inf():
    a = _ind([1.0, 2.0])
    b = _ind([2.0, 1.0])
    calculate_crowding_distance([a, b])
    assert a.meta["crowding_distance"] == float("inf")
    assert b.meta["crowding_distance"] == float("inf")


def test_calculate_crowding_distance_empty_front_no_crash():
    calculate_crowding_distance([])  # should not raise
