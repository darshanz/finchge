import pytest

from finchge.core.individual import Individual


def test_from_genotype_sets_genome_and_none_fitness():
    ind = Individual.from_genotype([10, 20, 30])
    assert ind.genotype == [10, 20, 30]
    assert ind.fitness == []
    assert ind.phenotype is None


def test_has_usable_fitness_empty():
    ind = Individual.from_genotype([1])
    assert ind.has_usable_fitness() is False


def test_has_usable_fitness_nan():
    ind = Individual.from_genotype([1])
    ind.phenotype = "x"
    ind.fitness = [float("nan")]
    assert ind.has_usable_fitness() is False


def test_has_usable_fitness_inf():
    ind = Individual.from_genotype([1])
    ind.phenotype = "x"
    ind.fitness = [float("inf")]
    assert ind.has_usable_fitness() is False


def test_has_usable_fitness_valid():
    ind = Individual.from_genotype([1])
    ind.phenotype = "x"
    ind.fitness = [3.5]
    assert ind.has_usable_fitness() is True


def test_has_usable_fitness_invalid_individual():
    ind = Individual.from_genotype([1])
    ind.mark_invalid()
    ind.fitness = [1.0]
    assert ind.has_usable_fitness() is False


def test_sort_key_maximize_valid_returns_fitness():
    ind = Individual.from_genotype([1])
    ind.phenotype = "x"
    ind.fitness = [7.0]
    assert ind.sort_key(maximize=True) == 7.0


def test_sort_key_minimize_valid_returns_fitness():
    ind = Individual.from_genotype([1])
    ind.phenotype = "x"
    ind.fitness = [3.0]
    assert ind.sort_key(maximize=False) == 3.0


def test_sort_key_maximize_invalid_returns_neg_inf():
    ind = Individual.from_genotype([1])
    assert ind.sort_key(maximize=True) == float("-inf")


def test_sort_key_minimize_invalid_returns_pos_inf():
    ind = Individual.from_genotype([1])
    assert ind.sort_key(maximize=False) == float("inf")


def test_sort_key_orders_maximize_correctly():
    good = Individual.from_genotype([1])
    good.phenotype = "x"
    good.fitness = [5.0]

    bad = Individual.from_genotype([2])
    bad.phenotype = "y"
    bad.fitness = [2.0]

    unevaluated = Individual.from_genotype([3])

    ranked = sorted(
        [bad, unevaluated, good], key=lambda i: i.sort_key(True), reverse=True
    )
    assert ranked[0] is good
    assert ranked[-1] is unevaluated


def test_sort_key_orders_minimize_correctly():
    good = Individual.from_genotype([1])
    good.phenotype = "x"
    good.fitness = [1.0]

    bad = Individual.from_genotype([2])
    bad.phenotype = "y"
    bad.fitness = [9.0]

    unevaluated = Individual.from_genotype([3])

    ranked = sorted([bad, unevaluated, good], key=lambda i: i.sort_key(False))
    assert ranked[0] is good
    assert ranked[-1] is unevaluated


def test_is_valid_after_phenotype_set():
    ind = Individual.from_genotype([1])
    ind.phenotype = "x"
    assert ind.is_valid() is True


def test_is_valid_false_when_invalid_flag():
    ind = Individual.from_genotype([1])
    ind.mark_invalid()
    assert ind.is_valid() is False


def test_is_mapped_false_before_any_evaluation():
    ind = Individual.from_genotype([1])
    assert ind.is_mapped() is False


def test_clone_is_independent():
    ind = Individual.from_genotype([1, 2, 3])
    ind.phenotype = "abc"
    ind.fitness = [4.0]
    ind.meta["rank"] = 0

    clone = ind.clone()
    clone.genotype[0] = 99
    clone.fitness[0] = 99.0

    assert ind.genotype[0] == 1
    assert ind.fitness[0] == 4.0


def test_clone_copies_all_fields():
    ind = Individual.from_genotype([5, 6])
    ind.phenotype = "test"
    ind.fitness = [1.0, 2.0]
    ind.used_codon_count = 3

    clone = ind.clone()
    assert clone.genotype == [5, 6]
    assert clone.phenotype == "test"
    assert clone.fitness == [1.0, 2.0]
    assert clone.used_codon_count == 3


def test_get_meta_raises_on_missing_key():
    ind = Individual.from_genotype([1])
    with pytest.raises(ValueError, match="Missing meta key"):
        ind.get_meta("nonexistent")


def test_has_meta_true_after_set():
    ind = Individual.from_genotype([1])
    ind.set_meta("rank", 0)
    assert ind.has_meta("rank") is True


def test_mark_invalid_clears_phenotype():
    ind = Individual.from_genotype([1])
    ind.phenotype = "x"
    ind.mark_invalid()
    assert ind.invalid is True
    assert ind.phenotype is None
