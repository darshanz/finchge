import pytest

from finchge.core.individual import Individual
from finchge.operators.mutation import InversionMutation


def make_individual(genome):
    return Individual(genotype=genome)


def test_raises_error_without_genotype():
    ind = Individual()
    mutation = InversionMutation(segment_probability=0.5, random_state=42)
    with pytest.raises(ValueError):
        mutation.mutate(ind)


def test_probability_out_of_range_raises():
    with pytest.raises(ValueError):
        InversionMutation(segment_probability=1.5)


def test_returns_new_individual():
    ind = make_individual([1, 2, 3, 4])
    mutation = InversionMutation(segment_probability=1.0, random_state=42)
    assert mutation.mutate(ind) is not ind


def test_does_not_modify_parent():
    genome = [1, 2, 3, 4, 5]
    ind = make_individual(genome.copy())
    mutation = InversionMutation(segment_probability=1.0, random_state=42)
    mutation.mutate(ind)
    assert ind.genotype == genome


def test_genome_length_preserved():
    ind = make_individual([10, 20, 30, 40, 50])
    mutation = InversionMutation(segment_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    assert len(child.genotype) == 5


def test_codon_multiset_preserved():
    genome = [5, 1, 8, 3, 9, 2, 7]
    ind = make_individual(genome.copy())
    mutation = InversionMutation(segment_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    assert sorted(child.genotype) == sorted(genome)


def test_zero_probability_returns_clone():
    genome = [10, 20, 30, 40]
    ind = make_individual(genome.copy())
    mutation = InversionMutation(segment_probability=0.0, random_state=42)
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_inverted_segment_is_reversed():
    genome = [1, 2, 3, 4, 5, 6, 7, 8]
    ind = make_individual(genome.copy())
    mutation = InversionMutation(segment_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    # Inversion reverses a contiguous segment; the sorted multiset must be unchanged
    assert sorted(child.genotype) == sorted(genome)
    # With a diverse genome, the chosen segment is almost certainly non-trivial
    assert child.genotype != genome


def test_minimum_valid_genome_length():
    genome = [1, 2]
    ind = make_individual(genome.copy())
    mutation = InversionMutation(segment_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    assert sorted(child.genotype) == sorted(genome)


def test_multiset_preserved_across_many_runs():
    genome = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    for seed in range(20):
        ind = make_individual(genome.copy())
        mutation = InversionMutation(segment_probability=1.0, random_state=seed)
        child = mutation.mutate(ind)
        assert sorted(child.genotype) == sorted(genome)


def test_deterministic_with_fixed_seed():
    ind = make_individual([3, 1, 4, 1, 5, 9, 2, 6])
    m1 = InversionMutation(segment_probability=1.0, random_state=13)
    m2 = InversionMutation(segment_probability=1.0, random_state=13)
    assert m1.mutate(ind).genotype == m2.mutate(ind).genotype


def test_inversion_varies_across_seeds():
    results = set()
    genome = [1, 2, 3, 4, 5, 6, 7, 8]
    for seed in range(30):
        ind = make_individual(genome.copy())
        mutation = InversionMutation(segment_probability=1.0, random_state=seed)
        results.add(tuple(mutation.mutate(ind).genotype))
    assert len(results) > 1
