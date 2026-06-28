import pytest

from finchge.core.individual import Individual
from finchge.operators.mutation import SwapMutation


def make_individual(genome):
    return Individual(genotype=genome)


def test_raises_error_without_genotype():
    """Requires genotype; raises ValueError when missing"""
    ind = Individual()
    mutation = SwapMutation(mutation_probability=0.5, random_state=42)
    with pytest.raises(ValueError):
        mutation.mutate(ind)


def test_probability_out_of_range_raises():
    """Probability outside [0, 1] raises ValueError"""
    with pytest.raises(ValueError):
        SwapMutation(mutation_probability=1.5)


def test_returns_new_individual():
    """Mutate returns a new object, not the original"""
    ind = make_individual([1, 2, 3, 4])
    mutation = SwapMutation(mutation_probability=0.5, random_state=42)
    assert mutation.mutate(ind) is not ind


def test_does_not_modify_parent():
    """Parent genotype is unchanged after swap mutation"""
    genome = [1, 2, 3, 4]
    ind = make_individual(genome.copy())
    mutation = SwapMutation(mutation_probability=1.0, random_state=42)
    mutation.mutate(ind)
    assert ind.genotype == genome


def test_genome_length_preserved():
    """Genome length is unchanged after swap"""
    ind = make_individual([10, 20, 30, 40, 50])
    mutation = SwapMutation(mutation_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    assert len(child.genotype) == 5


def test_codon_multiset_preserved():
    """Swap is a permutation: sorted values match before and after"""
    genome = [1, 2, 3, 4, 5, 6, 7, 8]
    ind = make_individual(genome.copy())
    mutation = SwapMutation(mutation_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    assert sorted(child.genotype) == sorted(genome)


def test_zero_probability_no_swap():
    """Probability zero selects no positions; genome is returned unchanged"""
    genome = [10, 20, 30, 40]
    ind = make_individual(genome.copy())
    mutation = SwapMutation(mutation_probability=0.0, random_state=42)
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_odd_selected_positions_preserves_multiset():
    """When an odd number of positions is selected, the last is dropped but multiset is intact"""
    # A genome of length 3 with p=1.0 selects all 3 positions (odd count).
    # The last unpaired position is silently dropped; the values are still a permutation.
    genome = [10, 20, 30]
    ind = make_individual(genome.copy())
    mutation = SwapMutation(mutation_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    assert sorted(child.genotype) == sorted(genome)
    assert len(child.genotype) == 3


def test_single_element_genome_no_swap():
    """A genome of length 1 cannot form a pair; returned unchanged"""
    genome = [42]
    ind = make_individual(genome.copy())
    mutation = SwapMutation(mutation_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_multiset_preserved_across_many_runs():
    """Codon multiset invariant holds across repeated calls"""
    genome = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    for seed in range(20):
        ind = make_individual(genome.copy())
        mutation = SwapMutation(mutation_probability=0.5, random_state=seed)
        child = mutation.mutate(ind)
        assert sorted(child.genotype) == sorted(genome)


def test_deterministic_with_fixed_seed():
    """Same random_state produces identical swap results"""
    ind = make_individual([5, 10, 15, 20, 25, 30])
    m1 = SwapMutation(mutation_probability=0.5, random_state=99)
    m2 = SwapMutation(mutation_probability=0.5, random_state=99)
    assert m1.mutate(ind).genotype == m2.mutate(ind).genotype


def test_high_probability_varies_ordering():
    """With p=1.0, mutation produces different orderings across seeds"""
    results = set()
    for seed in range(50):
        ind = make_individual([1, 2, 3, 4, 5, 6])
        mutation = SwapMutation(mutation_probability=1.0, random_state=seed)
        results.add(tuple(mutation.mutate(ind).genotype))
    assert len(results) > 1
