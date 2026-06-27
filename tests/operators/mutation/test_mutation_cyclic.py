import pytest

from finchge.core.individual import Individual
from finchge.operators.mutation import CyclicMutation


def make_individual(genome):
    return Individual(genotype=genome)


def test_raises_error_without_genotype():
    """Requires genotype; raises ValueError when missing"""
    ind = Individual()
    mutation = CyclicMutation(mutation_probability=0.5, random_state=42)
    with pytest.raises(ValueError):
        mutation.mutate(ind)


def test_probability_out_of_range_raises():
    """Probability outside [0, 1] raises ValueError"""
    with pytest.raises(ValueError):
        CyclicMutation(mutation_probability=1.5)


def test_segment_size_less_than_two_raises():
    """segment_size < 2 raises ValueError"""
    with pytest.raises(ValueError):
        CyclicMutation(mutation_probability=0.5, segment_size=1)


def test_returns_new_individual():
    """Mutate returns a new object, not the original"""
    ind = make_individual([1, 2, 3, 4, 5])
    mutation = CyclicMutation(mutation_probability=0.5, random_state=42)
    assert mutation.mutate(ind) is not ind


def test_does_not_modify_parent():
    """Parent genotype is unchanged after cyclic mutation"""
    genome = [1, 2, 3, 4, 5]
    ind = make_individual(genome.copy())
    mutation = CyclicMutation(mutation_probability=1.0, random_state=42)
    mutation.mutate(ind)
    assert ind.genotype == genome


def test_genome_length_preserved():
    """Genome length is unchanged after cyclic mutation"""
    ind = make_individual([10, 20, 30, 40, 50])
    mutation = CyclicMutation(mutation_probability=1.0, random_state=42)
    child = mutation.mutate(ind)
    assert len(child.genotype) == 5


def test_codon_multiset_preserved():
    """Cyclic rotation is a permutation: sorted values match before and after"""
    genome = [3, 1, 4, 1, 5, 9, 2, 6]
    ind = make_individual(genome.copy())
    mutation = CyclicMutation(mutation_probability=1.0, segment_size=3, random_state=42)
    child = mutation.mutate(ind)
    assert sorted(child.genotype) == sorted(genome)


def test_right_rotation_direction():
    """Each triggered rotation moves the last segment element to the front position"""
    # genome=[1,2,3], segment_size=3, p=1.0 always triggers at pos=0.
    # pos=1 and pos=2 both have end > len, so only pos=0 fires.
    # segment [1,2,3] -> segment[-1:] + segment[:-1] = [3,1,2]
    genome = [1, 2, 3]
    ind = make_individual(genome.copy())
    mutation = CyclicMutation(mutation_probability=1.0, segment_size=3, random_state=42)
    child = mutation.mutate(ind)
    assert child.genotype == [3, 1, 2]


def test_segment_out_of_bounds_is_skipped():
    """Positions where pos + segment_size exceeds genome length are not rotated"""
    # With segment_size=4 and genome length=5, only pos=0 and pos=1 can form full segments.
    genome = [10, 20, 30, 40, 50]
    ind = make_individual(genome.copy())
    mutation = CyclicMutation(mutation_probability=1.0, segment_size=4, random_state=42)
    child = mutation.mutate(ind)
    assert sorted(child.genotype) == sorted(genome)
    assert len(child.genotype) == 5


def test_zero_probability_no_rotation():
    """Probability zero means no rotation occurs; genome is returned unchanged"""
    genome = [1, 2, 3, 4, 5]
    ind = make_individual(genome.copy())
    mutation = CyclicMutation(mutation_probability=0.0, random_state=42)
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_multiset_preserved_across_many_runs():
    """Rotation preserves the codon multiset across varied seeds"""
    genome = [5, 1, 9, 3, 7, 2, 8, 4]
    for seed in range(20):
        ind = make_individual(genome.copy())
        mutation = CyclicMutation(
            mutation_probability=0.5, segment_size=3, random_state=seed
        )
        child = mutation.mutate(ind)
        assert sorted(child.genotype) == sorted(genome)


def test_deterministic_with_fixed_seed():
    """Same random_state produces identical rotation results"""
    ind = make_individual([5, 1, 9, 3, 7, 2, 8])
    m1 = CyclicMutation(mutation_probability=0.5, segment_size=3, random_state=11)
    m2 = CyclicMutation(mutation_probability=0.5, segment_size=3, random_state=11)
    assert m1.mutate(ind).genotype == m2.mutate(ind).genotype


def test_segment_size_two_rotates_pairs():
    """With segment_size=2, each triggered position rotates a pair: [a,b] -> [b,a]"""
    # With p=1.0 and segment_size=2 on [1,2]:
    # pos=0: [1,2] -> [2,1]
    # pos=1: end=3 > 2 -> skip
    genome = [1, 2]
    ind = make_individual(genome.copy())
    mutation = CyclicMutation(mutation_probability=1.0, segment_size=2, random_state=42)
    child = mutation.mutate(ind)
    assert child.genotype == [2, 1]
