import pytest

from finchge.core.individual import Individual
from finchge.operators.mutation import DuplicationMutation


def make_individual(genome):
    return Individual(genotype=genome)


def test_raises_error_without_genotype():
    """Requires genotype; raises ValueError when missing"""
    ind = Individual()
    mutation = DuplicationMutation(mutation_probability=1.0, random_state=42)
    with pytest.raises(ValueError):
        mutation.mutate(ind)


def test_probability_out_of_range_raises():
    """Probability outside [0, 1] raises ValueError"""
    with pytest.raises(ValueError):
        DuplicationMutation(mutation_probability=1.5)


def test_segment_size_zero_raises():
    """segment_size of zero raises ValueError"""
    with pytest.raises(ValueError):
        DuplicationMutation(mutation_probability=0.5, segment_size=0)


def test_segment_size_negative_raises():
    """Negative segment_size raises ValueError"""
    with pytest.raises(ValueError):
        DuplicationMutation(mutation_probability=0.5, segment_size=-1)


def test_returns_new_individual():
    """Mutate returns a new object, not the original"""
    ind = make_individual([1, 2, 3, 4, 5, 6])
    mutation = DuplicationMutation(mutation_probability=1.0, random_state=42)
    assert mutation.mutate(ind) is not ind


def test_does_not_modify_parent():
    """Parent genotype is unchanged after duplication"""
    genome = [1, 2, 3, 4, 5, 6]
    ind = make_individual(genome.copy())
    mutation = DuplicationMutation(mutation_probability=1.0, random_state=42)
    mutation.mutate(ind)
    assert ind.genotype == genome


def test_genome_length_preserved():
    """Duplication overwrites the target in-place; genome length is unchanged"""
    ind = make_individual([10, 20, 30, 40, 50, 60])
    mutation = DuplicationMutation(
        mutation_probability=1.0, segment_size=2, random_state=42
    )
    child = mutation.mutate(ind)
    assert len(child.genotype) == 6


def test_zero_probability_no_mutation():
    """Probability zero returns a clone without any duplication"""
    genome = [1, 2, 3, 4, 5, 6]
    ind = make_individual(genome.copy())
    mutation = DuplicationMutation(
        mutation_probability=0.0, segment_size=2, random_state=42
    )
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_genome_too_short_for_two_non_overlapping_segments_returns_unchanged():
    """When genome length < 2 * segment_size, no valid source+target pair exists"""
    # length=2, segment_size=2 requires at least length=4 for non-overlapping segments
    genome = [10, 20]
    ind = make_individual(genome.copy())
    mutation = DuplicationMutation(
        mutation_probability=1.0, segment_size=2, random_state=42
    )
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_target_region_contains_source_values():
    """After duplication, positions that changed hold values from the original genome"""
    genome = [10, 20, 30, 40, 50, 60]
    ind = make_individual(genome.copy())
    mutation = DuplicationMutation(
        mutation_probability=1.0, segment_size=2, random_state=42
    )
    child = mutation.mutate(ind)

    result = child.genotype
    changed_positions = [i for i in range(len(genome)) if result[i] != genome[i]]

    if changed_positions:
        # Values at changed positions must come from the original genome
        assert all(result[i] in genome for i in changed_positions)


def test_output_values_are_integers():
    """All output values are integers after duplication"""
    ind = make_individual([5, 10, 15, 20, 25, 30])
    mutation = DuplicationMutation(
        mutation_probability=1.0, segment_size=2, random_state=42
    )
    child = mutation.mutate(ind)
    assert all(isinstance(v, int) for v in child.genotype)


def test_genome_length_invariant_across_many_runs():
    """Genome length is stable across repeated mutation calls"""
    for seed in range(20):
        ind = make_individual([10, 20, 30, 40, 50, 60])
        mutation = DuplicationMutation(
            mutation_probability=1.0, segment_size=2, random_state=seed
        )
        child = mutation.mutate(ind)
        assert len(child.genotype) == 6


def test_deterministic_with_fixed_seed():
    """Same random_state produces identical duplication results"""
    ind = make_individual([5, 10, 15, 20, 25, 30])
    m1 = DuplicationMutation(mutation_probability=1.0, segment_size=2, random_state=3)
    m2 = DuplicationMutation(mutation_probability=1.0, segment_size=2, random_state=3)
    assert m1.mutate(ind).genotype == m2.mutate(ind).genotype


def test_source_values_preserved_at_source_position():
    """The source segment in the result still holds its original values"""
    # After duplication: source region is read-only; only target is overwritten.
    genome = [1, 2, 3, 4, 5, 6]
    ind = make_individual(genome.copy())
    mutation = DuplicationMutation(
        mutation_probability=1.0, segment_size=2, random_state=42
    )
    child = mutation.mutate(ind)

    result = child.genotype
    # Find two-element segments that equal their original position in genome
    # Source segment must appear at its original indices (unchanged in result)
    found_intact_source = False
    for start in range(len(genome) - 1):
        if result[start] == genome[start] and result[start + 1] == genome[start + 1]:
            found_intact_source = True
            break
    assert found_intact_source
