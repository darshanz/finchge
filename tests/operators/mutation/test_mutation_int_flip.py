import pytest

from finchge.core.individual import Individual
from finchge.operators.mutation import IntFlipMutation


def make_individual(genome, used_genome=None, invalid=False):
    return Individual(genotype=genome, used_genome=used_genome, invalid=invalid)


def test_raises_error_without_genotype():
    """Requires genotype; raises ValueError when missing"""
    ind = Individual()
    mutation = IntFlipMutation(
        mutation_probability=0.5, codon_size=255, random_state=42
    )
    with pytest.raises(ValueError):
        mutation.mutate(ind)


def test_invalid_mode_raises():
    """Unknown mode string raises ValueError at construction"""
    with pytest.raises(ValueError):
        IntFlipMutation(mutation_probability=0.5, codon_size=255, mode="bad_mode")


def test_probability_out_of_range_raises():
    """Probability outside [0, 1] raises ValueError"""
    with pytest.raises(ValueError):
        IntFlipMutation(mutation_probability=1.5, codon_size=255)


def test_negative_mutation_events_raises():
    """Negative mutation_events in per_ind mode raises ValueError"""
    with pytest.raises(ValueError):
        IntFlipMutation(
            mutation_probability=0.5,
            codon_size=255,
            mode="per_ind",
            mutation_events=-1,
        )


def test_returns_new_individual():
    """Mutate returns a new object, not the original"""
    ind = make_individual([10, 20, 30, 40])
    mutation = IntFlipMutation(
        mutation_probability=0.5, codon_size=255, random_state=42
    )
    assert mutation.mutate(ind) is not ind


def test_does_not_modify_parent():
    """Parent genotype is unchanged after mutation"""
    genome = [10, 20, 30, 40]
    ind = make_individual(genome.copy())
    mutation = IntFlipMutation(
        mutation_probability=1.0, codon_size=255, random_state=42
    )
    mutation.mutate(ind)
    assert ind.genotype == genome


def test_genome_length_preserved():
    """Genome length is unchanged after int-flip mutation"""
    ind = make_individual([1, 2, 3, 4, 5])
    mutation = IntFlipMutation(
        mutation_probability=1.0, codon_size=255, random_state=42
    )
    child = mutation.mutate(ind)
    assert len(child.genotype) == 5


def test_values_within_codon_range():
    """All mutated codons are within [0, codon_size]"""
    ind = make_individual([0] * 100)
    codon_size = 50
    mutation = IntFlipMutation(
        mutation_probability=1.0, codon_size=codon_size, random_state=42
    )
    child = mutation.mutate(ind)
    assert all(0 <= v <= codon_size for v in child.genotype)


def test_zero_probability_no_change():
    """Probability zero produces a clone of the parent"""
    genome = [10, 20, 30, 40]
    ind = make_individual(genome.copy())
    mutation = IntFlipMutation(
        mutation_probability=0.0, codon_size=255, random_state=42
    )
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_full_probability_values_in_range():
    """Probability of 1.0 flips every codon; all results remain within valid range"""
    ind = make_individual([0] * 50)
    mutation = IntFlipMutation(
        mutation_probability=1.0, codon_size=255, random_state=42
    )
    child = mutation.mutate(ind)
    assert all(0 <= v <= 255 for v in child.genotype)
    assert len(child.genotype) == 50


def test_within_used_restricts_mutation_to_used_region():
    """With within_used=True, codons beyond used_genome length are not touched"""
    genome = [1, 2, 100, 200]
    ind = Individual(genotype=genome.copy(), used_genome=[1, 2])
    mutation = IntFlipMutation(
        mutation_probability=1.0,
        codon_size=255,
        within_used=True,
        random_state=42,
    )
    child = mutation.mutate(ind)
    assert child.genotype[2:] == [100, 200]


def test_within_used_false_mutates_full_genome():
    """With within_used=False, all codons including unused tail are eligible"""
    genome = [0] * 8
    ind = Individual(genotype=genome.copy(), used_genome=[0, 0])
    mutation = IntFlipMutation(
        mutation_probability=1.0,
        codon_size=255,
        within_used=False,
        random_state=42,
    )
    child = mutation.mutate(ind)
    assert len(child.genotype) == 8
    assert all(0 <= v <= 255 for v in child.genotype)


def test_invalid_individual_with_within_used_falls_back_to_full_genome():
    """Invalid individuals fall back to full genome mutation regardless of within_used"""
    genome = [1, 2, 100, 200]
    ind = Individual(genotype=genome.copy(), used_genome=[1, 2], invalid=True)
    mutation = IntFlipMutation(
        mutation_probability=1.0,
        codon_size=255,
        within_used=True,
        random_state=42,
    )
    child = mutation.mutate(ind)
    # Full genome eligible; tail values 100, 200 can change
    assert len(child.genotype) == 4


def test_deterministic_with_fixed_seed():
    """Same random_state produces identical mutation results"""
    ind = make_individual([10, 20, 30, 40, 50])
    m1 = IntFlipMutation(mutation_probability=0.5, codon_size=255, random_state=7)
    m2 = IntFlipMutation(mutation_probability=0.5, codon_size=255, random_state=7)
    assert m1.mutate(ind).genotype == m2.mutate(ind).genotype


def test_per_ind_mode_returns_valid_individual():
    """per_ind mode returns an individual with correct genome length and valid values"""
    ind = make_individual([10, 20, 30, 40, 50])
    mutation = IntFlipMutation(
        mutation_probability=0.5,
        codon_size=255,
        mode="per_ind",
        mutation_events=3,
        random_state=42,
    )
    child = mutation.mutate(ind)
    assert len(child.genotype) == 5
    assert all(0 <= v <= 255 for v in child.genotype)


def test_per_ind_zero_events_no_change():
    """per_ind mode with mutation_events=0 produces a clone of the parent"""
    genome = [10, 20, 30, 40]
    ind = make_individual(genome.copy())
    mutation = IntFlipMutation(
        mutation_probability=0.5,
        codon_size=255,
        mode="per_ind",
        mutation_events=0,
        random_state=42,
    )
    child = mutation.mutate(ind)
    assert child.genotype == genome


def test_default_probability_none_applies_without_error():
    """mutation_probability=None uses 1/L per-codon rate and runs without error"""
    ind = make_individual([5, 10, 15, 20, 25])
    mutation = IntFlipMutation(
        mutation_probability=None, codon_size=255, random_state=42
    )
    child = mutation.mutate(ind)
    assert len(child.genotype) == 5
    assert all(0 <= v <= 255 for v in child.genotype)


def test_mutation_varies_across_seeds():
    """Different seeds produce different mutation outcomes"""
    results = set()
    for seed in range(30):
        ind = make_individual([0] * 10)
        mutation = IntFlipMutation(
            mutation_probability=1.0, codon_size=255, random_state=seed
        )
        results.add(tuple(mutation.mutate(ind).genotype))
    assert len(results) > 1
