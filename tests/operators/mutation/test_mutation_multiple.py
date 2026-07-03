import pytest

from finchge.core.individual import Individual
from finchge.operators.mutation import IntFlipMutation, MultipleMutation, SwapMutation


def make_individual(genome):
    return Individual(genotype=genome)


def make_flip(probability=0.5, codon_size=255, seed=1):
    return IntFlipMutation(
        mutation_probability=probability, codon_size=codon_size, random_state=seed
    )


def test_empty_strategies_raises():
    """Empty strategy list raises ValueError at construction"""
    with pytest.raises(ValueError):
        MultipleMutation(strategies=[])


def test_probabilities_length_mismatch_raises():
    """Probabilities list length different from strategies raises ValueError"""
    s = make_flip()
    with pytest.raises(ValueError):
        MultipleMutation(strategies=[s], probabilities=[0.5, 0.5])


def test_probabilities_sum_to_zero_raises():
    """All-zero probabilities raises ValueError"""
    s = make_flip()
    with pytest.raises(ValueError):
        MultipleMutation(strategies=[s, s], probabilities=[0.0, 0.0])


def test_returns_new_individual():
    """Mutate returns a new individual, not the original"""
    ind = make_individual([1, 2, 3, 4])
    mutation = MultipleMutation(strategies=[make_flip()], random_state=42)
    assert mutation.mutate(ind) is not ind


def test_genome_length_preserved():
    """Output genome length matches input regardless of which strategy is selected"""
    ind = make_individual([10, 20, 30, 40, 50])
    mutation = MultipleMutation(
        strategies=[make_flip(0.5, seed=1), make_flip(0.8, seed=2)],
        random_state=42,
    )
    child = mutation.mutate(ind)
    assert len(child.genotype) == 5


def test_probabilities_normalized_on_construction():
    """Custom probabilities are normalized to sum to 1.0"""
    s1 = make_flip(seed=1)
    s2 = make_flip(seed=2)
    mutation = MultipleMutation(strategies=[s1, s2], probabilities=[1.0, 3.0])
    assert abs(sum(mutation.probabilities) - 1.0) < 1e-9
    assert abs(mutation.probabilities[0] - 0.25) < 1e-9
    assert abs(mutation.probabilities[1] - 0.75) < 1e-9


def test_uniform_probabilities_when_none_provided():
    """Equal probability assigned to each strategy when probabilities=None"""
    strategies = [make_flip(seed=i) for i in range(4)]
    mutation = MultipleMutation(strategies=strategies)
    assert all(abs(p - 0.25) < 1e-9 for p in mutation.probabilities)


def test_single_strategy_always_selected():
    """With one strategy, every call applies that strategy"""
    flip = IntFlipMutation(mutation_probability=1.0, codon_size=255, random_state=1)
    mutation = MultipleMutation(strategies=[flip], random_state=42)
    for _ in range(10):
        ind = make_individual([0] * 10)
        child = mutation.mutate(ind)
        assert all(0 <= v <= 255 for v in child.genotype)


def test_biased_weight_selects_favoured_strategy_more_often():
    """Higher weight causes a strategy to be selected proportionally more"""
    genome = [50] * 30

    # strat_a with p=0.0 never changes the genome.
    # strat_b with p=1.0 always changes all codons.
    # Weight 9:1 means strat_a is expected ~90% of the time.
    strat_a = IntFlipMutation(mutation_probability=0.0, codon_size=255, random_state=1)
    strat_b = IntFlipMutation(mutation_probability=1.0, codon_size=255, random_state=2)

    mutation = MultipleMutation(
        strategies=[strat_a, strat_b],
        probabilities=[9.0, 1.0],
        random_state=42,
    )

    unchanged = sum(
        1
        for _ in range(200)
        for ind in [make_individual(genome.copy())]
        if mutation.mutate(ind).genotype == genome
    )
    assert unchanged > 130


def test_output_values_respect_strategy_constraints():
    """With IntFlipMutation as strategy, all output values respect codon_size"""
    flip = IntFlipMutation(mutation_probability=1.0, codon_size=100, random_state=5)
    mutation = MultipleMutation(strategies=[flip], random_state=42)
    ind = make_individual([0] * 20)
    child = mutation.mutate(ind)
    assert all(0 <= v <= 100 for v in child.genotype)


def test_does_not_modify_parent():
    """Parent genotype is unchanged after MultipleMutation"""
    genome = [10, 20, 30, 40]
    ind = make_individual(genome.copy())
    flip = IntFlipMutation(mutation_probability=1.0, codon_size=255, random_state=1)
    mutation = MultipleMutation(strategies=[flip], random_state=42)
    mutation.mutate(ind)
    assert ind.genotype == genome


def test_mixed_strategy_types_produce_valid_output():
    """MultipleMutation works with heterogeneous strategy types"""
    flip = IntFlipMutation(mutation_probability=0.5, codon_size=255, random_state=1)
    swap = SwapMutation(mutation_probability=0.5, random_state=2)
    mutation = MultipleMutation(strategies=[flip, swap], random_state=42)
    ind = make_individual([1, 2, 3, 4, 5, 6])
    child = mutation.mutate(ind)
    assert len(child.genotype) == 6


def test_deterministic_with_fixed_seed():
    """Same random_state and sub-strategy seeds produce identical results"""
    ind = make_individual([10, 20, 30, 40, 50])

    def build():
        return MultipleMutation(
            strategies=[
                IntFlipMutation(
                    mutation_probability=0.5, codon_size=255, random_state=1
                ),
                SwapMutation(mutation_probability=0.5, random_state=2),
            ],
            random_state=99,
        )

    assert build().mutate(ind).genotype == build().mutate(ind).genotype


def test_inject_rng_propagates_to_sub_strategies():
    import random

    flip = IntFlipMutation(mutation_probability=1.0, codon_size=255, random_state=1)
    swap = SwapMutation(mutation_probability=0.5, random_state=2)
    mutation = MultipleMutation(strategies=[flip, swap], random_state=42)

    shared_rng = random.Random(99)
    mutation.inject_rng(shared_rng)

    assert mutation._rng is shared_rng
    assert mutation.strategies[0]._rng is shared_rng
    assert mutation.strategies[1]._rng is shared_rng
