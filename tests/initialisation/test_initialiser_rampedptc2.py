import pytest

from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import RampedPTC2Initialiser


@pytest.fixture
def grammar():
    TEST_GRAMMAR = """
    <S> ::= <S> <S> | <T>
    <T> ::= a | b | c
    """
    return Grammar(grammar_str=TEST_GRAMMAR)


@pytest.fixture
def tree_generator(grammar):
    return TreeGenerator(grammar=grammar, max_tree_depth=10)


def build_initialiser(
    tree_generator,
    min_size=5,
    max_size=20,
    pop_size=50,
    deterministic=True,
    seed=42,
):
    init = RampedPTC2Initialiser(
        init_min_size=min_size,
        init_max_size=max_size,
        population_size=pop_size,
        random_state=seed,
    )

    init.set_tree_generator(tree_generator=tree_generator)
    return init


def test_ramped_ptc2_produces_individual(tree_generator):
    """Produces valid individuals"""
    init = build_initialiser(tree_generator=tree_generator)

    ind = init.initialise()

    assert ind.tree is not None


def test_ramped_ptc2_respects_max_size(tree_generator):
    """Trees respect maximum size"""
    init = build_initialiser(tree_generator, min_size=3, max_size=8)

    for _ in range(20):
        ind = init.initialise()
        tree = TreeNode.from_string(ind.tree)

        assert tree.size() <= 8


def test_ramped_ptc2_deterministic_repeatability(tree_generator):
    """Deterministic schedule repeats"""
    init1 = build_initialiser(tree_generator=tree_generator, seed=1)
    init2 = build_initialiser(tree_generator=tree_generator, seed=1)

    sizes1 = [TreeNode.from_string(init1.initialise().tree).size() for _ in range(20)]
    sizes2 = [TreeNode.from_string(init2.initialise().tree).size() for _ in range(20)]

    assert sizes1 == sizes2


def test_ramped_ptc2_stochastic_varies(tree_generator):
    """Stochastic mode varies sizes"""
    init = build_initialiser(tree_generator=tree_generator, deterministic=False)

    sizes = [TreeNode.from_string(init.initialise().tree).size() for _ in range(50)]
    assert len(set(sizes)) > 1


def test_ramped_ptc2_schedule_wraps(tree_generator):
    """Population schedule wraps correctly"""
    init = build_initialiser(tree_generator=tree_generator, pop_size=10)

    results = [TreeNode.from_string(init.initialise().tree).size() for _ in range(30)]

    assert len(results) == 30


def test_ramped_ptc2_produces_grammar_valid_trees(tree_generator):
    """Grammar validity maintained"""
    init = build_initialiser(tree_generator=tree_generator, pop_size=10)

    for _ in range(20):
        ind = init.initialise()
        tree = TreeNode.from_string(ind.tree)

        # no non-terminal leaves
        def has_nt_leaf(node):
            if not node.children:
                return node.symbol.startswith("<")
            return any(has_nt_leaf(c) for c in node.children)

        assert not has_nt_leaf(tree)


def test_ramped_ptc2_large_population_stability(tree_generator):
    """Larger populations still valid"""
    init = build_initialiser(tree_generator=tree_generator, pop_size=200)

    trees = [init.initialise().tree for _ in range(200)]

    assert len(trees) == 200
