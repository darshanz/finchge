import pytest

from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import PIGrowInitialiser


@pytest.fixture
def grammar():
    gramar_str = """
    <expr> ::= <expr> <op> <expr>
         | ( <expr> )
         | <var>
    <op> ::= + | - | * | /
    <var> ::= x | y | 1 | 2
    """
    return Grammar(grammar_str=gramar_str)


@pytest.fixture
def tree_generator(grammar):
    return TreeGenerator(grammar=grammar, max_tree_depth=10)


def test_pigrow_generates_valid_tree(tree_generator, grammar):
    """
    Generated tree should contain only grammar-valid symbols.
    """
    init = PIGrowInitialiser(init_max_depth=4, population_size=10, random_state=42)
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    for node in tree.iter_nodes():
        assert node.symbol in (grammar.terminals + grammar.non_terminals)


def test_pigrow_respects_max_depth(tree_generator):
    """
    Tree depth must never exceed init_max_depth.
    """
    max_depth = 4

    init = PIGrowInitialiser(
        init_max_depth=max_depth, population_size=10, random_state=42
    )
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    assert tree.max_depth <= max_depth


def test_pigrow_can_generate_shallow_trees(tree_generator):
    """
    Grow initialiser should allow terminals before max depth.
    """
    init = PIGrowInitialiser(init_max_depth=5, population_size=10, random_state=42)
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    assert tree.max_depth <= 5  # grow may terminate early


def test_pigrow_is_position_independent(tree_generator):
    """
    PI-Grow should expand nodes in non depth-first order (stochastic frontier expansion).
    """
    init = PIGrowInitialiser(init_max_depth=4, population_size=10, random_state=42)
    init.set_tree_generator(tree_generator)

    tree1 = TreeNode.from_string(init.initialise().tree)
    tree2 = TreeNode.from_string(init.initialise().tree)

    # Not strict equality check — expect diversity
    assert tree1.to_string() != tree2.to_string()


def test_pigrow_deterministic_with_same_seed(grammar):
    """
    Same RNG seed should reproduce identical trees across runs.
    """
    tg1 = TreeGenerator(grammar, 10)
    tg2 = TreeGenerator(grammar, 10)

    init1 = PIGrowInitialiser(init_max_depth=4, population_size=10, random_state=42)
    init2 = PIGrowInitialiser(init_max_depth=4, population_size=10, random_state=42)

    init1.set_tree_generator(tg1)
    init2.set_tree_generator(tg2)

    tree1 = TreeNode.from_string(init1.initialise().tree)
    tree2 = TreeNode.from_string(init2.initialise().tree)

    assert tree1.to_string() == tree2.to_string()


def test_pigrow_requires_tree_generator():
    """
    Initializer should fail if TreeGenerator not injected.
    """
    init = PIGrowInitialiser(init_max_depth=4, population_size=10, random_state=42)

    with pytest.raises(RuntimeError):
        init.initialise()


def test_pigrow_multiple_initialisations_produce_diverse_trees(tree_generator):
    """
    Multiple calls should produce different structures.
    """
    init = PIGrowInitialiser(init_max_depth=4, population_size=10, random_state=42)
    init.set_tree_generator(tree_generator)

    trees = {init.initialise().tree for _ in range(5)}

    assert len(trees) > 1


def test_pigrow_handles_minimal_depth(tree_generator):
    """Initializer should work for small depths like min_ramp +2."""
    min_ramp = tree_generator.grammar.compute_min_ramp(
        population_size=10, max_init_depth=5
    )
    init = PIGrowInitialiser(
        init_max_depth=min_ramp + 2, population_size=10, random_state=42
    )
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    assert tree.max_depth <= 2


def test_pigrow_only_expands_non_terminals(tree_generator, grammar):
    """
    Terminal nodes must never have children.
    """
    init = PIGrowInitialiser(init_max_depth=4, population_size=10, random_state=42)
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    for node in tree.iter_nodes():
        if node.symbol not in grammar.non_terminals:
            assert not node.children


def test_pigrow_tree_root_matches_start_symbol(tree_generator, grammar):
    """
    Root symbol must always be grammar start symbol.
    """
    init = PIGrowInitialiser(init_max_depth=4, population_size=10, random_state=42)
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    assert tree.symbol == grammar.start_rule
