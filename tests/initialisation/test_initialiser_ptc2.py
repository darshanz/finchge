import pytest

from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.tree_generator import TreeGenerator
from finchge.initialisation.initialisers import PTC2Initialiser


@pytest.fixture
def grammar():
    gramar_str = """
     <expr> ::= <expr> <op> <expr> | <var>
    <op> ::= + | - | * | /
    <var> ::= x | y | 1 | 2
    """
    return Grammar(grammar_str=gramar_str)


@pytest.fixture
def tree_generator(grammar):
    return TreeGenerator(grammar=grammar, max_tree_depth=10)


def test_ptc2_generates_individual(tree_generator):
    """Generates Individual"""
    init = PTC2Initialiser(target_size=10, random_state=42)
    init.set_tree_generator(tree_generator)

    ind = init.initialise()

    assert ind.tree is not None


def test_ptc2_produces_complete_trees(tree_generator, grammar):
    """
    PTC2 should always produce complete trees (no unexpanded non-terminals).
    """
    for target in [1, 5, 10, 20]:
        init = PTC2Initialiser(target_size=target, random_state=42)
        init.set_tree_generator(tree_generator)

        for _ in range(5):  # Multiple trials
            tree = TreeNode.from_string(init.initialise().tree)

            # Check all non-terminal nodes have children (were expanded)
            for node in tree.iter_nodes():
                if node.symbol in grammar.non_terminals:
                    assert node.children, f"Non-terminal {node.symbol} wasn't expanded"

            # Check all leaves are terminals
            for node in tree.iter_nodes():
                if node.symbol in grammar.terminals:
                    assert not node.children


def test_ptc2_size_variation(tree_generator, grammar):
    """
    PTC2 should produce varied size with same target.
    """
    target = 10
    init = PTC2Initialiser(target_size=target, random_state=42)  # No fixed seed
    init.set_tree_generator(tree_generator)
    sizes = []

    for _ in range(20):
        tree = TreeNode.from_string(init.initialise().tree)

        sizes.append(tree.size())

    # Should have some variation (not all trees identical)
    assert max(sizes) - min(sizes) > 0


def test_ptc2_small_targets(grammar, tree_generator):
    """
    PTC2 should handle small target sizes gracefully.
    """
    for target in [1, 2, 3]:
        init = PTC2Initialiser(target_size=target, random_state=42)
        init.set_tree_generator(tree_generator)

        tree = TreeNode.from_string(init.initialise().tree)

        # Tree should be valid and complete
        for node in tree.iter_nodes():
            if node.symbol in grammar.terminals:
                assert not node.children

        # With target=0, if start symbol is non-terminal,
        # we still need at least 1 expansion
        if target == 0 and grammar.start_rule in grammar.non_terminals:
            # Should have at least some nodes
            assert tree.size() > 1


def test_ptc2_deterministic(tree_generator):
    """
    Deterministic with same seed
    """
    init1 = PTC2Initialiser(target_size=25, random_state=55)
    init2 = PTC2Initialiser(target_size=25, random_state=55)

    init1.set_tree_generator(tree_generator)
    init2.set_tree_generator(tree_generator)

    assert init1.initialise().tree == init2.initialise().tree


def test_ptc2_small_target(tree_generator):
    """Handles Small Target"""
    init = PTC2Initialiser(target_size=2, random_state=4)
    init.set_tree_generator(tree_generator)

    ind = init.initialise()

    assert TreeNode.from_string(ind.tree).size() >= 1


def test_ptc2_population_diversity(tree_generator):
    """Population Diversity"""
    init = PTC2Initialiser(target_size=12, random_state=1)
    init.set_tree_generator(tree_generator)

    trees = {init.initialise().tree for _ in range(6)}

    assert len(trees) > 1


def test_ptc2_requires_generator():
    """Raises Without Generator"""
    init = PTC2Initialiser(target_size=10)

    with pytest.raises(RuntimeError):
        init.initialise()


def test_ptc2_depth_reasonable(tree_generator):
    """Tree Depth Does Not Explode"""
    init = PTC2Initialiser(target_size=30, random_state=9)
    init.set_tree_generator(tree_generator)

    tree = TreeNode.from_string(init.initialise().tree)

    assert tree.depth < 50


def test_ptc2_multiple_calls(tree_generator):
    """Multiple Calls Stable"""
    init = PTC2Initialiser(target_size=15, random_state=8)
    init.set_tree_generator(tree_generator)

    for _ in range(5):
        assert init.initialise().tree is not None


def test_ptc2_target_size_small(tree_generator):
    """Target Size very small 1 Edge Case"""
    init = PTC2Initialiser(target_size=1, random_state=2)
    init.set_tree_generator(tree_generator)

    tree = init.initialise().tree

    assert TreeNode.from_string(tree).size() >= 1
