import json

import pytest

from finchge.grammar.derivation_tree import TreeNode


def _simple_tree() -> TreeNode:
    """Build: add(x, y)  — root with two leaf children."""
    root = TreeNode("add")
    x = TreeNode("x")
    y = TreeNode("y")
    root.add_child(x)
    root.add_child(y)
    return root


def _deep_tree() -> TreeNode:
    """Build: add(mul(x, y), z)  — three levels."""
    root = TreeNode("add")
    mul = TreeNode("mul")
    root.add_child(mul)
    mul.add_child(TreeNode("x"))
    mul.add_child(TreeNode("y"))
    root.add_child(TreeNode("z"))
    return root


# to_phenotype


def test_to_phenotype_joins_terminals_in_order():
    root = _simple_tree()
    assert root.to_phenotype() == "xy"


def test_to_phenotype_deep_tree():
    root = _deep_tree()
    assert root.to_phenotype() == "xyz"


def test_to_phenotype_non_terminal_skipped():
    root = TreeNode("<expr>")
    root.add_child(TreeNode("a"))
    assert root.to_phenotype() == "a"


# depth and max_depth


def test_root_depth_is_one():
    root = TreeNode("root")
    assert root.depth == 1


def test_leaf_depth_is_correct():
    root = _simple_tree()
    leaf = root.children[0]
    assert leaf.depth == 2


def test_max_depth_simple_tree():
    root = _simple_tree()
    assert root.max_depth == 2


def test_max_depth_deep_tree():
    root = _deep_tree()
    assert root.max_depth == 3


# size


def test_size_counts_terminals_only():
    root = _simple_tree()
    # x and y are terminals
    assert root.size() == 2


def test_size_deep_tree():
    root = _deep_tree()
    # x, y, z are terminals
    assert root.size() == 3


#  add_child


def test_add_child_sets_parent():
    root = TreeNode("root")
    child = TreeNode("child")
    root.add_child(child)
    assert child.parent is root


def test_add_child_rejects_non_treenode():
    root = TreeNode("root")
    with pytest.raises(TypeError, match="TreeNode"):
        root.add_child("not_a_node")  # type: ignore


# to_dict, from_dict round-trip


def test_to_dict_from_dict_roundtrip():
    root = _simple_tree()
    d = root.to_dict()
    restored = TreeNode.from_dict(d)
    assert restored == root


def test_to_dict_structure():
    root = TreeNode("root")
    root.add_child(TreeNode("a"))
    d = root.to_dict()
    assert d["symbol"] == "root"
    assert len(d["children"]) == 1
    assert d["children"][0]["symbol"] == "a"


def test_from_dict_missing_key_raises():
    with pytest.raises(KeyError):
        TreeNode.from_dict({"symbol": "root"})  # missing "children"


# to_json, from_json round-trip


def test_to_json_from_json_roundtrip():
    root = _simple_tree()
    json_str = root.to_json()
    restored = TreeNode.from_json(json_str)
    assert restored == root


def test_from_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        TreeNode.from_json("{not valid json")


# to_string, from_string round-trip


def test_to_string_from_string_roundtrip_leaf():
    leaf = TreeNode("x")
    s = leaf.to_string()
    restored = TreeNode.from_string(s)
    assert restored == leaf


def test_to_string_from_string_roundtrip_tree():
    root = _simple_tree()
    s = root.to_string()
    restored = TreeNode.from_string(s)
    assert restored == root


def test_from_string_trailing_chars_raises():
    leaf = TreeNode("x")
    s = leaf.to_string() + "extra"
    with pytest.raises(ValueError, match="Extra trailing"):
        TreeNode.from_string(s)


def test_clone_is_equal_but_independent():
    root = _simple_tree()
    clone = root.clone()
    assert clone == root
    clone.children[0].symbol = "CHANGED"
    assert root.children[0].symbol == "x"


def test_clone_parent_pointers_correct():
    root = _simple_tree()
    clone = root.clone()
    for child in clone.children:
        assert child.parent is clone


def test_equal_trees():
    a = _simple_tree()
    b = _simple_tree()
    assert a == b


def test_unequal_trees_different_symbol():
    a = TreeNode("a")
    b = TreeNode("b")
    assert a != b


def test_unequal_trees_different_children():
    a = _simple_tree()
    b = TreeNode("add")
    b.add_child(TreeNode("z"))
    b.add_child(TreeNode("y"))
    assert a != b
