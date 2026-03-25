import json
import logging
from typing import Any, Iterable, Optional, TypeAlias


class TreeNode:
    """

    Represents a derivation tree (root node) and its children.
    The Derivation Tree is used during genotype-to-phenotype mapping.

    As TreeNode is used to represent both node and full derivation tree, the properties such as depth,
    max_depth should be used with caution. The property`depth` is the depth of current node,
     while `max_depth` should be used to get the depth of the tree.

    Each node holds a symbol (terminal or non-terminal) and may have children.
    The tree supports JSON and string (CSV-like) serialization.
    Args:
        symbol (str): The symbol associated with this node.
        depth (int, optional): The depth of this node in the tree. Defaults to 0.
    """

    def __init__(self, symbol: str, depth: int = 0):
        self.symbol = symbol
        self.children: list["TreeNode"] = []
        self.parent: Optional["TreeNode"] = None

    # Derived metadata (deterministic)

    @property
    def depth(self) -> int:
        d = 0
        node = self
        while node.parent is not None:
            node = node.parent
            d += 1
        return d

    @property
    def root(self) -> "TreeNode":
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    @property
    def max_depth(self) -> int:
        return max(n.depth for n in self.root.iter_nodes())

    # Internal helpers

    def _rebuild_parents(self) -> None:
        """
        Ensures parent pointers are consistent across subtree.
        Deterministic and idempotent.
        """

        for child in self.children:
            child.parent = self
            child._rebuild_parents()

    # Tree manipulation

    def add_child(self, child: "TreeNode") -> None:
        """
        Adds a child to the current node and updates depth, parent, and root.

        Args:
            child (TreeNode): Node to add as a child.

        Raises:
            TypeError: If `child` is not a TreeNode instance.
        """
        if not isinstance(child, TreeNode):
            raise TypeError("Child must be a TreeNode instance")

        child.parent = self
        self.children.append(child)

    # Phenotype mapping
    def size(self) -> int:
        """
        convert the tree into a phenotype
        """
        nodes = []

        def traverse(node: "TreeNode") -> None:
            if not node.children and not node.symbol.startswith("<"):
                nodes.append(node.symbol)

            for child in node.children:
                traverse(child)

        traverse(self)
        return len(nodes)

    # TODO Check what size means just Terminals or all nodes

    def to_phenotype(self) -> str:
        """
        convert the tree into a phenotype
        """
        result = []

        def traverse(node: "TreeNode") -> None:
            if not node.children and not node.symbol.startswith("<"):
                result.append(node.symbol)

            for child in node.children:
                traverse(child)

        traverse(self)
        return "".join(result)

    # Serialization

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the node and its subtree into a dictionary.

        Returns:
            dict: Dictionary representation of the tree.
        """
        return {
            "symbol": self.symbol,
            "children": [child.to_dict() for child in self.children],
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Serializes the tree into a JSON-formatted string.

        Args:
            indent (int): Indentation for the JSON string.

        Returns:
            str: JSON string representation of the tree.
        """
        try:
            return json.dumps(self.to_dict(), indent=indent)
        except Exception as e:
            logging.exception(f"Failed to save Tree Json {e}")
            raise

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        """
        Constructs a TreeNode from a dictionary.

        Args:
            data (dict): Dictionary with keys 'symbol' and 'children'.

        Returns:
            TreeNode: Reconstructed tree.

        Raises:
            KeyError: If required fields are missing.
        """
        if not all(key in data for key in ["symbol", "children"]):
            raise KeyError("Missing required fields in dictionary")

        node = cls(data["symbol"])

        for child_data in data["children"]:
            child_node = cls.from_dict(child_data)
            node.add_child(child_node)

        return node

    @classmethod
    def from_json(cls, json_str: str) -> "TreeNode":
        """
        Constructs a TreeNode from a JSON string.

        Args:
            json_str (str): JSON representation of the tree.

        Returns:
            TreeNode: Reconstructed tree.

        Raises:
            json.JSONDecodeError: If JSON string is invalid.
        """
        try:
            data = json.loads(json_str)
            node = cls.from_dict(data)
            node._rebuild_parents()
            return node
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON string: {str(e)}", e.doc, e.pos)

    # String serialization

    def to_string(self) -> str:
        """
        Canonical, reversible tree serialization using length-prefixed symbols.

        Format:
            <len>:<symbol>{child1,child2,...}

        Leaf example:
            1:x

        Internal node example:
            3:add{1:x,1:y}

        Returns:
            str: Canonical serialized tree string.
        """
        symbol_part = f"{len(self.symbol)}:{self.symbol}"

        if not self.children:
            return symbol_part

        children_part = ",".join(child.to_string() for child in self.children)
        return f"{symbol_part}{{{children_part}}}"

    @staticmethod
    def from_string(s: str) -> "TreeNode":
        """
        Deserialize a tree from its canonical string representation.

        Args:
            s (str): Serialized tree string.

        Returns:
            TreeNode: Root node of reconstructed tree.
        """
        node, idx = TreeNode._parse_node(s, 0)

        if idx != len(s):
            raise ValueError("Extra trailing characters in serialized TreeNode")

        return node

    @staticmethod
    def _parse_node(s: str, i: int) -> tuple["TreeNode", int]:
        """
        Internal recursive descent parser.

        Args:
            s (str): Full serialized string.
            i (int): Current parsing index.

        Returns:
            (TreeNode, int): Parsed node and new index.
        """
        # parse symbol length
        start = i
        while i < len(s) and s[i].isdigit():
            i += 1

        if start == i or i >= len(s) or s[i] != ":":
            raise ValueError(f"Invalid symbol length at position {start}")

        length = int(s[start:i])
        i += 1  # skip ':'

        # parse symbol
        if i + length > len(s):
            raise ValueError("Symbol length exceeds input size")

        symbol = s[i : i + length]
        i += length

        node = TreeNode(symbol)

        # parse children (optional)
        if i < len(s) and s[i] == "{":
            i += 1  # skip '{'

            while True:
                child, i = TreeNode._parse_node(s, i)
                node.add_child(child)

                if i >= len(s):
                    raise ValueError("Unterminated children block")

                if s[i] == ",":
                    i += 1
                    continue

                if s[i] == "}":
                    i += 1
                    break

                raise ValueError(
                    f"Unexpected character '{s[i]}' while parsing children"
                )

        return node, i

    # Representation

    def __repr__(self) -> str:
        return self._tree_repr()

    def _tree_repr(self, prefix: str = "", is_last: bool = True) -> str:
        """
        Pretty ASCII tree representation.
        """
        connector = "└── " if is_last else "├── "
        lines = [f"{prefix}{connector}{self.symbol}"]

        if self.children:
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(self.children):
                last = i == len(self.children) - 1
                lines.append(child._tree_repr(new_prefix, last))

        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        """
        Compares two TreeNodes for equality.

        Args:
            other: Object to compare with

        Returns:
            True if nodes are equal, False otherwise
        """
        if not isinstance(other, TreeNode):
            return False

        return self.symbol == other.symbol and self.children == other.children

    # Cloning

    def clone(self) -> "TreeNode":
        """
        Deep copy of this node and its entire subtree.
        Parent pointers are reconstructed correctly.
        """
        new_node = TreeNode(self.symbol)

        for child in self.children:
            child_copy = child.clone()
            new_node.add_child(child_copy)

        return new_node

    # Structural operators for crossover and mutation

    def swap_subtree_with(
        self,
        other: "TreeNode",
    ) -> tuple["TreeNode", "TreeNode"]:
        """
        Swap this subtree with another subtree.

        Returns the new roots of both trees.
        """

        p0, p1 = self.parent, other.parent

        if p0 is None and p1 is None:
            return other, self

        if p0 is None and p1 is not None:
            idx1 = p1.children.index(other)
            p1.children[idx1] = self
            self.parent = p1
            other.parent = None
            return other, self

        if p1 is None and p0 is not None:
            idx0 = p0.children.index(self)
            p0.children[idx0] = other
            other.parent = p0
            self.parent = None
            return self, other

        assert p0 is not None  # to silent mypy errors
        assert p1 is not None

        idx0 = p0.children.index(self)
        idx1 = p1.children.index(other)

        p0.children[idx0] = other
        p1.children[idx1] = self

        other.parent = p0
        self.parent = p1

        return self.root, other.root

    def replace_subtree_with(self, new_subtree: "TreeNode") -> "TreeNode":
        """
        Replace this subtree with `new_subtree`.

        Args:
            new_subtree: Root of the subtree that will replace `self`.

        Returns:
            The root of the resulting tree.
        """

        parent = self.parent

        if parent is None:
            new_subtree.parent = None
            return new_subtree

        idx = parent.children.index(self)
        parent.children[idx] = new_subtree
        new_subtree.parent = parent

        return self.root

    # Traversal helpers

    def iter_nodes(self) -> Iterable["TreeNode"]:
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def find_by_symbol(self, symbol: str) -> list["TreeNode"]:
        return [n for n in self.iter_nodes() if n.symbol == symbol]

    def collect_symbols(self, allowed: list[str]) -> list[str]:
        return [n.symbol for n in self.iter_nodes() if n.symbol in allowed]


# Let's also call it DerivationTree, whenever it is a full tree. Just to avoid confusion with the name TreeNode
DerivationTree: TypeAlias = TreeNode
