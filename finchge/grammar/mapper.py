import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

from tabulate import tabulate

from finchge.grammar import Grammar
from finchge.grammar.derivation_tree import TreeNode
from finchge.grammar.repair_strategy import RepairStrategy
from finchge.utils.random_mixin import RandomStateMixin


@dataclass(frozen=True)
class MappingResult:
    """Result of genotype to phenotype mapping."""

    phenotype: str | None
    used_genome: list[int]
    used_codon_count: int
    invalid: bool
    tree: TreeNode
    tree_str: str

    @property
    def is_valid(self) -> bool:
        return not self.invalid and self.phenotype is not None


class GenotypeMapper(RandomStateMixin):
    """
    Bidirectional genotype-phenotype codec for Grammatical Evolution.

    `GenotypeMapper` implements the core mapping logic of Grammatical Evolution (GE),
    providing:

    - Decoding (map): genotype >>> derivation tree >>> phenotype
    - Encoding (reverse_map): derivation tree >>> genotype (that reproduces the same tree)

    Follows classic GE semantics:
        - Stack-based depth-first expansion
        - Modulo-based production selection
        - Leftmost-first expansion (children pushed reversed onto a LIFO stack)
        - Codon wrapping with a configurable limit
        - Explicit recursion depth control

    This class is grammar-aware but grammar-agnostic: all syntactic structure
    comes from the provided `Grammar` instance.


    Attributes:
        grammar (Grammar):
            Grammar defining the genotype-to-phenotype mapping rules.
        rules (dict[str, Rule]):
            Grammar production rules indexed by non-terminal symbols.
        non_terminals (list[str]):
            Set of grammar non-terminal symbols.
        start_rule (str):
            Start symbol used for derivation.
        max_recursion_depth (int):
            Maximum allowed recursion depth during decoding.
        max_wraps (int):
            Maximum number of genotype wraps allowed during decoding.
        repair_strategy (Optional[RepairStrategy]):
            Optional phenotype repair strategy applied after decoding.
    """

    def __init__(
        self,
        *,
        grammar: Grammar,
        max_tree_depth: Optional[int] = None,
        max_recursion_depth: Optional[int] = None,
        max_wraps: int = 6,
        repair_strategy: Optional[RepairStrategy] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize a GenotypeMapper with grammar and mapping constraints.

        Args:
            grammar (Grammar):
                Grammar defining production rules and non-terminals used for mapping.
            max_tree_depth (Optional[int]):
                Max depth limit for resulting trees.
            max_recursion_depth (int, optional):
                Maximum depth allowed during derivation tree expansion.
                Mapping is marked invalid if exceeded. Defaults to 20.
            max_wraps (int, optional):
                Maximum number of times the genotype may be wrapped when codons
                are exhausted. Mapping is marked invalid if exceeded.
                Defaults to 6.
            repair_strategy (Optional[RepairStrategy], optional):
                Optional strategy to repair or post-process the phenotype
                after successful decoding. Defaults to None.
            random_state (int): random state
        """

        super().__init__(random_state=random_state)
        self.grammar = grammar
        self.rules = grammar.rules
        self.non_terminals = grammar.non_terminals
        self.start_rule: str | None = getattr(grammar, "start_rule", None)
        self.max_recursion_depth = max_recursion_depth
        self.max_tree_depth = max_tree_depth
        self.max_wraps = max_wraps
        self.repair_strategy = repair_strategy

        if (
            self.max_tree_depth is not None
            and self.max_recursion_depth is not None
            and self.max_recursion_depth <= self.max_tree_depth
        ):
            logging.warning(
                "GenotypeMapper: max_recursion_depth (%s) <= max_tree_depth (%s). "
                "Recursion depth may become the effective mapping limit before the "
                "tree-depth constraint.",
                self.max_recursion_depth,
                self.max_tree_depth,
            )

        elif self.max_tree_depth is None and self.max_recursion_depth is not None:
            logging.warning(
                "GenotypeMapper: max_recursion_depth is set without max_tree_depth. "
                "Mapping is constrained by recursion depth only."
            )

    def map(self, genotype: list[int]) -> MappingResult:
        """
        Decode a genotype into a derivation tree and phenotype using
        standard Grammatical Evolution (GE) mapping. This implementation performs stack-based depth-first expansion
        using leftmost derivation semantics.

        Follows leftmost derivation semantics:
            - Children are attached to the tree in left-to-right order
            - Children are pushed onto the stack in reverse order ensuring leftmost child expands first

        Codons are consumed sequentially and mapped to grammar productions
        using modulo selection.

        Wrapping is supported when the genotype is exhausted, up to
        `max_wraps` times.

        Mapping is aborted and marked invalid if:
            - wrapping limit exceeds
            - recursion depth exceeds `max_recursion_depth`
            - the final tree contains unresolved non-terminals

        Args:
            genotype (list[int]):
                List of integer codons representing the genotype.

        Returns:
            MappingResult:
                Object containing:
                    ```
                        - phenotype (str): Generated phenotype string.
                        - used_genome (list[int]): Codons actually consumed.
                        - used_codon_count (int): Number of codons consumed.
                        - invalid (bool): Whether mapping failed.
                        - tree (TreeNode): Root of the derivation tree.
                        - tree_json (str): Serialized tree representation.
                    ```
        """

        if not self.start_rule:
            raise ValueError("Start Rule must exist.")
        # Initialize derivation tree
        root = TreeNode(self.start_rule)

        # Stack holds (node, depth)
        stack: list[tuple[TreeNode, int]] = [(root, 1)]

        genotype_len = len(genotype)
        current_codon_index = 0
        used_codons_count = 0
        used_genome: list[int] = []
        wraps = 0

        # Empty genotype is always invalid
        if genotype_len == 0:
            return self._invalid_result(root, used_genome, used_codons_count)

        # Mapping loop
        while stack:
            current_node, depth = stack.pop()
            current_symbol = current_node.symbol

            # Enforce recursion depth constraint
            if (
                self.max_recursion_depth is not None
                and depth > self.max_recursion_depth
            ):
                return self._invalid_result(root, used_genome, used_codons_count)

            # Max tree depth limit
            if self.max_tree_depth is not None and depth > self.max_tree_depth:
                return self._invalid_result(root, used_genome, used_codons_count)

            # Terminal symbol - nothing to expand
            if current_symbol not in self.rules:
                continue

            # A non-terminal exactly at the max tree depth would lead to tree with depth higher than the max_tree_depth
            # So, just discard such trees, so that max_tree_depth cap is respected.
            if self.max_tree_depth is not None and depth >= self.max_tree_depth:
                return self._invalid_result(root, used_genome, used_codons_count)

            # if we have exhausted the genome and still need to expand a non-terminal,
            # wrap back to the start and count a wrap.
            if current_codon_index > 0 and current_codon_index % genotype_len == 0:
                wraps += 1
                if wraps > self.max_wraps:
                    return self._invalid_result(root, used_genome, used_codons_count)
                current_codon_index = 0

            # Retrieve production rule (fail-fast if inconsistent grammar)
            rule = self.rules[current_symbol]

            if not rule.choices:
                return self._invalid_result(root, used_genome, used_codons_count)

            # Select production using modulo selection
            codon = genotype[current_codon_index]
            choice_index = codon % len(rule.choices)

            used_genome.append(codon)
            used_codons_count += 1
            current_codon_index += 1

            selected_production = rule.choices[choice_index]

            children: list[TreeNode] = []

            for symbol in selected_production:
                child_node = TreeNode(symbol)
                current_node.add_child(child_node)
                children.append(child_node)

            # Stack is LIFO - push reversed order to expand leftmost first
            for child in reversed(children):
                stack.append((child, depth + 1))

        # Validate final tree
        if self._has_nonterminal_leaves(root):
            return self._invalid_result(root, used_genome, used_codons_count)

        # Convert tree to phenotype
        phenotype = self._tree_to_string(root)

        # Apply optional repair strategy
        if self.repair_strategy is not None:
            phenotype = self.repair_strategy.repair(phenotype)

        return MappingResult(
            phenotype=phenotype,
            used_genome=used_genome,
            used_codon_count=used_codons_count,
            invalid=False,
            tree=root,
            tree_str=root.to_string(),
        )

    # Encode: tree to genotype
    def reverse_map(
        self,
        tree: Union[TreeNode, str],
        *,
        codon_size: int = 127,
        pad_to_length: Optional[int] = None,
        pad_mode: str = "random",
    ) -> list[int]:
        """
        Encode a derivation tree into a genotype that reproduces the same tree.

        This method traverses the tree in the same order that `map()` consumes
        codons: leftmost depth-first expansion of non-terminals.

        Args:
            tree:
                TreeNode root or JSON serialized tree.

            codon_size:
                Maximum codon value (inclusive).

            pad_to_length:
                Optional genome length padding.

            pad_mode:
                "zeros" or "random".

        Returns:
            list[int]: Genotype reproducing this tree.
        """

        root = TreeNode.from_string(tree) if isinstance(tree, str) else tree
        genome: list[int] = []

        # Traverse nodes in exact decode order used by map()
        stack = [root]

        while stack:
            node = stack.pop()

            # Only non-terminals consume codons
            if node.symbol in self.rules:
                rule = self.rules[node.symbol]

                rhs = [c.symbol for c in node.children]

                choice_index = self._find_production_index(rule.choices, rhs)

                genome.append(
                    # prepare the condons based on random integers withing codon size
                    self._pick_codon_for_choice(
                        choice_index,
                        len(rule.choices),
                        codon_size,
                    )
                )

            # Push children reversed to match decoder stack behaviour
            for child in reversed(node.children):
                stack.append(child)

        # genome padding
        if pad_to_length is not None:
            genome = self._pad_genome(
                genome=genome,
                target_len=pad_to_length,
                codon_size=codon_size,
                pad_mode=pad_mode,
            )

        return genome

    def _invalid_result(
        self,
        root: TreeNode,
        used_genome: list[int],
        used_codon_count: int,
    ) -> MappingResult:
        return MappingResult(
            phenotype=None,
            used_genome=used_genome,
            used_codon_count=used_codon_count,
            invalid=True,
            tree=root,
            tree_str=root.to_string(),
        )

    def _tree_to_string(self, node: TreeNode) -> str:
        if not node.children:
            return node.symbol
        return "".join(self._tree_to_string(child) for child in node.children)

    def _has_nonterminal_leaves(self, node: TreeNode) -> bool:
        # During recursion if we stop based on max recursion depth or max wraps
        # Some trees may remain unfinished. non-terminals may be in the leaf nodes.
        # so discard them
        if not node.children:
            return node.symbol in self.non_terminals
        return any(self._has_nonterminal_leaves(c) for c in node.children)

    def _iter_nonterminals_in_decode_order(self, root: TreeNode) -> Iterable[TreeNode]:
        stack: list[TreeNode] = [root]

        while stack:
            node = stack.pop()

            if node.symbol in self.rules:
                yield node

            for child in reversed(node.children):
                stack.append(child)

    def _find_production_index(
        self, productions: Sequence[Sequence[str]], rhs: Sequence[str]
    ) -> int:
        rhs_list = list(rhs)
        for i, prod in enumerate(productions):
            if list(prod) == rhs_list:
                return i
        raise ValueError(
            f"Tree RHS {rhs_list} does not match any grammar production among: {productions}"
        )

    def _pick_codon_for_choice(
        self,
        chosen_idx: int,
        num_choices: int,
        codon_size: int,
    ) -> int:
        if num_choices <= 1:
            return 0

        max_k = (codon_size - chosen_idx) // num_choices
        if max_k < 0:
            raise ValueError("No valid codon.")
        # isntead of always having smallest possible codons we can randomize
        k = self.rng.randint(0, max_k)
        return chosen_idx + k * num_choices

    def _pad_genome(
        self,
        genome: list[int],
        target_len: int,
        codon_size: int,
        pad_mode: str,
    ) -> list[int]:
        """
        Pad or truncate genome to target length.
        """
        cur_len = len(genome)

        if cur_len >= target_len:
            return genome

        pad_len = target_len - cur_len

        if pad_mode == "zeros":
            padding = [0] * pad_len
        elif pad_mode == "random":
            padding = [self.rng.randint(0, codon_size) for _ in range(pad_len)]
        else:
            raise ValueError(f"Unknown pad_mode={pad_mode!r}")
        return genome + padding

    def __str__(self) -> str:
        """
        Returns string representation of grammar rules in BNF format.

        Returns:
            str: BNF grammar as a formatted string
        """
        mapper_str = (
            f"GenotypeMapper\n"
            f"max_recursion_depth={self.max_recursion_depth}\n"
            f"max_wraps={self.max_wraps}"
        )
        return mapper_str

    def __repr__(self) -> str:
        """
        Returns representation shown in Jupyter when object is evaluated.
        """
        return self.__str__()

    def _get_ipython(self) -> Optional[Any]:
        try:
            from IPython import get_ipython  # type: ignore

            return get_ipython()  # type: ignore[no-untyped-call]
        except Exception:
            return None

    def _repr_html_(self) -> str:
        in_jupyter: bool = False
        ip = self._get_ipython()
        in_jupyter = ip is not None and hasattr(ip, "config")
        tablefmt = "simple"
        if in_jupyter:
            tablefmt = "html"

        info_str: str = ""

        if in_jupyter:
            info_str = "GenotypeMapper:<br />"
        else:
            info_str = "GenotypeMapper:\n"
        table_data: list[list[str | int | None]] = [
            ["Max Recursion Depth", self.max_recursion_depth],
            ["Max Wraps", self.max_wraps],
        ]
        headers = ["Attribute", "Value"]
        info_str += tabulate(table_data, headers=headers, tablefmt=tablefmt)
        return info_str
