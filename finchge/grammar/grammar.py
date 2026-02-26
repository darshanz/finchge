from typing import Any, Dict, Optional, Set

from tabulate import tabulate

from finchge.grammar.parser import BNFGrammarParser, GrammarParser
from finchge.utils.display_utils import display_html, highlight_bnf, is_jupyter


class Grammar:
    """
    Represents a grammar and provides methods for mapping genotypes
    to phenotypes using grammatical evolution.

    This class utilizes GrammarParser to parse the grammar (typically, Backus-Naur Form (BNF))

    Args:
        grammar_str (str): The grammar definition as a BNF-formatted string.
        parser (Optional[GrammarParser]): GrammarParser will be used if not provided
    """

    def __init__(
        self,
        grammar_str: str,
        parser: Optional[GrammarParser] = None,
    ) -> None:
        parser = BNFGrammarParser(grammar_str) if not parser else parser
        (
            self.rules_original,
            self.rules,
            self.start_rule,
            self.terminals,
            self.non_terminals,
        ) = parser.parse()

        self._analyzed = False

    def analyze(self) -> None:
        """
        Analyze grammar structure for initialisation and diagnostics.
        Computes recursion, min/max derivation depth, arity, and termination
        feasibility. Results are stored on Rule objects.


        References
         [1] Conor Ryan, J. J. Collins, and Michael O'Neill. 1998. Grammatical Evolution: Evolving Programs for an Arbitrary Language.
         In Proceedings of the First European Workshop on Genetic Programming (EuroGP '98).
         Springer-Verlag, Berlin, Heidelberg, 83-96.

         [2] O’Neill, M. and Ryan, C. (2001,) Grammatical Evolution.
         IEEE Transactions on Evolutionary Computation, 5, 349-358. https://doi.org/10.1109/4235.942529

        """
        if self._analyzed:  # just to avoid repeated calls
            return

        # Arity = number of non-terminal children
        for rule in self.rules.values():
            arities = [
                sum(1 for sym in choice if sym in self.non_terminals)
                for choice in rule.choices
            ]
            rule.min_arity = min(arities)
            rule.max_arity = max(arities)

            # production length (Not widely used, keeping just in case we need it for grammar stats)
            rule.max_rhs_len = max(len(choice) for choice in rule.choices)

        # max arity of a grammar
        self.max_arity = max(rule.max_arity for rule in self.rules.values())

        # Dependency Graph
        deps: dict[str, list[str]] = {nt: [] for nt in self.non_terminals}

        for nt, rule in self.rules.items():
            for choice in rule.choices:
                for sym in choice:
                    if sym in self.non_terminals and sym not in deps[nt]:
                        deps[nt].append(sym)

        #  Recursion detection
        self._detect_recursion(deps)

        #  Min-path (shortest terminating derivation)
        self._compute_min_path()

        # Max-path (longest terminating derivation)
        self._compute_max_path()

        # Grammar-level feasibility
        start_rule = self.rules[self.start_rule]
        self.can_terminate = start_rule.min_path is not None
        self._analyzed = True

    def _detect_recursion(self, deps: Dict[str, list[str]]) -> None:
        """
        Detect recursive non-terminals using deterministic DFS cycle detection.

        A non-terminal is marked recursive if it can derive itself directly
        or indirectly through other non-terminals.

        Args:
            deps:
                Mapping from non-terminal -> ordered list of dependent
                non-terminals appearing in its productions.
        """

        visited: Set[str] = set()
        stack: Set[str] = set()

        def dfs(nt: str) -> bool:
            """
            Returns True if 'nt' participates in recursion.
            """

            # Cycle detected → recursion
            if nt in stack:
                self.rules[nt].recursive = True
                return True

            # Already processed → return stored result
            if nt in visited:
                return self.rules[nt].recursive

            visited.add(nt)
            stack.add(nt)

            is_recursive = False

            for child in deps[nt]:
                if dfs(child):
                    is_recursive = True

            stack.remove(nt)

            # Store final recursion result
            self.rules[nt].recursive = is_recursive
            return is_recursive

        # Run DFS in deterministic non-terminal order
        for nt in self.non_terminals:
            dfs(nt)

    def _compute_min_path(self) -> None:
        # min_path(A) = shortest derivation height from A to terminals (as in Ryan et al., 1998).

        # Initialize
        for rule in self.rules.values():
            rule.min_path = None

        changed = True
        while changed:
            changed = False

            for rule in self.rules.values():
                if rule.min_path is not None:
                    continue

                candidate_depths: list[int] = []

                for choice in rule.choices:
                    depths: list[int] = []

                    valid = True
                    for sym in choice:
                        if sym in self.non_terminals:
                            child = self.rules[sym]
                            if child.min_path is None:
                                valid = False
                                break
                            depths.append(child.min_path)

                    if valid:
                        candidate_depths.append(1 + (max(depths) if depths else 0))

                if candidate_depths:
                    rule.min_path = min(candidate_depths)
                    changed = True

    def _compute_max_path(self) -> None:
        # NOTE:
        # This is an approximate infinite derivation detection.
        # A rule is considered unbounded if it is recursive and can terminate.
        # Full SCC-based analysis could provide more precise classification.

        # rules that can't terminate get max_path = None (infinite)
        for rule in self.rules.values():
            if rule.recursive and rule.min_path is not None and rule.min_path > 0:
                # Recursive + can terminate :: potentially unbounded
                rule.max_path = None
            elif rule.min_path is None or rule.min_path == 0:
                # Cannot terminate at all :: infinite (no valid derivations)
                rule.max_path = None
            else:
                # Finite case - start with minimal value
                rule.max_path = 0

        # Fixed-point iteration
        changed = True
        while changed:
            changed = False

            for rule in self.rules.values():
                # Skip rules already marked as infinite
                if rule.max_path is None:
                    continue

                candidate_depths: list[int] = []

                for choice in rule.choices:
                    depths: list[int] = []
                    valid = True

                    for sym in choice:
                        if sym in self.non_terminals:
                            child = self.rules[sym]
                            if child.max_path is None:
                                # Found infinite child :: this choice leads to infinite derivations
                                valid = False
                                break
                            depths.append(child.max_path)

                    if valid:
                        # All symbols in this choice have finite max paths
                        choice_max = 1 + (max(depths) if depths else 0)
                        candidate_depths.append(choice_max)

                if not candidate_depths:
                    # No finite choices :: rule has infinite max path
                    if rule.max_path is not None:
                        rule.max_path = None
                        changed = True
                else:
                    new_val = max(candidate_depths)
                    if rule.max_path is None or new_val > rule.max_path:
                        rule.max_path = new_val
                        changed = True

    @classmethod
    def from_file(
        cls,
        filename: str,
        parser: Optional[GrammarParser] = None,
    ) -> "Grammar":
        """
        Create a Grammar instance from a file containing BNF rules.
        This method reads the BNF grammar from a text file and initializes the grammar
        with the same parameters as the direct constructor.

        Args:
            filename (str): Path to the file containing BNF grammar rules
            parser (Optional[GrammarParser]): GrammarParser will be used if not provided
        """
        with open(filename, "r") as f:
            grammar_str = f.read()
        return cls(grammar_str=grammar_str, parser=parser)

    def __str__(self) -> str:
        """
        Returns string representation of grammar rules in BNF format.

        Returns:
            str: BNF grammar as a formatted string
        """
        return "\n".join([str(rule) for rule in self.rules.values()])

    def __repr__(self) -> str:
        """
        Returns representation shown in Jupyter when object is evaluated.
        """
        return "\n".join([str(rule) for rule in self.rules.values()])

    def _repr_html_(self) -> str:
        return highlight_bnf("\n".join([str(rule) for rule in self.rules.values()]))

    def _get_ipython(self) -> Optional[Any]:
        try:
            from IPython import get_ipython  # type: ignore

            return get_ipython()  # type: ignore[no-untyped-call]
        except Exception:
            return None

    def describe(self, expanded: bool = True) -> str:
        """
        Returns summary information about the grammar, including rule counts and structure.
        Set expanded=False to display original contracted versions like [a-z] for range if the grammar uses that syntax
        Default setting is to display expanded version.

        Args:
            expanded (bool): flag whether to show expanded version True by default.

        Returns:
           str: A formatted string containing grammar statistics and structure.
        """
        self.analyze()

        grammar_string = (
            "\n".join([str(rule) for rule in self.rules.values()])
            if expanded
            else "\n".join([str(rule) for rule in self.rules_original.values()])
        )

        # start_rule and recursive_rules for showing grammar analysis results
        start_rule = self.rules[self.start_rule]
        recursive_rules = [r.symbol for r in self.rules.values() if r.recursive]

        # handle display for jupyter notebook
        in_notebook: bool = is_jupyter()

        info_str: str = ""

        if in_notebook:
            tablefmt = "html"
            info_str = "Grammar:<br />"
            info_str += highlight_bnf(grammar_string)
            info_str += "<br /><br />"
        else:
            tablefmt = "simple"
            info_str = "Grammar:\n"
            info_str += "===========GRAMMAR============\n"
            info_str += grammar_string
            info_str += "\n\n"

        table_data: list[list[Any]] = [
            ["Number of Rules", len(self.rules)],
            ["Start Rule", self.start_rule],
            ["Number of Terminals", len(self.terminals)],
            ["Number of Non-Terminals", len(self.non_terminals)],
            ["Max Arity", self.max_arity],
            ["Can Terminate", self.can_terminate],
            ["Start Min Depth", start_rule.min_path],
            [
                "Start Max Depth",
                "Inf" if start_rule.max_path is None else start_rule.max_path,
            ],
            ["Can Terminate", self.can_terminate],
            ["Recursive Rules", len(recursive_rules)],
        ]
        headers = ["Description", "Value"]
        info_str += tabulate(table_data, headers=headers, tablefmt=tablefmt)
        info_str += "\n\n"

        if in_notebook:
            display_html(info_str)
            return ""
        else:  # terminal
            return info_str
