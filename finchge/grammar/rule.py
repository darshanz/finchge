from typing import Optional

from finchge.utils.display_utils import highlight_bnf


class Rule:
    """
    Represents a single production rule in a BNF grammar.

    Each rule contains a non-terminal symbol (e.g., "<expr>") and a list of
    possible expansions (choices) for that symbol.

    Args:
        symbol (str): The non-terminal symbol on the left-hand side (LHS) of the rule.
        choices (list[list[str]]): A list of possible right-hand side (RHS) expansions.
            Each choice is a list of symbols (either terminals or non-terminals).
    """

    def __init__(self, symbol: str, choices: list[list[str]]) -> None:
        self.symbol = symbol
        self.choices = choices

        # Analysis fields (populated by grammar.analyze)
        self.recursive: bool = False
        self.min_path: Optional[int] = None
        self.max_path: Optional[int] = None
        self.min_arity: int = 0
        self.max_arity: int = 0
        self.max_rhs_len: int = 0

    def __repr__(self) -> str:
        """
        String representation of the node showing its symbol and choices.
        """
        choices_str = " | ".join([" ".join(choice) for choice in self.choices])
        return f"{self.symbol} ::= {choices_str}"

    def __str__(self) -> str:
        """
        String representation of the node showing its symbol and choices.
        """
        choices_str = " | ".join([" ".join(choice) for choice in self.choices])
        return f"{self.symbol} ::= {choices_str}"

    def _repr_html_(self) -> str:
        """
        String representation of the node showing its symbol and choices.
        """
        choices_str = " | ".join([" ".join(choice) for choice in self.choices])
        rule_string = f"{self.symbol} ::= {choices_str}"
        return highlight_bnf(rule_string)
