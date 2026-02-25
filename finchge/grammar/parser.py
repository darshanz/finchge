import re
from abc import ABC, abstractmethod
from typing import Optional

from finchge.grammar import Rule
from finchge.grammar.range_handlers import RangeHandlerRegistry


class GrammarParser(ABC):
    """
    Base class for all grammar parsers.

    Any custom parser must implement the parse() method which returns
    a standardized tuple containing grammar components.
    """

    @abstractmethod
    def parse(
        self,
    ) -> tuple[dict[str, "Rule"], dict[str, "Rule"], str, list[str], list[str]]:
        """
        Parse input data and return grammar components.

        Returns:
            tuple containing:
                rules_contracted (dict[str, Rule]): Mapping of non-terminal symbols to Rule objects (original written form withoutrange expansion).
                rules (dict[str, Rule]): Mapping of non-terminal symbols to Rule objects in Expanded (range notation) form.
                start_rule (str): The first non-terminal rule, considered the start rule.
                terminals (set[str]): Set of terminal symbols.
                non_terminals (set[str]): Set of non-terminal symbols.
        """
        pass


def remove_comments(lines: list[str]) -> list[str]:
    """
    Remove comments and empty lines.
    """
    COMMENT_CHAR = "#"
    cleaned = []
    for line in lines:
        # Strip whitespace
        stripped = line.strip()
        if not stripped:  # empty lines
            cleaned.append("")

        # Remove comments
        if COMMENT_CHAR in stripped:
            stripped = stripped.split(COMMENT_CHAR, 1)[0].rstrip()
            if not stripped:  # Whole line was a comment
                continue

        cleaned.append(stripped)
    return cleaned


def consolidate_multiline_rules(lines: list[str]) -> list[str]:
    """
    If same rule sreads across multiple lines, then just make them a single line
    Simple rule is unti we see a token before ::= , its the same rule

    Args:
        lines:

    Returns:

    """
    consolidated: list[str] = []
    current_rule: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue  # Skip truly empty lines

        # new rule is detected based on presence of  "::=".
        # Until it appears it's the same rule, no matter how many line breaks it has
        if "::=" in stripped:
            #  building previous rule?
            if current_rule:
                consolidated.append(" ".join(current_rule))
                current_rule = []
            # new rule with this line
            current_rule.append(stripped)
        else:
            # continuation line.just append to current rule.
            if not current_rule:
                raise ValueError(f"Line '{line}' has no rule to continue.")
            current_rule.append(stripped)
    # last rule
    if current_rule:
        consolidated.append(" ".join(current_rule))

    return consolidated


def split_choices(rhs_symbols: list[str]) -> list[list[str]]:
    """
    Split RHS  into production choices
    For now it simply finds choices based on presence of choice symbol (Can be improved later, if needed.)
    """
    # placeholder variables
    production_choices: list[list[str]] = []
    current_symbol: list[str] = []
    # just iterate in rhs symbols if choice symbol (pipe) is found
    for item in rhs_symbols:
        if item == "|":
            if current_symbol:
                production_choices.append(current_symbol)
                current_symbol = []
        else:
            current_symbol.append(item)
    if current_symbol:
        production_choices.append(current_symbol)
    return production_choices


class BNFGrammarParser(GrammarParser):
    """
    Parses a grammar string written in Backus-Naur Form (BNF) into structured rules.

    The parser identifies terminal and non-terminal symbols, splits productions into alternatives,
    and validates the structure of the grammar.

    Attributes:
        grammar_str (str): The raw BNF grammar string.
    Args:
        grammar_str (str): The grammar definition in BNF format.

    """

    def __init__(self, grammar_str: str) -> None:
        super().__init__()
        self.rules_original: dict[str, "Rule"] = {}
        self.rules: dict[str, "Rule"] = {}
        self.non_terminals: dict[str, None] = {}
        self.terminals: dict[str, None] = {}
        self.start_rule: Optional[str] = None
        self.grammar_str: str = grammar_str

        # Setup
        self.non_terminal_pattern = re.compile(r"^<\w+>$")

        # Registry for registering range handlers
        self.range_registry = RangeHandlerRegistry()

        # Build the symbol pattern (combining handler patterns with base patterns)
        self.symbol_pattern = self._build_symbol_pattern()

    def _build_symbol_pattern(self) -> re.Pattern[str]:
        """Build the complete symbol pattern using registered handlers and base patterns."""

        # Get range patterns from handlers
        range_pattern = self.range_registry.get_combined_pattern()

        # Base patterns for other tokens, the symbol patterns.
        # IMPORTANT:: Update from v1.0.1alpha5 separated - sign to support negative integers in range handling.
        base_patterns = [
            r'"[^"]*"',  # Double-quoted strings
            r"'[^']*'",  # Single-quoted strings
            r"<[^>]+>",  # Non-terminals
            r"-?\d+\.\d+",  # Float numbers (including negative)
            r"-?\d+",  # Integer numbers (including negative)
            r"\w+",  # Identifiers
            r"[{}()\[\];|:=,+*/=]",  # Symbols without minus  (-).
            r"-",  # Minus sign separately
        ]

        # IMPORTANT :: Handler patterns come first so that they match before base patterns
        all_patterns = [range_pattern] + base_patterns

        # Build final pattern string
        pattern_str = "|".join(f"({p})" for p in all_patterns)

        return re.compile(pattern_str, re.VERBOSE)

    def _fix_hyphenated_tokens(self, tokens: list[str]) -> list[str]:
        """
        Recombine tokens that should be hyphenated terminals. for example turn-left
        """
        fixed = []
        i = 0
        while i < len(tokens):
            # Look for pattern: word, '-', word
            if (
                i + 2 < len(tokens)
                and tokens[i + 1] == "-"
                and re.match(r"^[a-zA-Z]+$", tokens[i])
                and re.match(r"^[a-zA-Z]+$", tokens[i + 2])
            ):
                # Combine into hyphenated token
                fixed.append(f"{tokens[i]}-{tokens[i + 2]}")
                i += 3
            else:
                fixed.append(tokens[i])
                i += 1
        return fixed

    def parse(
        self,
    ) -> tuple[dict[str, Rule], dict[str, Rule], str, list[str], list[str]]:
        """
        Parses the BNF grammar string into rules, terminals, and non-terminals.

        Returns:
            Tuple:
                rules (dict[str, Rule]): Mapping of non-terminal symbols to Rule objects.
                rules_expanded (dict[str, Rule]): Mapping of non-terminal symbols to Rule objects in Expanded (range notation) form.
                start_rule (str): The first non-terminal rule, considered the start rule.
                terminals (set[str]): List of terminal symbols.
                non_terminals (set[str]): List of non-terminal symbols.


        Raises:
            ValueError: If the grammar syntax is invalid or contains undefined symbols.
        """
        # get all lines from the grammar
        lines = self.grammar_str.strip().split("\n")
        #  filter the comments
        filtered_lines = remove_comments(lines)
        # Consolidate multi-line rules
        consolidated_lines = consolidate_multiline_rules(filtered_lines)

        rhs_tokens = []

        for line in consolidated_lines:
            if "::=" in line:
                lhs, rhs = line.split("::=", 1)
                lhs = lhs.strip()

                # Validate LHS
                if not self.non_terminal_pattern.match(lhs):
                    raise ValueError(f"Value Error: Invalid non-terminal '{lhs}'")

                # Validate RHS brackets
                if rhs.count("<") != rhs.count(">"):
                    raise ValueError(
                        f"Syntax Error in RHS of '{lhs}': unmatched angle brackets in '{rhs}'"
                    )

                # Find all tokens in RHS
                tokens = []
                for match in self.symbol_pattern.finditer(rhs):
                    token = match.group(0)
                    if token:
                        tokens.append(token)

                #  Recombine hyphenated tokens
                tokens = self._fix_hyphenated_tokens(tokens)

                # Process tokens using the handler registry
                processed_tokens = []
                for token in tokens:
                    # Use registry to expand token
                    expanded = self.range_registry.expand_token(token)
                    processed_tokens.extend(expanded)

                # Store for later validation
                rhs_tokens.append((lhs, processed_tokens))

                # Create rules
                original_choices = split_choices(tokens)
                expanded_choices = split_choices(processed_tokens)

                self.rules_original[lhs] = Rule(lhs, original_choices)
                self.rules[lhs] = Rule(lhs, expanded_choices)
                self.non_terminals.setdefault(lhs, None)

        # Validate all non-terminals are defined
        all_defined_non_terminals = self.non_terminals

        for lhs, tokens in rhs_tokens:
            for token in tokens:
                if token == "|":
                    continue

                if self.non_terminal_pattern.match(token):
                    if token not in all_defined_non_terminals:
                        raise ValueError(
                            f"Undefined non-terminal symbol '{token}', found in rule for '{lhs}'"
                        )
                else:
                    self.terminals.setdefault(token, None)

        self.start_rule = None
        for line in consolidated_lines:
            if "::=" in line:
                lhs = line.split("::=", 1)[0].strip()
                if lhs in self.non_terminals:
                    self.start_rule = lhs
                    break

        # This should never happen, but let's be defensive
        if self.start_rule is None:
            # Fallback: use first non-terminal
            self.start_rule = next(iter(self.non_terminals.keys()))

        return (
            self.rules_original,
            self.rules,
            self.start_rule,
            list(self.terminals.keys()),
            list(self.non_terminals.keys()),
        )
