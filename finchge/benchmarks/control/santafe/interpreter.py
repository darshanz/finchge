from typing import List, Optional, Tuple


class SantaFeInterpreter:
    """
    Interpreter for Santa Fe ant programs using cleaner grammar.

    """

    def __init__(self) -> None:
        self.actions: List[str] = ["move", "turn-left", "turn-right"]
        self.keywords: List[str] = self.actions + ["ifelse", "food-ahead"]

    def tokenize(self, program: str) -> List[str]:
        """Convert program string to list of tokens."""
        tokens: List[str] = []
        i = 0
        while i < len(program):
            # Skip whitespace
            if program[i].isspace():
                i += 1
                continue

            # Parentheses as separate tokens
            if program[i] == "(":
                tokens.append("(")
                i += 1
                continue
            if program[i] == ")":
                tokens.append(")")
                i += 1
                continue

            # Check for keywords
            matched = False
            for kw in self.keywords:
                if program.startswith(kw, i):
                    tokens.append(kw)
                    i += len(kw)
                    matched = True
                    break

            if not matched:
                # Skip unknown characters
                i += 1

        return tokens

    def evaluate(
        self, tokens: List[str], observation: bool, pc: int = 0
    ) -> Tuple[Optional[str], int]:
        """
        Evaluate tokenized program.

        Args:
            tokens: List of tokens from tokenize()
            observation: Current food-ahead sensor value
            pc: Program counter (index into tokens)

        Returns:
            Tuple of (action, new_pc)
        """
        if pc >= len(tokens):
            return None, pc

        token = tokens[pc]

        # Actions
        if token in self.actions:
            return token, pc + 1

        # ifelse statement
        if token == "ifelse":
            # Expect 'food-ahead'
            if pc + 1 >= len(tokens) or tokens[pc + 1] != "food-ahead":
                return None, pc + 2

            # Expect '('
            if pc + 2 >= len(tokens) or tokens[pc + 2] != "(":
                return None, pc + 3

            # Parse true branch
            true_action, true_pc = self.evaluate(tokens, observation, pc + 3)

            # Expect ')'
            if true_pc >= len(tokens) or tokens[true_pc] != ")":
                return None, true_pc

            # Expect '(' for false branch
            if true_pc + 1 >= len(tokens) or tokens[true_pc + 1] != "(":
                return None, true_pc + 2

            # Parse false branch
            false_action, false_pc = self.evaluate(tokens, observation, true_pc + 2)

            # Expect ')'
            if false_pc >= len(tokens) or tokens[false_pc] != ")":
                return None, false_pc

            # Return appropriate action based on observation
            if observation:
                return true_action, false_pc + 1
            else:
                return false_action, false_pc + 1

        return self.evaluate(tokens, observation, pc + 1)

    def program_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens back to readable program string.
        Tokens separated by space
        """
        return " ".join(tokens)
