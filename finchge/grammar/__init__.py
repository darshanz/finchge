from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finchge.grammar.grammar import Grammar
    from finchge.grammar.mapper import GenotypeMapper
    from finchge.grammar.parser import BNFGrammarParser, GrammarParser
    from finchge.grammar.rule import Rule


def __getattr__(name: str) -> Any:
    if name in {"Grammar"}:
        from finchge.grammar.grammar import Grammar

        return locals()[name]
    if name in {"Rule"}:
        from finchge.grammar.rule import Rule

        return locals()[name]
    if name in {"BNFGrammarParser", "GrammarParser"}:
        from finchge.grammar.parser import BNFGrammarParser, GrammarParser

        return locals()[name]
    if name in {"GenotypeMapper"}:
        from finchge.grammar.mapper import GenotypeMapper

        return locals()[name]
    raise AttributeError(name)


__all__ = ["Grammar", "Rule", "GrammarParser", "BNFGrammarParser", "GenotypeMapper"]
