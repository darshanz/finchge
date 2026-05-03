from pathlib import Path

import typer

from finchge.config.config import FinchConfig, validate_config
from finchge.grammar import Grammar

app = typer.Typer()


def validate_config_file(config_path: str) -> bool:
    """Validate a GE configuration file."""
    try:
        config = FinchConfig.from_file(config_path)
        issues, warning_issues = validate_config(
            config
        )  # Our existing config validator

        if warning_issues:
            typer.secho(
                f"Configuration file '{config_path}' has warnings:",
                fg=typer.colors.YELLOW,
            )
            for warning_issue in warning_issues:
                typer.secho(f"  • {warning_issue}", fg=typer.colors.YELLOW)
        if not issues:
            typer.secho(
                f"Configuration file '{config_path}' is valid!", fg=typer.colors.GREEN
            )
            return True
        else:
            typer.secho(
                f"Configuration file '{config_path}' has issues:", fg=typer.colors.RED
            )
            for issue in issues:
                typer.secho(f"  • {issue}", fg=typer.colors.RED)
            return False

    except Exception as e:
        typer.secho(f"Error loading config file: {e}", fg=typer.colors.RED)
        return False


def validate_grammar_file(grammar_path: str) -> bool:
    """
    Validate that FinchGE can parse and analyse a grammar file.
    """
    try:
        grammar = Grammar.from_file(grammar_path)
        grammar.analyze()

        summary = (
            f"Parsed grammar with {len(grammar.rules)} rules, "
            f"start rule {grammar.start_rule}, "
            f"can terminate: {grammar.can_terminate}"
        )

        if not grammar.can_terminate:
            typer.secho(
                f"Grammar file '{grammar_path}' is invalid. {summary}.",
                fg=typer.colors.RED,
            )
            return False

        typer.secho(
            f"Grammar file '{grammar_path}' is valid. {summary}.",
            fg=typer.colors.GREEN,
        )
        return True

    except FileNotFoundError:
        typer.secho(f"Grammar file not found: {grammar_path}", fg=typer.colors.RED)
        return False
    except Exception as e:
        typer.secho(f"Invalid grammar file '{grammar_path}': {e}", fg=typer.colors.RED)
        return False


def find_files_recursive(directory: Path, patterns: list[str]) -> list[Path]:
    """
    Find files matching patterns by searching through current folder and all subfolders.
    """
    files = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return files


def find_files_non_recursive(directory: Path, patterns: list[str]) -> list[Path]:
    """
    Find files matching the given patterns, but only in the current directory.
    Won't go into subfolders... just a simple top-level search.
    """
    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return files
