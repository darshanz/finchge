from pathlib import Path

import typer

from finchge.config.config import FinchConfig, validate_config

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
    Quick check to make sure a grammar file is actually valid BNF format.
    Don't want to pass garbage to the parser later.
    """
    try:
        # Basic grammar validation
        with open(grammar_path, "r") as f:
            content = f.read()

        # first check - file shouldn't be completely empty
        if not content.strip():
            typer.secho(f"Grammar file '{grammar_path}' is empty", fg=typer.colors.RED)
            return False

        # try to spot if there are actual grammar rules in here
        has_rules = False
        for line in content.split("\n"):
            line = line.strip()
            # skip comments, look for rule definitions
            if line and not line.startswith("#"):
                if "::=" in line or "->" in line or ":" in line:
                    has_rules = True
                    break

        if not has_rules:
            typer.secho(
                f"Grammar file '{grammar_path}' doesn't appear to contain rules",
                fg=typer.colors.RED,
            )
            return False
        # looks good
        typer.secho(f"Grammar file '{grammar_path}' looks valid", fg=typer.colors.GREEN)
        return True

    except FileNotFoundError:
        typer.secho(f"Grammar file not found: {grammar_path}", fg=typer.colors.RED)
        return False
    except Exception as e:
        typer.secho(f"Error reading grammar file: {e}", fg=typer.colors.RED)
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
