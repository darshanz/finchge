from pathlib import Path

import typer

from finchge import __version__
from finchge.cli.project import create_project
from finchge.cli.run import run_project
from finchge.cli.validate import (
    find_files_non_recursive,
    find_files_recursive,
    validate_config_file,
    validate_grammar_file,
)

app = typer.Typer(help="finchGE command line tools")


@app.callback()
def cli(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show finchGE version and exit",
        is_eager=True,
    )
):
    if version:
        typer.echo(f"finchGE {__version__}")
        raise typer.Exit()


@app.command()
def new(
    name: str = typer.Argument(..., help="Name of the project directory"),
    template: str = typer.Option("basic", help="Project template to use"),
    notebook: bool = typer.Option(
        False, "--notebook", help="Include a starter Jupyter notebook"
    ),
):
    create_project(name, template, notebook)


@app.command()
def validate(
    config: str = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    grammar: str = typer.Option(None, "--grammar", "-g", help="Path to grammar file"),
    all_files: bool = typer.Option(
        False,
        "--all",
        help="Validate all config and grammar files in current directory",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Search recursively"
    ),
):
    """
    Validate configuration and grammar files.

    Examples:
        finchge validate --config my_config.yaml --grammar rules.bnf
        finchge validate --all
        finchge validate --config params.yml
        finchge validate
    """

    if all_files:
        typer.secho("Searching for files...", fg=typer.colors.BLUE)

        # Choose search function based on recursive flag
        search_func = find_files_recursive if recursive else find_files_non_recursive

        # Define file patterns
        config_patterns = ["*.yaml", "*.yml", "*.ini"]
        grammar_patterns = ["*.bnf", "*.grammar", "*.ebnf"]

        # Find files
        config_files = search_func(Path.cwd(), config_patterns)
        grammar_files = search_func(Path.cwd(), grammar_patterns)

        # Display found files

        if config_files:
            typer.secho(
                f"\nFound {len(config_files)} config file(s):", fg=typer.colors.BLUE
            )
            for f in sorted(config_files):
                # Show relative path for recursive search
                rel_path = f.relative_to(Path.cwd())
                typer.echo(f"  • {rel_path}")
        else:
            typer.secho("\nNo config file(s) found.", fg=typer.colors.BLUE)

        if grammar_files:
            typer.secho(
                f"\nFound {len(grammar_files)} grammar file(s):", fg=typer.colors.BLUE
            )
            for f in sorted(grammar_files):
                rel_path = f.relative_to(Path.cwd())
                typer.echo(f"  • {rel_path}")
        else:
            typer.secho("\nNo grammar file(s) found.", fg=typer.colors.BLUE)

        # Validate files
        results = []
        for config_file in config_files:
            typer.echo(f"\nValidating config: {config_file.relative_to(Path.cwd())}")
            results.append(validate_config_file(str(config_file)))

        for grammar_file in grammar_files:
            typer.echo(f"\nValidating grammar: {grammar_file.relative_to(Path.cwd())}")
            results.append(validate_grammar_file(str(grammar_file)))

        # Show results
        if config_files or grammar_files:
            if all(results):
                typer.secho("\n All files are valid!", fg=typer.colors.GREEN)
            else:
                raise typer.Exit(1)

    elif config or grammar:
        # Validate specific files
        results = []

        if config:
            results.append(validate_config_file(config))

        if grammar:
            results.append(validate_grammar_file(grammar))

        if all(results):
            typer.secho("\n All specified files are valid!", fg=typer.colors.GREEN)
        else:
            raise typer.Exit(1)

    else:
        # Interactive mode: help user find and validate files
        typer.secho(
            "No files specified. Let's find and validate them interactively.",
            fg=typer.colors.BLUE,
        )

        # Look for config files
        config_candidates = []
        for ext in ["yaml", "yml", "ini"]:
            config_candidates.extend(Path.cwd().glob(f"*.{ext}"))

        if config_candidates:
            typer.secho(
                f"\nFound {len(config_candidates)} config file(s):",
                fg=typer.colors.BLUE,
            )
            for i, candidate in enumerate(config_candidates, 1):
                typer.echo(f"  {i}. {candidate.name}")

            choice = typer.prompt(
                "\nEnter number to validate (or press Enter to skip)",
                default="",
                show_default=False,
            )
            if choice.isdigit() and 1 <= int(choice) <= len(config_candidates):
                validate_config_file(str(config_candidates[int(choice) - 1]))
        else:
            typer.secho(
                "No config files found in current directory.", fg=typer.colors.YELLOW
            )

        # Look for grammar files
        grammar_candidates = list(Path.cwd().glob("*.bnf")) + list(
            Path.cwd().glob("*.grammar")
        )

        if grammar_candidates:
            typer.secho(
                f"\nFound {len(grammar_candidates)} grammar file(s):",
                fg=typer.colors.BLUE,
            )
            for i, candidate in enumerate(grammar_candidates, 1):
                typer.echo(f"  {i}. {candidate.name}")

            choice = typer.prompt(
                "\nEnter number to validate (or press Enter to skip)",
                default="",
                show_default=False,
            )
            if choice.isdigit() and 1 <= int(choice) <= len(grammar_candidates):
                validate_grammar_file(str(grammar_candidates[int(choice) - 1]))
        else:
            typer.secho(
                "No grammar files found in current directory.", fg=typer.colors.YELLOW
            )


@app.command()
def check_config(path: str = typer.Argument(..., help="Path to configuration file")):
    """
    Validate only a configuration file.

    Example:
        finchge check-config my_config.yaml
    """
    if validate_config_file(path):
        typer.Exit(0)
    else:
        typer.Exit(1)


@app.command()
def check_grammar(path: str = typer.Argument(..., help="Path to grammar file")):
    """
    Validate only a grammar file.

    Example:
        finchge check-grammar syntax.bnf
    """
    if validate_grammar_file(path):
        typer.Exit(0)
    else:
        typer.Exit(1)


@app.command()
def run():
    """
    Run a finchGE experiment in the current directory.
    """
    run_project()


def main():
    app()
