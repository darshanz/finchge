import subprocess
import sys
from pathlib import Path

import typer


def run_project():
    cwd = Path.cwd()
    main_file = cwd / "main.py"

    if not main_file.exists():
        typer.secho(
            "Error: main.py not found in current directory.", fg=typer.colors.RED
        )
        raise typer.Exit(1)

    typer.secho("Running finchGE experiment...", fg=typer.colors.GREEN)

    result = subprocess.run([sys.executable, "main.py"], cwd=cwd)

    if result.returncode != 0:
        typer.secho("Experiment failed.", fg=typer.colors.RED)
        raise typer.Exit(result.returncode)
