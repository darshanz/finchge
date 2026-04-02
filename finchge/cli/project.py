import shutil
from pathlib import Path

import typer

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_project(name: str, template: str, notebook: bool):
    project_path = Path(name)
    template_path = TEMPLATES_DIR / template

    if project_path.exists():
        typer.secho(f"Error: directory '{name}' already exists.", fg=typer.colors.RED)
        raise typer.Exit(1)

    if not template_path.exists():
        typer.secho(f"Error: template '{template}' not found.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Copy base template
    shutil.copytree(template_path, project_path)

    # Always create logs directory
    (project_path / "logs").mkdir(exist_ok=True)

    # Optionally add notebook
    if notebook:
        nb_template = TEMPLATES_DIR / "notebook" / f"{template}.ipynb"
        if nb_template.exists():
            notebooks_dir = project_path / "notebooks"
            notebooks_dir.mkdir(exist_ok=True)
            shutil.copy(nb_template, notebooks_dir / f"{template}.ipynb")
        else:
            typer.secho(
                f"Notebook not available for '{template}'. Skipping Notebook creation.",
                fg=typer.colors.YELLOW,
            )

    typer.secho(
        f"finchGE project '{name}' created successfully.", fg=typer.colors.GREEN
    )
