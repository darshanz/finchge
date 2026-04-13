import shutil
from importlib import resources
from pathlib import Path

import typer


def create_project(name: str, template: str, notebook: bool):
    project_path = Path(name)
    templates_root = resources.files("finchge.cli.templates")
    template_ref = templates_root / template

    if project_path.exists():
        typer.secho(f"Error: directory '{name}' already exists.", fg=typer.colors.RED)
        raise typer.Exit(1)

    if not template_ref.is_dir():
        typer.secho(f"Error: template '{template}' not found.", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Copy base template
    with resources.as_file(template_ref) as template_path:
        shutil.copytree(template_path, project_path)

    # Always create logs directory
    (project_path / "logs").mkdir(exist_ok=True)

    # Optionally add notebook
    if notebook:
        notebook_ref = templates_root / "notebook" / f"{template}.ipynb"
        if notebook_ref.is_file():
            notebooks_dir = project_path / "notebooks"
            notebooks_dir.mkdir(exist_ok=True)
            with resources.as_file(notebook_ref) as nb_template:
                shutil.copy(nb_template, notebooks_dir / f"{template}.ipynb")
        else:
            typer.secho(
                f"Notebook not available for '{template}'. Skipping Notebook creation.",
                fg=typer.colors.YELLOW,
            )

    typer.secho(
        f"finchGE project '{name}' created successfully.", fg=typer.colors.GREEN
    )
