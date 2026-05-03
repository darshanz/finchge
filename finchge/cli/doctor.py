from __future__ import annotations

import importlib.util
import platform
import sys
from importlib import resources

import typer

from finchge import __version__
from finchge.cli.project import list_templates

CORE_DEPENDENCIES = {
    "cloudpickle": "cloudpickle",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "PyYAML": "yaml",
    "scikit-learn": "sklearn",
    "tabulate": "tabulate",
    "tqdm": "tqdm",
    "typer": "typer",
}

OPTIONAL_DEPENDENCIES = {
    "IPython": "IPython",
    "jupyter": "jupyter",
    "torch": "torch",
}


def _module_status(module_name: str) -> str:
    return "ok" if importlib.util.find_spec(module_name) is not None else "missing"


def _resources_status() -> str:
    try:
        templates_root = resources.files("finchge.cli.templates")
        return "ok" if templates_root.is_dir() else "missing"
    except Exception:
        return "missing"


def run_doctor() -> None:
    typer.echo("finchGE doctor")
    typer.echo("")
    typer.echo(f"finchGE version: {__version__}")
    typer.echo(f"Python version: {sys.version.split()[0]}")
    typer.echo(f"Platform: {platform.platform()}")

    typer.echo("")
    typer.echo("Core dependencies:")
    for label, module_name in CORE_DEPENDENCIES.items():
        typer.echo(f"  {label}: {_module_status(module_name)}")

    typer.echo("")
    typer.echo("Optional dependencies:")
    for label, module_name in OPTIONAL_DEPENDENCIES.items():
        typer.echo(f"  {label}: {_module_status(module_name)}")

    typer.echo("")
    typer.echo(f"Package resources: {_resources_status()}")

    typer.echo("")
    typer.echo("Templates:")
    for template_name, title in list_templates():
        typer.echo(f"  {template_name}: {title}")
