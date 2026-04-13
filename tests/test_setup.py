from typer.testing import CliRunner

from finchge.cli.main import app


def test_import():
    import finchge

    assert finchge.__version__


# test CLI
runner = CliRunner()


def test_new_basic_project_copies_config_and_main(tmp_path):
    project_dir = tmp_path / "demo"

    result = runner.invoke(
        app,
        ["new", str(project_dir), "--template", "basic"],
    )

    assert result.exit_code == 0, result.output
    assert project_dir.exists()
    assert (project_dir / "config.yaml").exists()
    assert (project_dir / "main.py").exists()
    assert (project_dir / "README.md").exists()
    assert (project_dir / "grammar.bnf").exists()
    assert (project_dir / "logs").exists()
