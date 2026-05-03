from typer.testing import CliRunner

from finchge.cli import main as cli_main

runner = CliRunner()


def test_new_lists_available_templates():
    result = runner.invoke(cli_main.app, ["new", "--list-templates"])

    assert result.exit_code == 0
    assert "basic: finchGE Example Project - StringMatch" in result.output
    assert "control: finchGE Example Project - SantaFe Trail Problem" in result.output
    assert "logic: finchGE Example Project - Multiplexer-6 Benchmark" in result.output
    assert (
        "symbolic_regression: finchGE Example Project - Nguyen6 Benchmark"
        in result.output
    )
    assert "notebook" not in result.output


def test_new_requires_name_when_not_listing_templates():
    result = runner.invoke(cli_main.app, ["new"])

    assert result.exit_code == 1
    assert "Project name is required" in result.output
