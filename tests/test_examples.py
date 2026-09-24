"""Every example script runs and prints without error."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(path, capsys):
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out.strip()
