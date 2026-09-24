"""The README's Python snippets run in order (with one worker and a smaller budget)."""

import os
import re
from pathlib import Path

README = (Path(__file__).parents[1] / "README.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", README, flags=re.S)


def test_readme_snippets_run(tmp_path):
    assert len(BLOCKS) >= 4
    ns: dict = {}
    cwd = os.getcwd()
    os.chdir(tmp_path)  # snippets may write checkpoints
    try:
        for block in BLOCKS:
            code = block.replace("workers=4", "workers=1").replace("max_calls=10_000", "max_calls=500")
            exec(compile(code, "README.md", "exec"), ns)
    finally:
        os.chdir(cwd)
    assert ns["res"].accounting["objective_calls"] > 0
