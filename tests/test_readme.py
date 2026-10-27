"""The README quickstart runs."""

import re
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"


def test_quickstart_runs(capsys):
    if not README.exists():
        pytest.skip("not a source checkout")
    code = re.search(r"```python\n(.*?)```", README.read_text(), re.S)
    assert code
    exec(compile(code.group(1), "README.md", "exec"), {})  # noqa: S102
    assert "IH7" in capsys.readouterr().out
