"""Packaging metadata checks for lllm-core."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
NOT_IN_ALL = {"all", "docs", "dev"}


def test_all_extra_includes_every_runtime_extra():
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    everything = set(extras["all"])
    missing = {
        name: sorted(set(requirements) - everything)
        for name, requirements in extras.items()
        if name not in NOT_IN_ALL and set(requirements) - everything
    }
    assert not missing, f"add these to the `all` extra: {missing}"
