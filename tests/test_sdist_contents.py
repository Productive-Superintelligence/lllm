import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_sdist_includes_repo_materials(tmp_path):
    pytest.importorskip("build")
    dist_dir = tmp_path / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(dist_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    archives = list(dist_dir.glob("*.tar.gz"))
    assert len(archives) == 1
    root = archives[0].name.removesuffix(".tar.gz")
    with tarfile.open(archives[0]) as archive:
        names = set(archive.getnames())

    required = [
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "assets/lllm-logo-dark.svg",
        "assets/lllm-logo-text-dark.png",
        "assets/lllm-logo-text-white.png",
        "mkdocs.yml",
        "docs/index.md",
        "docs/assets/logo.svg",
        "docs/assets/lllm-logo-text-dark.png",
        "docs/assets/lllm-logo-text-white.png",
        "docs/javascripts/vendor/mermaid.min.js",
        "docs/javascripts/vendor/mermaid-LICENSE.txt",
        "docs/tutorials/first-tactic.md",
        "examples/echo_service/README.md",
        "examples/echo_service/app.py",
        "examples/echo_service/tactics.py",
        "examples/native_dialog/demo.py",
        "examples/pydantic_ai_tactic/fake_agent.py",
        "examples/pydantic_ai_tactic/structured_agent.py",
        "examples/pydantic_ai_tactic/surrounding_features.py",
    ]
    missing = [path for path in required if f"{root}/{path}" not in names]

    assert not missing
    assert not any(name.startswith(f"{root}/site/") for name in names)
