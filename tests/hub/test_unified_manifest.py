"""One `lllm.toml` serves both the hub and the native runtime."""

import textwrap

import pytest

from lllm.hub import load_manifest, validate_package
from lllm.runtimes.native.core.config import load_package
from lllm.runtimes.native.core.runtime import Runtime


def write_package(root, manifest, files=()):
    root.mkdir(parents=True, exist_ok=True)
    (root / "lllm.toml").write_text(textwrap.dedent(manifest).lstrip(), encoding="utf-8")
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    for path, text in files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


UNIFIED = """
[package]
org = "demo"
name = "unified"
version = "0.1.0"

[assets]
paths = ["./data"]

[assets.readme]
path = "README.md"

[extras]
paths = ["./extras"]

[docs.readme]
path = "README.md"

[card]
summary = "Unified demo."
"""


def test_one_manifest_feeds_hub_and_native_runtime(tmp_path):
    root = write_package(
        tmp_path / "pkg",
        UNIFIED,
        files=[
            ("data/table.json", "{}"),
            ("extras/more.json", "{}"),
            ("docs/guide.md", "guide"),
        ],
    )

    manifest = load_manifest(root)
    assert {name: section.paths for name, section in manifest.discovery.items()} == {
        "assets": ("./data",),
        "extras": ("./extras",),
    }
    assert list(manifest.assets) == ["readme"]
    assert list(manifest.docs) == ["readme"]
    assert validate_package(root).ok

    runtime = Runtime()
    load_package(str(root / "lllm.toml"), runtime=runtime)
    # Native discovery scans paths and custom sections, but never treats
    # hub-declared sections such as [docs.readme] as folders to discover.
    assert sorted(runtime.keys()) == [
        "unified.assets:table.json",
        "unified.extras:more.json",
    ]


def test_native_only_manifest_is_a_valid_hub_package(tmp_path):
    root = write_package(
        tmp_path / "native",
        """
        [package]
        name = "native"
        description = "v1-style package."

        [prompts]
        paths = ["./prompts under chat"]

        [dependencies]
        packages = ["./vendor/shared as shared"]
        """,
        files=[
            ("prompts/__init__.py", ""),
            ("vendor/shared/lllm.toml", '[package]\nname = "shared"\n'),
        ],
    )

    manifest = load_manifest(root)
    report = validate_package(root)

    assert manifest.identifier == "local/native"
    assert manifest.discovery["prompts"].paths == ("./prompts under chat",)
    assert manifest.dependencies.packages == ("./vendor/shared as shared",)
    assert report.ok
    assert all(issue.level == "warning" for issue in report.issues)


def test_section_that_discovers_nothing_is_flagged(tmp_path):
    root = write_package(
        tmp_path / "typo",
        """
        [package]
        name = "typo"

        [tactic.echo]
        entry = "demo:Echo"
        """,
    )

    report = validate_package(root)

    assert report.ok
    assert [
        issue.code for issue in report.issues if issue.resource == "tactic"
    ] == ["discovery_section_empty"]


@pytest.mark.parametrize(
    ("path", "level", "code"),
    [
        ("./missing", "warning", "discovery_path_missing"),
        ("../outside", "error", "discovery_path_outside_package"),
        ("/abs/prompts", "error", "discovery_path_absolute"),
    ],
)
def test_discovery_paths_must_stay_inside_the_package(tmp_path, path, level, code):
    (tmp_path / "outside").mkdir()
    root = write_package(
        tmp_path / "pkg",
        f"""
        [package]
        name = "paths"

        [prompts]
        paths = [{path!r}]
        """,
    )

    report = validate_package(root)

    matching = [issue for issue in report.issues if issue.code == code]
    assert [issue.level for issue in matching] == [level]
    assert report.ok is (level == "warning")


def test_dependency_without_manifest_is_flagged(tmp_path):
    root = write_package(
        tmp_path / "pkg",
        """
        [package]
        name = "deps"

        [dependencies]
        packages = [{path = "./vendor/empty", alias = "empty"}]
        """,
        files=[("vendor/empty/README.md", "")],
    )

    codes = [issue.code for issue in validate_package(root).issues]

    assert "dependency_manifest_missing" in codes


@pytest.mark.parametrize(
    "section",
    [
        '[prompts]\npaths = "./prompts"',
        "[prompts]\npaths = [{dir = './prompts'}]",
        '[dependencies]\npackages = [{path = "./x", rename = "y"}]',
        '[discovery.prompts]\npaths = ["./prompts"]',
    ],
)
def test_malformed_native_sections_fail_to_load(tmp_path, section):
    root = write_package(
        tmp_path / "bad",
        '[package]\nname = "bad"\n\n' + section + "\n",
    )

    report = validate_package(root)

    assert not report.ok
    assert report.issues[0].code == "manifest_load_failed"
