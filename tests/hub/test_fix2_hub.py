"""Regression tests for hub fixes: static catalog zip paths (#2) and layout
(#18), secret metadata keys (#13), discovery fallback folders (#14), and
non-UTF-8 READMEs (#29)."""

import io
import zipfile

import pytest

from lllm.hub import (
    LocalHub,
    PublishValidationError,
    build_catalog,
    load_manifest,
    validate_package,
)
from lllm.hub.server import _zip_folder


def write_package(root, manifest, files=()):
    root.mkdir(parents=True, exist_ok=True)
    (root / "lllm.toml").write_text(manifest, encoding="utf-8")
    for path, data in files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return root


def library(org, name):
    return (
        f'[package]\norg = "{org}"\nname = "{name}"\n'
        'version = "1.0.0"\nkind = "library"\n'
    )


# #2: static catalog zips live at packages/<org>/<name>/<version>.zip.


def test_catalog_zips_of_dash_joined_names_do_not_collide(tmp_path):
    hub = LocalHub(tmp_path / "hub")
    for org, name in (("acme-labs", "tool"), ("acme", "labs-tool")):
        source = write_package(
            tmp_path / "src" / f"{org}__{name}",
            library(org, name),
            [("MARKER.txt", f"{org}/{name}")],
        )
        hub.publish(source)
    site = tmp_path / "site"

    catalog = build_catalog(hub, site, generated_at="x")

    urls = {summary["identifier"]: summary["download_url"] for summary in catalog["packages"]}
    assert urls == {
        "acme-labs/tool": "packages/acme-labs/tool/1.0.0.zip",
        "acme/labs-tool": "packages/acme/labs-tool/1.0.0.zip",
    }
    for identifier, url in urls.items():
        assert catalog["details"][identifier]["download_url"] == url
        with zipfile.ZipFile(site / url) as archive:
            assert archive.read("MARKER.txt").decode() == identifier


# #18: static catalog zips hold files at the root, like the API /download.


def test_catalog_zip_unzips_into_a_servable_package_folder(tmp_path):
    source = write_package(
        tmp_path / "src" / "echo",
        library("demo", "echo"),
        [("pkg/data.txt", "hi")],
    )
    hub = LocalHub(tmp_path / "hub")
    record = hub.publish(source)
    site = tmp_path / "site"
    [summary] = build_catalog(hub, site, generated_at="x")["packages"]

    with zipfile.ZipFile(site / summary["download_url"]) as archive:
        names = sorted(archive.namelist())
        # The hub's steps: unzip -d packages/<name>; lllm serve packages/<name>
        archive.extractall(tmp_path / "packages" / "echo")
    with zipfile.ZipFile(io.BytesIO(_zip_folder(record.root))) as api_download:
        api_names = sorted(api_download.namelist())

    assert names == api_names == ["lllm.toml", "pkg/data.txt"]
    assert load_manifest(tmp_path / "packages" / "echo").identifier == "demo/echo"


# #13: secret-shaped metadata keys fail validation; lllm.toml ships as written.

SECRETS = library("demo", "secrets") + '\n[channels.events]\nform = "log"\n'
EVENTS = "lllm://demo/secrets/channels/events"


@pytest.mark.parametrize(
    ("extra", "resource", "key"),
    [
        ('\n[channels.events.metadata]\napi_key = "sk-live-SECRET"\n', EVENTS, "api_key"),
        (
            '\n[channels.events.metadata.headers]\nAuthorization = "Bearer raw"\n',
            EVENTS,
            "Authorization",
        ),
        ('token = "raw-token"\n', EVENTS, "token"),
        ('\n[card]\nsummary = "s"\n\n[card.metadata]\nclientSecret = "raw"\n', "card", "clientSecret"),
    ],
    ids=["metadata", "nested-metadata", "extra-table-key", "card-metadata"],
)
def test_validation_rejects_secret_metadata_keys(tmp_path, extra, resource, key):
    root = write_package(tmp_path / "secrets", SECRETS + extra)

    report = validate_package(root)

    errors = [issue for issue in report.issues if issue.level == "error"]
    assert not report.ok
    assert [(issue.code, issue.resource) for issue in errors] == [
        ("metadata_secret_key", resource)
    ]
    assert repr(key) in errors[0].message
    with pytest.raises(PublishValidationError):
        LocalHub(tmp_path / "hub").publish(root)


def test_validation_allows_secret_references_and_schema_properties(tmp_path):
    root = write_package(
        tmp_path / "refs",
        SECRETS
        + "\n[channels.events.metadata]\n"
        'api_key_ref = "credentials/openai"\n'
        'apiKeyRef = "credentials/camel-openai"\n'
        "max_tokens = 512\n"
        "\n[channels.events.metadata.input_schema.properties.password]\n"
        'type = "string"\n',
    )

    report = validate_package(root)

    assert report.ok, report.issues


# #14: a section without paths falls back to ./<section>, checked like paths.


def discovery_issues(report):
    return [
        (issue.level, issue.code)
        for issue in report.issues
        if issue.code.startswith("discovery")
    ]


@pytest.mark.parametrize(
    ("section", "code"),
    [
        ('"../outside"', "discovery_path_outside_package"),
        ('"/abs/outside"', "discovery_path_absolute"),
    ],
)
def test_discovery_fallback_folder_must_stay_inside_the_package(tmp_path, section, code):
    (tmp_path / "outside").mkdir()
    root = write_package(tmp_path / "pkg", library("demo", "pkg") + f"\n[{section}]\n")

    report = validate_package(root)

    assert not report.ok
    assert discovery_issues(report) == [("error", code)]


def test_discovery_fallback_folder_cannot_symlink_out_of_the_package(tmp_path):
    (tmp_path / "outside").mkdir()
    root = write_package(tmp_path / "pkg", library("demo", "pkg") + "\n[linked]\n")
    try:
        (root / "linked").symlink_to(tmp_path / "outside", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    report = validate_package(root)

    assert discovery_issues(report) == [("error", "discovery_path_outside_package")]


def test_discovery_fallback_folder_inside_the_package_still_validates(tmp_path):
    root = write_package(
        tmp_path / "pkg",
        library("demo", "pkg") + "\n[datasets]\n\n[missing]\n",
        [("datasets/rows.json", "[]")],
    )

    report = validate_package(root)

    assert report.ok
    assert discovery_issues(report) == [("warning", "discovery_section_empty")]


# #29: a README that is not UTF-8 does not break the catalog build.


def test_catalog_reads_non_utf8_readmes_with_replacement(tmp_path):
    source = write_package(
        tmp_path / "src" / "latin",
        library("demo", "latin"),
        [("README.md", "# Café\n".encode("latin-1"))],
    )
    hub = LocalHub(tmp_path / "hub")
    hub.publish(source)

    catalog = build_catalog(hub, tmp_path / "site", generated_at="x")

    assert catalog["details"]["demo/latin"]["readme"] == "# Caf\ufffd\n"
