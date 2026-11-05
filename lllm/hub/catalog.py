"""Build a static catalog for browsing a hub (for example lllm.one/hub).

``build_catalog(hub, output_dir)`` writes::

    output_dir/
      data/catalog.json            # summaries + per-package details
      packages/<org>-<name>-<version>.zip

The catalog is what the hub web app reads when it runs as static files. Each
package's latest version is included with its card, agent card, local config
template, README, and a download link.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from ..protocol._validation import public_boundary_value
from .local import LocalHub, _should_ignore_publish_name
from .models import PackageRecord

TEXT_SUFFIXES = {".md", ".txt", ".toml", ".py", ".json", ".yml", ".yaml"}


def build_catalog(
    hub: LocalHub,
    output_dir: str | Path,
    *,
    generated_at: str | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``data/catalog.json`` and package zips for every package in ``hub``."""

    output = Path(output_dir)
    data_dir = output / "data"
    package_dir = output / "packages"
    data_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for record in _latest_records(hub):
        filename = download_filename(record)
        download_url = f"packages/{filename}"
        (package_dir / filename).write_bytes(package_zip(record))
        summary = {**summary_payload(record), "download_url": download_url}
        summaries.append(summary)
        details[record.identifier] = {
            **detail_payload(record),
            "card": hub.card(record.identifier, version=record.version),
            "agent_card": hub.agent_card(record.identifier, version=record.version),
            "config_template": hub.config_template(
                record.identifier, version=record.version
            ),
            "readme": package_readme(record),
            "download_url": download_url,
        }

    catalog: dict[str, Any] = {
        "generated_at": generated_at
        or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "packages": summaries,
        "details": details,
    }
    if bundle is not None:
        catalog["bundle"] = dict(bundle)
    (data_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return catalog


def summary_payload(record: PackageRecord) -> dict[str, Any]:
    """Search-result fields for one package."""

    card = record.card
    return {
        "identifier": record.identifier,
        "org": record.org,
        "name": record.name,
        "version": record.version,
        "kind": record.kind,
        "description": record.description,
        "summary": (card.summary if card is not None and card.summary else record.description),
        "tags": list(card.tags) if card is not None else [],
        "resources": [
            {"kind": resource.kind, "name": resource.name, "ref": resource.ref}
            for resource in record.resources
        ],
    }


def detail_payload(record: PackageRecord) -> dict[str, Any]:
    """Package page fields: resources with entries, validation, and commands."""

    return {
        **summary_payload(record),
        "key": record.key,
        "validation": record.validation.model_dump(mode="json"),
        "primary": _primary(record),
        "resources": [
            {
                "kind": resource.kind,
                "name": resource.name,
                "ref": resource.ref,
                "entry": resource.entry,
                "description": resource.description,
                "metadata": public_boundary_value(resource.metadata),
            }
            for resource in record.resources
        ],
        "commands": {
            "get": f"lllm hub get {record.identifier}",
            "card": f"lllm hub card {record.identifier}",
            "agent_card": f"lllm hub agent-card {record.identifier}",
            "config": f"lllm hub config-template {record.identifier}",
            "serve": f"lllm serve {record.name}",
        },
    }


def download_filename(record: PackageRecord) -> str:
    parts = (record.org, record.name, record.version)
    return "-".join(part.replace("/", "-") for part in parts) + ".zip"


def package_zip(record: PackageRecord) -> bytes:
    """Zip a published package folder, skipping secrets and build leftovers."""

    root = record.root
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(_should_ignore_publish_name(part) for part in relative.parts):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            info = zipfile.ZipInfo(f"{record.name}/{relative.as_posix()}")
            info.date_time = (2020, 1, 1, 0, 0, 0)  # reproducible archives
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def package_readme(record: PackageRecord) -> str:
    """The package's declared README (``[docs.readme]``), or README.md."""

    path = next(
        (
            resource.entry
            for resource in record.resources
            if resource.kind == "doc" and resource.name == "readme" and resource.entry
        ),
        "README.md",
    )
    target = (record.root / path).resolve()
    root = record.root.resolve()
    if root not in target.parents or target.suffix.lower() not in TEXT_SUFFIXES:
        return ""
    if target.is_symlink() or not target.is_file():
        return ""
    return target.read_text(encoding="utf-8")


def _latest_records(hub: LocalHub) -> list[PackageRecord]:
    identifiers = sorted({record.identifier for record in hub.list()})
    return [hub.get(identifier) for identifier in identifiers]


def _primary(record: PackageRecord) -> str | None:
    """The declared ``[package] primary``, else the first tactic."""

    from .manifest import load_manifest

    try:
        primary = load_manifest(record.manifest_path).package.primary
    except Exception:
        primary = None
    if primary:
        return primary
    tactic = next((r for r in record.resources if r.kind == "tactic"), None)
    return f"tactics.{tactic.name}" if tactic is not None else None
