"""Disk-backed local LLLM hub."""

from __future__ import annotations

import json
import shutil
from pathlib import Path, PureWindowsPath
from typing import Any

from ._metadata import (
    is_public_sensitive_metadata_key as _is_sensitive_metadata_key,
    is_schema_metadata_key as _is_schema_metadata_key,
)
from .cards import render_agent_card, render_config_template, render_package_card
from .files import has_symlink_component, invalid_portable_file_path, is_relative_to
from .manifest import load_manifest, manifest_path, require_path_value
from .models import HubResource, PackageManifest, PackageRecord, ValidationReport
from .validator import validate_package


PUBLISH_IGNORE_NAMES = {
    ".aws",
    ".azure",
    ".direnv",
    ".gcloud",
    ".git",
    ".mypy_cache",
    ".lllm",
    ".pytest_cache",
    ".ruff_cache",
    ".ssh",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "site",
    "venv",
}
SAFE_ENV_TEMPLATE_NAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    ".envrc.example",
    ".envrc.sample",
    ".envrc.template",
}
SECRET_CONFIG_FILE_NAMES = {
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}


class PublishValidationError(ValueError):
    """Raised when validated publish receives an invalid package."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("Package validation failed.")


class LocalHub:
    """A deterministic local package hub."""

    def __init__(self, root: str | Path = ".lllm/hub") -> None:
        self.root = Path(require_path_value(root, "hub root")).expanduser().resolve()
        self.packages_dir = self.root / "packages"
        self.index_dir = self.root / "index"
        self.index_path = self.index_dir / "packages.json"
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, PackageRecord] = {}
        self._load()

    def publish(self, package_path: str | Path, *, validate: bool = True) -> PackageRecord:
        if not isinstance(validate, bool):
            raise ValueError("publish validate must be a boolean.")
        source_manifest = load_manifest(package_path)
        report = validate_package(package_path) if validate else ValidationReport(ok=True)
        if validate and not report.ok:
            raise PublishValidationError(report)
        destination = self._package_destination(source_manifest)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            source_manifest.base_dir or manifest_path(package_path).parent,
            destination,
            ignore=_package_copy_ignore,
        )
        manifest = load_manifest(destination)
        record = record_from_manifest(manifest, validation=report)
        self._records[record.key] = record
        self._write()
        return record.model_copy(deep=True)

    def get(self, identifier: str, *, version: str | None = None) -> PackageRecord:
        org, name = _split_identifier(identifier)
        matches = [
            record
            for record in self._records.values()
            if record.org == org and record.name == name
        ]
        if version is not None:
            _validate_identifier_segment(version, "package version")
            matches = [record for record in matches if record.version == version]
        if not matches:
            raise KeyError(f"Package not found: {identifier}")
        return sorted(matches, key=lambda record: _version_sort_key(record.version))[
            -1
        ].model_copy(
            deep=True
        )

    def download(
        self,
        identifier: str,
        destination: str | Path,
        *,
        version: str | None = None,
    ) -> Path:
        destination_root = Path(
            require_path_value(destination, "download destination")
        ).expanduser()
        record = self.get(identifier, version=version)
        target = destination_root / record.name
        target_resolved = target.resolve(strict=False)
        if is_relative_to(target_resolved, self.root):
            raise ValueError(
                "download destination must resolve outside the local hub root."
            )
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise ValueError(
                    "download target already exists and is not a directory."
                )
            shutil.rmtree(target)
        shutil.copytree(record.root, target, ignore=_package_copy_ignore)
        return target

    def list(self) -> tuple[PackageRecord, ...]:
        return tuple(
            record.model_copy(deep=True)
            for record in sorted(self._records.values(), key=lambda record: record.key)
        )

    def card(self, identifier: str, *, version: str | None = None) -> str:
        return render_package_card(self.get(identifier, version=version))

    def agent_card(self, identifier: str, *, version: str | None = None) -> str:
        return render_agent_card(self.get(identifier, version=version))

    def config_template(self, identifier: str, *, version: str | None = None) -> str:
        return render_config_template(self.get(identifier, version=version))

    def _package_destination(self, manifest: PackageManifest) -> Path:
        return (
            self.packages_dir
            / manifest.package.org
            / manifest.package.name
            / manifest.package.version
        )

    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid local hub index: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Invalid local hub index: root must be an object.")
        records = data.get("records", [])
        if not isinstance(records, list):
            raise ValueError("Invalid local hub index: records must be a list.")

        loaded: dict[str, PackageRecord] = {}
        for index, item in enumerate(records):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid local hub index record {index}: must be an object."
                )
            key = item.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"Invalid local hub index record {index}: key must be a string."
                )
            raw_record = item.get("record")
            if not isinstance(raw_record, dict):
                raise ValueError(
                    f"Invalid local hub index record {index}: record must be an object."
                )
            record = PackageRecord.model_validate(raw_record)
            self._validate_loaded_record_paths(index, record)
            if key != record.key:
                raise ValueError(
                    f"Invalid local hub index record {index}: key does not match record."
                )
            if key in loaded:
                raise ValueError(
                    f"Invalid local hub index record {index}: duplicate key."
                )
            loaded[key] = record
        self._records = loaded

    def _validate_loaded_record_paths(
        self, index: int, record: PackageRecord
    ) -> None:
        expected_root = (
            self.packages_dir / record.org / record.name / record.version
        )
        if record.root != expected_root:
            raise ValueError(
                f"Invalid local hub index record {index}: package root must "
                "match packages/org/name/version."
            )
        if expected_root.is_symlink() or not expected_root.is_dir():
            raise ValueError(
                f"Invalid local hub index record {index}: package root "
                "must be an existing directory inside the hub packages tree."
            )
        expected_manifest = expected_root / "lllm.toml"
        if record.manifest_path != expected_manifest:
            raise ValueError(
                f"Invalid local hub index record {index}: manifest path "
                "must match packages/org/name/version/lllm.toml."
            )
        if expected_manifest.is_symlink() or not expected_manifest.is_file():
            raise ValueError(
                f"Invalid local hub index record {index}: manifest path "
                "must be an existing package lllm.toml file."
            )

    def _write(self) -> None:
        payload: dict[str, Any] = {
            "records": [
                {
                    "key": record.key,
                    "record": record.model_dump(mode="json"),
                }
                for record in self.list()
            ]
        }
        self.index_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def record_from_manifest(
    manifest: PackageManifest,
    *,
    validation: ValidationReport | None = None,
) -> PackageRecord:
    resources: list[HubResource] = []
    for name, schema in manifest.schemas.items():
        resources.append(
            HubResource(
                kind="schema",
                name=name,
                ref=manifest.ref("schema", name),
                entry=schema.entry,
                description=schema.description,
                metadata=_resource_metadata(schema),
            )
        )
    for name, tactic in manifest.tactics.items():
        resources.append(
            HubResource(
                kind="tactic",
                name=name,
                ref=manifest.ref("tactic", name),
                entry=tactic.entry,
                description=tactic.description,
                metadata=_resource_metadata(
                    tactic,
                    runtime=tactic.runtime,
                    input=tactic.input,
                    output=tactic.output,
                    examples=list(tactic.examples),
                ),
            )
        )
    for name, service in manifest.services.items():
        resources.append(
            HubResource(
                kind="service",
                name=name,
                ref=manifest.ref("service", name),
                entry=service.entry,
                description=service.description,
                metadata=_resource_metadata(
                    service,
                    tactic=service.tactic,
                    tactics=list(service.tactics),
                    transport=service.transport,
                    subscribes=list(service.subscribes),
                    publishes=list(service.publishes),
                ),
            )
        )
    for name, channel in manifest.channels.items():
        resources.append(
            HubResource(
                kind="channel",
                name=name,
                ref=manifest.ref("channel", name),
                description=channel.description,
                metadata=_resource_metadata(
                    channel,
                    schema=channel.schema,
                    form=channel.form,
                ),
            )
        )
    for name, snapshot in manifest.snapshots.items():
        resources.append(
            HubResource(
                kind="snapshot",
                name=name,
                ref=manifest.ref("snapshot", name),
                description=snapshot.description,
                metadata=_resource_metadata(
                    snapshot,
                    schema=snapshot.schema,
                    channel=snapshot.channel,
                ),
            )
        )
    for name, run in manifest.runs.items():
        resources.append(
            HubResource(
                kind="run",
                name=name,
                ref=manifest.ref("run", name),
                description=run.description,
                metadata=_resource_metadata(
                    run,
                    services=list(run.services),
                    tactics=list(run.tactics),
                    channels=list(run.channels),
                    snapshots=list(run.snapshots),
                ),
            )
        )
    if manifest.config is not None:
        resources.append(
            HubResource(
                kind="config",
                name="default",
                ref=manifest.ref("config", "default"),
                description=manifest.config.description,
                metadata=_resource_metadata(
                    manifest.config,
                    schema=manifest.config.schema,
                    defaults=manifest.config.defaults,
                ),
            )
        )
    for name, doc in manifest.docs.items():
        path = _record_file_path(manifest, doc.path, "doc")
        resources.append(
            HubResource(
                kind="doc",
                name=name,
                ref=manifest.ref("doc", name),
                entry=path,
                description=doc.description,
                metadata=_resource_metadata(
                    doc,
                    title=doc.title,
                    path=path,
                ),
            )
        )
    for name, example in manifest.examples.items():
        path = (
            _record_file_path(manifest, example.path, "example")
            if example.path is not None
            else None
        )
        resources.append(
            HubResource(
                kind="example",
                name=name,
                ref=manifest.ref("example", name),
                entry=path,
                description=example.description,
                metadata=_resource_metadata(
                    example,
                    path=path,
                    command=example.command,
                ),
            )
        )
    for name, asset in manifest.assets.items():
        path = _record_file_path(manifest, asset.path, "asset")
        resources.append(
            HubResource(
                kind="asset",
                name=name,
                ref=manifest.ref("asset", name),
                entry=path,
                description=asset.description,
                metadata=_resource_metadata(
                    asset,
                    path=path,
                    media_type=asset.media_type,
                ),
            )
        )
    return PackageRecord(
        org=manifest.package.org,
        name=manifest.package.name,
        version=manifest.package.version,
        kind=manifest.package.kind,
        description=manifest.package.description,
        root=manifest.base_dir or Path.cwd(),
        manifest_path=(manifest.base_dir or Path.cwd()) / "lllm.toml",
        resources=tuple(resources),
        requirements=manifest.requirements,
        validation=validation or ValidationReport(ok=False),
        card=manifest.card,
    )


def _resource_metadata(resource: Any, **canonical: Any) -> dict[str, Any]:
    return _public_resource_metadata(
        {
            **_resource_extra(resource),
            **getattr(resource, "metadata", {}),
            **canonical,
        }
    )


def _resource_extra(resource: Any) -> dict[str, Any]:
    return dict(getattr(resource, "model_extra", None) or {})


def _record_file_path(manifest: PackageManifest, path: str, label: str) -> str:
    if (
        invalid_portable_file_path(path)
        or ":" in path
        or any(ch.isspace() for ch in path)
        or Path(path).is_absolute()
        or PureWindowsPath(path).is_absolute()
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError(
            f"{label} path must be a portable relative package path."
        )
    if manifest.base_dir is not None:
        base_dir = manifest.base_dir.resolve()
        relative = Path(path)
        if has_symlink_component(manifest.base_dir, relative):
            raise ValueError(
                f"{label} path must not traverse symlink components."
            )
        target = (base_dir / relative).resolve()
        if not is_relative_to(target, base_dir):
            raise ValueError(f"{label} path must stay inside the package.")
    return path


def _public_resource_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        metadata: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            if _is_sensitive_metadata_key(key):
                continue
            if _is_schema_metadata_key(key):
                metadata[key] = item
            else:
                metadata[key] = _public_resource_metadata(item)
        return metadata
    if isinstance(value, list):
        return [_public_resource_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_public_resource_metadata(item) for item in value)
    return value


def _package_copy_ignore(directory: str, names: list[str]) -> set[str]:
    root = Path(directory)
    return {
        name
        for name in names
        if _should_ignore_publish_name(name) or (root / name).is_symlink()
    }


def _should_ignore_publish_name(name: str) -> bool:
    if name in PUBLISH_IGNORE_NAMES:
        return True
    if name in SECRET_CONFIG_FILE_NAMES:
        return True
    if name == ".env":
        return True
    if name.startswith(".env.") and name not in SAFE_ENV_TEMPLATE_NAMES:
        return True
    if name == ".envrc":
        return True
    if name.startswith(".envrc.") and name not in SAFE_ENV_TEMPLATE_NAMES:
        return True
    return (
        name.endswith(".egg-info")
        or name.endswith(".pyc")
        or name.endswith(".pyo")
        or name == ".coverage"
    )


def _split_identifier(identifier: str) -> tuple[str, str]:
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Package identifier must have shape org/name.")
    parts = identifier.split("/")
    if len(parts) != 2:
        raise ValueError("Package identifier must have shape org/name.")
    org, name = parts
    _validate_identifier_segment(org, "package identifier org")
    _validate_identifier_segment(name, "package identifier name")
    return org, name


def _validate_identifier_segment(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\;%")
    ):
        raise ValueError(f"{field_name} must be a non-empty path segment.")


def _version_sort_key(version: str) -> tuple[Any, ...]:
    public, _, _local = version.partition("+")
    release, prerelease_separator, prerelease = public.partition("-")
    release_parts = release.split(".")
    if release_parts and all(part.isdecimal() for part in release_parts):
        return (
            1,
            tuple(int(part) for part in release_parts),
            1 if not prerelease_separator else 0,
            _natural_text_key(prerelease),
        )
    return (0, _natural_text_key(version))


def _natural_text_key(text: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    current = ""
    current_is_digit: bool | None = None
    for char in text:
        is_digit = char.isdigit()
        if current and current_is_digit != is_digit:
            parts.append(_natural_text_part(current, current_is_digit))
            current = char
            current_is_digit = is_digit
            continue
        current += char
        current_is_digit = is_digit
    if current:
        parts.append(_natural_text_part(current, current_is_digit))
    return tuple(parts)


def _natural_text_part(text: str, is_digit: bool | None) -> tuple[int, int | str]:
    if is_digit:
        return (0, int(text))
    return (1, text.lower())
