"""Local validation for LLLM hub packages."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

from ..protocol._validation import is_schema_metadata_key, is_sensitive_metadata_key
from .endpoints import (
    resource_extra as _resource_extra,
    validate_endpoint_metadata as _validate_endpoint_metadata,
)
from .entrypoints import import_entrypoint
from .files import has_symlink_component, is_relative_to
from .files import validate_declared_file as _validate_declared_file
from .manifest import load_manifest, require_path_value
from .models import PackageManifest, ValidationIssue, ValidationReport
from .refs import parse_resource_ref

BUILTIN_SCHEMAS = {
    "Any",
    "any",
    "str",
    "string",
    "int",
    "integer",
    "float",
    "bool",
    "boolean",
    "dict",
    "object",
    "list",
    "array",
}
def validate_package(path: str | Path) -> ValidationReport:
    issues: list[ValidationIssue] = []
    try:
        source_manifest = _unresolved_manifest_path(path)
        if source_manifest.is_symlink():
            return ValidationReport(
                ok=False,
                issues=(
                    ValidationIssue(
                        level="error",
                        code="manifest_symlink",
                        message="lllm.toml must be a regular package file, not a symlink.",
                        resource="lllm.toml",
                    ),
                ),
            )
        manifest = load_manifest(source_manifest)
    except Exception as exc:
        return ValidationReport(
            ok=False,
            issues=(_manifest_load_issue(exc),),
        )

    issues.extend(_validate_resource_names(manifest))
    issues.extend(_validate_readme(manifest))
    issues.extend(_validate_card_metadata(manifest))
    issues.extend(_validate_secret_metadata(manifest))
    issues.extend(_validate_primary(manifest))
    issues.extend(_validate_schemas(manifest))
    issues.extend(_validate_tactics(manifest))
    issues.extend(_validate_services(manifest))
    issues.extend(_validate_channels(manifest))
    issues.extend(_validate_snapshots(manifest))
    issues.extend(_validate_runs(manifest))
    issues.extend(_validate_config(manifest))
    issues.extend(_validate_docs(manifest))
    issues.extend(_validate_examples(manifest))
    issues.extend(_validate_assets(manifest))
    issues.extend(_validate_discovery(manifest))
    issues.extend(_validate_dependencies(manifest))
    return ValidationReport(
        ok=not any(issue.level == "error" for issue in issues),
        issues=tuple(issues),
    )


def _manifest_load_issue(exc: Exception) -> ValidationIssue:
    message = str(exc)
    return ValidationIssue(
        level="error",
        code=(
            "manifest_duplicate_name"
            if _looks_like_duplicate_toml_name(message)
            else "manifest_load_failed"
        ),
        message=message,
    )


def _looks_like_duplicate_toml_name(message: str) -> bool:
    normalized = message.lower()
    return (
        ("cannot declare" in normalized and "twice" in normalized)
        or ("duplicate" in normalized and "key" in normalized)
        or ("duplicate" in normalized and "table" in normalized)
    )


def _unresolved_manifest_path(path: str | Path) -> Path:
    value = Path(require_path_value(path, "manifest path")).expanduser()
    if value.is_dir():
        value = value / "lllm.toml"
    return value


def _validate_readme(manifest: PackageManifest) -> list[ValidationIssue]:
    if manifest.base_dir is None or (manifest.base_dir / "README.md").is_file():
        return []
    return [
        ValidationIssue(
            level="warning",
            code="readme_missing",
            message="Package should include README.md.",
        )
    ]


def _validate_resource_names(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    tables = {
        "schemas": manifest.schemas,
        "tactics": manifest.tactics,
        "services": manifest.services,
        "channels": manifest.channels,
        "snapshots": manifest.snapshots,
        "runs": manifest.runs,
        "docs": manifest.docs,
        "examples": manifest.examples,
        "assets": manifest.assets,
    }
    codes = {
        "schemas": "schema_name_invalid",
        "tactics": "tactic_name_invalid",
        "services": "service_name_invalid",
        "channels": "channel_name_invalid",
        "snapshots": "snapshot_name_invalid",
        "runs": "run_name_invalid",
        "docs": "doc_name_invalid",
        "examples": "example_name_invalid",
        "assets": "asset_name_invalid",
    }
    for section, table in tables.items():
        for name in table:
            if _invalid_segment(name):
                issues.append(
                    ValidationIssue(
                        level="error",
                        code=codes[section],
                        message=(
                            f"Resource name must be a non-empty path segment: "
                            f"{section}.{name}"
                        ),
                        resource=f"{section}.{name}",
                    )
                )
    return issues


def _invalid_segment(value: str) -> bool:
    return (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\;%")
    )


def _validate_card_metadata(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if manifest.card is None:
        issues.append(
            ValidationIssue(
                level="warning",
                code="card_metadata_missing",
                message="Package should declare [card] metadata for generated cards.",
                resource="card",
            )
        )
    elif not manifest.card.summary and not manifest.package.description:
        issues.append(
            ValidationIssue(
                level="warning",
                code="card_summary_missing",
                message="Package card should include a summary or package description.",
                resource="card.summary",
            )
        )
    if (
        manifest.base_dir is not None
        and (manifest.base_dir / "README.md").is_file()
        and "readme" not in manifest.docs
    ):
        issues.append(
            ValidationIssue(
                level="warning",
                code="readme_doc_missing",
                message="Package README.md should be declared as [docs.readme].",
                resource=manifest.ref("doc", "readme"),
            )
        )
    return issues


def _validate_secret_metadata(manifest: PackageManifest) -> list[ValidationIssue]:
    """Reject raw secret-shaped metadata keys; lllm.toml ships with the package."""

    resources: list[tuple[str, Any]] = [
        (manifest.ref(kind, name), resource)
        for kind, table in (
            ("schema", manifest.schemas),
            ("tactic", manifest.tactics),
            ("service", manifest.services),
            ("channel", manifest.channels),
            ("snapshot", manifest.snapshots),
            ("run", manifest.runs),
            ("doc", manifest.docs),
            ("example", manifest.examples),
            ("asset", manifest.assets),
        )
        for name, resource in table.items()
        if not _invalid_segment(name)
    ]
    if manifest.config is not None:
        resources.append((manifest.ref("config", "default"), manifest.config))
    if manifest.card is not None:
        resources.append(("card", manifest.card))
    return [
        ValidationIssue(
            level="error",
            code="metadata_secret_key",
            message=(
                f"Metadata must not include raw secret key {key!r}: lllm.toml is "
                f"published as written. Use a reference such as '{key}_ref' instead."
            ),
            resource=ref,
        )
        for ref, resource in resources
        # The hub indexes extra table keys as metadata too.
        for source in (_resource_extra(resource), resource.metadata)
        for key in _secret_metadata_keys(source)
    ]


def _secret_metadata_keys(value: Any) -> list[str]:
    """Secret-shaped keys in a metadata value, leaving JSON schemas alone."""

    if isinstance(value, dict):
        keys: list[str] = []
        for key, item in value.items():
            if is_sensitive_metadata_key(key):
                keys.append(key)
            elif not is_schema_metadata_key(key):
                keys.extend(_secret_metadata_keys(item))
        return keys
    if isinstance(value, (list, tuple)):
        return [key for item in value for key in _secret_metadata_keys(item)]
    return []


def _validate_primary(manifest: PackageManifest) -> list[ValidationIssue]:
    primary = manifest.package.primary
    if not primary:
        if manifest.package.kind in {"tactic", "channel", "service", "app"}:
            return [
                ValidationIssue(
                    level="warning",
                    code="primary_missing_for_kind",
                    message=(
                        f"Package kind {manifest.package.kind!r} should declare "
                        "package.primary."
                    ),
                    resource="package.primary",
                )
            ]
        return []
    if "." not in primary:
        return [
            ValidationIssue(
                level="error",
                code="primary_invalid",
                message="package.primary must look like resources.name.",
                resource="package.primary",
            )
        ]
    section, name = primary.split(".", 1)
    kind_issue = _validate_primary_kind(manifest.package.kind, section, primary)
    if kind_issue is not None:
        return [kind_issue]
    table = {
        "schemas": manifest.schemas,
        "tactics": manifest.tactics,
        "services": manifest.services,
        "channels": manifest.channels,
        "snapshots": manifest.snapshots,
        "runs": manifest.runs,
        "config": {"default": manifest.config} if manifest.config is not None else {},
        "docs": manifest.docs,
        "examples": manifest.examples,
        "assets": manifest.assets,
    }.get(section)
    if table is None or name not in table:
        return [
            ValidationIssue(
                level="error",
                code="primary_missing",
                message=f"package.primary references missing resource {primary!r}.",
                resource="package.primary",
            )
        ]
    return []


def _validate_primary_kind(
    package_kind: str,
    section: str,
    primary: str,
) -> ValidationIssue | None:
    expected = {
        "tactic": {"tactics"},
        "channel": {"channels", "snapshots"},
        "service": {"services"},
        "app": {"services", "runs"},
    }.get(package_kind)
    if expected is None or section in expected:
        return None
    allowed = ", ".join(sorted(expected))
    return ValidationIssue(
        level="error",
        code="primary_kind_mismatch",
        message=(
            f"Package kind {package_kind!r} expects package.primary in "
            f"{allowed}, got {primary!r}."
        ),
        resource="package.primary",
    )


def _validate_schemas(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, schema in manifest.schemas.items():
        if _invalid_segment(name):
            continue
        if schema.entry:
            issues.extend(_validate_import(schema.entry, manifest, manifest.ref("schema", name)))
    return issues


def _validate_tactics(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, tactic in manifest.tactics.items():
        if _invalid_segment(name):
            continue
        ref = manifest.ref("tactic", name)
        issues.extend(_validate_import(tactic.entry, manifest, ref))
        for schema_ref in (tactic.input, tactic.output):
            if schema_ref:
                issues.extend(_validate_schema_ref(schema_ref, manifest, ref))
        for index, example in enumerate(tactic.examples, start=1):
            if not any(key in example for key in ("input", "output", "command")):
                issues.append(
                    ValidationIssue(
                        level="warning",
                        code="tactic_example_empty",
                        message=(
                            f"Tactic {name!r} example #{index} should declare "
                            "input, output, or command."
                        ),
                        resource=ref,
                    )
                )
        issues.extend(_validate_endpoint_metadata(tactic, ref))
    return issues


def _validate_services(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, service in manifest.services.items():
        if _invalid_segment(name):
            continue
        ref = manifest.ref("service", name)
        if not service.entry and not service.tactic and not service.tactics:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="service_unbound",
                    message=(
                        f"Service {name!r} must declare an entrypoint or tactic."
                    ),
                    resource=ref,
                )
            )
        if service.entry:
            issues.extend(_validate_import(service.entry, manifest, ref))
        issues.extend(_validate_endpoint_metadata(service, ref))
        issues.extend(_validate_service_port_metadata(service, ref))
        for tactic in _service_tactic_refs(service):
            if _invalid_segment(tactic):
                issues.append(
                    _invalid_local_ref(
                        ref,
                        "service_tactic_invalid",
                        f"Service {name!r}",
                        "tactic",
                        tactic,
                    )
                )
            elif tactic not in manifest.tactics:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="service_tactic_missing",
                        message=(
                            f"Service {name!r} references missing tactic "
                            f"{tactic!r}."
                        ),
                        resource=ref,
                    )
                )
        for channel in (*service.subscribes, *service.publishes):
            if _invalid_segment(channel):
                issues.append(
                    _invalid_local_ref(
                        ref,
                        "service_channel_invalid",
                        f"Service {name!r}",
                        "channel",
                        channel,
                    )
                )
            elif channel not in manifest.channels:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="service_channel_missing",
                        message=f"Service {name!r} references missing channel {channel!r}.",
                        resource=ref,
                    )
                )
    return issues


def _service_tactic_refs(service: Any) -> tuple[str, ...]:
    tactics: list[str] = []
    if service.tactic is not None:
        tactics.append(service.tactic)
    tactics.extend(service.tactics)
    return tuple(tactics)


def _validate_service_port_metadata(service: Any, ref: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for port in _declared_service_ports(service):
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not (1 <= port <= 65535)
        ):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="service_port_invalid",
                    message=(
                        "Service port metadata must be an integer between "
                        f"1 and 65535: {port!r}."
                    ),
                    resource=ref,
                )
            )
    return issues


def _declared_service_ports(service: Any) -> tuple[Any, ...]:
    ports: list[Any] = []
    extra = _resource_extra(service)
    if "port" in extra:
        ports.append(extra["port"])
    if "port" in service.metadata:
        ports.append(service.metadata["port"])
    return tuple(ports)


def _validate_channels(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, channel in manifest.channels.items():
        if _invalid_segment(name):
            continue
        ref = manifest.ref("channel", name)
        if channel.schema:
            issues.extend(
                _validate_schema_ref(
                    channel.schema,
                    manifest,
                    ref,
                )
            )
        issues.extend(_validate_endpoint_metadata(channel, ref))
    return issues


def _validate_snapshots(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, snapshot in manifest.snapshots.items():
        if _invalid_segment(name):
            continue
        ref = manifest.ref("snapshot", name)
        if snapshot.schema:
            issues.extend(_validate_schema_ref(snapshot.schema, manifest, ref))
        if snapshot.channel is not None and _invalid_segment(snapshot.channel):
            issues.append(
                _invalid_local_ref(
                    ref,
                    "snapshot_channel_invalid",
                    f"Snapshot {name!r}",
                    "channel",
                    snapshot.channel,
                )
            )
        elif snapshot.channel and snapshot.channel not in manifest.channels:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="snapshot_channel_missing",
                    message=(
                        f"Snapshot {name!r} references missing channel "
                        f"{snapshot.channel!r}."
                    ),
                    resource=ref,
                )
            )
        issues.extend(_validate_endpoint_metadata(snapshot, ref))
    return issues


def _validate_runs(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, run in manifest.runs.items():
        if _invalid_segment(name):
            continue
        ref = manifest.ref("run", name)
        for service in run.services:
            if _invalid_segment(service):
                issues.append(
                    _invalid_local_ref(
                        ref,
                        "run_service_invalid",
                        f"Run {name!r}",
                        "service",
                        service,
                    )
                )
            elif service not in manifest.services:
                issues.append(_missing(ref, "run_service_missing", name, service))
        for tactic in run.tactics:
            if _invalid_segment(tactic):
                issues.append(
                    _invalid_local_ref(
                        ref,
                        "run_tactic_invalid",
                        f"Run {name!r}",
                        "tactic",
                        tactic,
                    )
                )
            elif tactic not in manifest.tactics:
                issues.append(_missing(ref, "run_tactic_missing", name, tactic))
        for channel in run.channels:
            if _invalid_segment(channel):
                issues.append(
                    _invalid_local_ref(
                        ref,
                        "run_channel_invalid",
                        f"Run {name!r}",
                        "channel",
                        channel,
                    )
                )
            elif channel not in manifest.channels:
                issues.append(_missing(ref, "run_channel_missing", name, channel))
        for snapshot in run.snapshots:
            if _invalid_segment(snapshot):
                issues.append(
                    _invalid_local_ref(
                        ref,
                        "run_snapshot_invalid",
                        f"Run {name!r}",
                        "snapshot",
                        snapshot,
                    )
                )
            elif snapshot not in manifest.snapshots:
                issues.append(_missing(ref, "run_snapshot_missing", name, snapshot))
    return issues


def _invalid_local_ref(
    resource: str,
    code: str,
    owner: str,
    target_kind: str,
    value: str,
) -> ValidationIssue:
    return ValidationIssue(
        level="error",
        code=code,
        message=(
            f"{owner} references malformed {target_kind} name {value!r}; "
            "same-package refs must be non-empty path segments."
        ),
        resource=resource,
    )


def _validate_config(manifest: PackageManifest) -> list[ValidationIssue]:
    config = manifest.config
    if config is None or not config.schema:
        return []
    issues: list[ValidationIssue] = []
    unknown_defaults = sorted(set(config.defaults) - set(config.schema))
    for key in unknown_defaults:
        issues.append(
            ValidationIssue(
                level="warning",
                code="config_default_without_schema",
                message=f"Config default {key!r} has no schema entry.",
                resource=manifest.ref("config", "default"),
            )
        )
    return issues


def _validate_docs(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, doc in manifest.docs.items():
        if _invalid_segment(name):
            continue
        issues.extend(
            _validate_declared_file(
                manifest,
                path=doc.path,
                resource=manifest.ref("doc", name),
                code="doc_path_missing",
                label=f"Doc {name!r}",
            )
        )
    return issues


def _validate_examples(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, example in manifest.examples.items():
        if _invalid_segment(name):
            continue
        ref = manifest.ref("example", name)
        if not example.path and not example.command:
            issues.append(
                ValidationIssue(
                    level="warning",
                    code="example_empty",
                    message=f"Example {name!r} should declare a path or command.",
                    resource=ref,
                )
            )
        if example.path is not None:
            issues.extend(
                _validate_declared_file(
                    manifest,
                    path=example.path,
                    resource=ref,
                    code="example_path_missing",
                    label=f"Example {name!r}",
                )
            )
    return issues


def _validate_assets(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, asset in manifest.assets.items():
        if _invalid_segment(name):
            continue
        issues.extend(
            _validate_declared_file(
                manifest,
                path=asset.path,
                resource=manifest.ref("asset", name),
                code="asset_path_missing",
                label=f"Asset {name!r}",
            )
        )
    return issues


def _validate_discovery(manifest: PackageManifest) -> list[ValidationIssue]:
    """Check native folder-discovery sections the runtime will scan."""

    issues: list[ValidationIssue] = []
    if manifest.base_dir is None:
        return issues
    for section, discovery in manifest.discovery.items():
        if not discovery.paths:
            # The runtime falls back to ./<section>; check it like a declared path.
            issue = _local_folder_issue(
                manifest,
                section,
                resource=section,
                code="discovery_path",
                label=f"[{section}] folder",
            )
            if issue is not None and issue.level == "warning":
                issue = ValidationIssue(
                    level="warning",
                    code="discovery_section_empty",
                    message=(
                        f"[{section}] has no paths and no ./{section} folder, "
                        "so the native runtime discovers nothing for it."
                    ),
                    resource=section,
                )
            if issue is not None:
                issues.append(issue)
            continue
        for entry in discovery.paths:
            raw = entry["path"] if isinstance(entry, dict) else entry
            path = raw.rsplit(" under ", 1)[0].strip() if isinstance(entry, str) else raw
            issue = _local_folder_issue(
                manifest,
                path,
                resource=section,
                code="discovery_path",
                label=f"[{section}] path",
            )
            if issue is not None:
                issues.append(issue)
    return issues


def _validate_dependencies(manifest: PackageManifest) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if manifest.base_dir is None:
        return issues
    for entry in manifest.dependencies.packages:
        raw = entry["path"] if isinstance(entry, dict) else entry
        path = raw.rsplit(" as ", 1)[0].strip() if isinstance(entry, str) else raw
        issue = _local_folder_issue(
            manifest,
            path,
            resource="dependencies",
            code="dependency_path",
            label="Dependency",
        )
        if issue is None and not (manifest.base_dir / path / "lllm.toml").is_file():
            issue = ValidationIssue(
                level="warning",
                code="dependency_manifest_missing",
                message=f"Dependency has no lllm.toml: {path}",
                resource="dependencies",
            )
        if issue is not None:
            issues.append(issue)
    return issues


def _local_folder_issue(
    manifest: PackageManifest,
    path: str,
    *,
    resource: str,
    code: str,
    label: str,
) -> ValidationIssue | None:
    base_dir = manifest.base_dir.resolve()
    if Path(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return ValidationIssue(
            level="error",
            code=f"{code}_absolute",
            message=f"{label} must be relative to the package: {path}",
            resource=resource,
        )
    target = (base_dir / path).resolve()
    if not is_relative_to(target, base_dir) or has_symlink_component(
        manifest.base_dir, Path(path)
    ):
        return ValidationIssue(
            level="error",
            code=f"{code}_outside_package",
            message=f"{label} must stay inside the package: {path}",
            resource=resource,
        )
    if not target.is_dir():
        return ValidationIssue(
            level="warning",
            code=f"{code}_missing",
            message=f"{label} does not exist: {path}",
            resource=resource,
        )
    return None


def _validate_import(
    entry: str,
    manifest: PackageManifest,
    resource: str,
) -> list[ValidationIssue]:
    try:
        import_entrypoint(entry, base_dir=manifest.base_dir)
    except Exception as exc:
        return [
            ValidationIssue(
                level="error",
                code="entrypoint_import_failed",
                message=str(exc),
                resource=resource,
            )
        ]
    return []


def _validate_schema_ref(
    schema_ref: str,
    manifest: PackageManifest,
    resource: str,
) -> list[ValidationIssue]:
    if schema_ref in BUILTIN_SCHEMAS or schema_ref in manifest.schemas:
        return []
    if schema_ref.startswith("lllm://"):
        try:
            parsed = parse_resource_ref(schema_ref)
        except ValueError:
            return [
                ValidationIssue(
                    level="error",
                    code="schema_ref_invalid",
                    message=f"Schema ref {schema_ref!r} is not a valid lllm:// ref.",
                    resource=resource,
                )
            ]
        if parsed.resource_kind != "schemas":
            return [
                ValidationIssue(
                    level="error",
                    code="schema_ref_kind_mismatch",
                    message=(
                        f"Schema ref {schema_ref!r} must point at schemas, "
                        f"not {parsed.resource_kind!r}."
                    ),
                    resource=resource,
                )
            ]
        if (
            parsed.org == manifest.package.org
            and parsed.package == manifest.package.name
            and parsed.name not in manifest.schemas
        ):
            return [
                ValidationIssue(
                    level="error",
                    code="schema_ref_missing",
                    message=f"Schema ref {schema_ref!r} is not declared.",
                    resource=resource,
                )
            ]
        return []
    if ":" in schema_ref:
        return _validate_import(schema_ref, manifest, resource)
    return [
        ValidationIssue(
            level="error",
            code="schema_ref_missing",
            message=f"Schema ref {schema_ref!r} is not declared.",
            resource=resource,
        )
    ]


def _missing(ref: str, code: str, run_name: str, missing_name: str) -> ValidationIssue:
    return ValidationIssue(
        level="error",
        code=code,
        message=f"Run {run_name!r} references missing resource {missing_name!r}.",
        resource=ref,
    )
