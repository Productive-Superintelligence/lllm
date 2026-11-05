"""LLLM hub package models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)

from ..protocol._validation import (
    copy_boundary_value,
    mapping_field_value,
    mapping_sequence_field_value,
    metadata_field_value,
)
from .refs import parse_resource_ref

PackageKind = Literal["tactic", "channel", "service", "app", "library", "mixed"]
ResourceKind = Literal[
    "schema",
    "tactic",
    "service",
    "channel",
    "snapshot",
    "run",
    "config",
    "doc",
    "example",
    "asset",
]
RESOURCE_KIND_VALUES = frozenset(get_args(ResourceKind))
#: Channel forms ``lllm.network`` can create; mirrors ``lllm.network.ChannelForm``
#: (a test keeps them equal) so the hub does not import the network layer.
CHANNEL_FORM_VALUES = frozenset(
    {
        "log",
        "queue",
        "topic",
        "blackboard",
        "latest-state",
        "artifact-index",
        "pipeline-edge",
        "database-view",
        "feed",
        "time-series",
        "tree",
        "forum",
        "hub",
    }
)
PRIMARY_SECTION_VALUES = frozenset(
    {
        "schemas",
        "tactics",
        "services",
        "channels",
        "snapshots",
        "runs",
        "config",
        "docs",
        "examples",
        "assets",
    }
)


class _MetadataModel(BaseModel):
    @field_validator("metadata", mode="before", check_fields=False)
    @classmethod
    def _validate_metadata(cls, value: Any) -> Any:
        return metadata_field_value("metadata", value)


class PackageInfo(BaseModel):
    """Top-level `[package]` metadata."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: StrictStr = "0.1"
    name: StrictStr
    org: StrictStr = "local"
    version: StrictStr = "0.1.0"
    kind: PackageKind = "mixed"
    primary: StrictStr | None = None
    description: StrictStr = ""
    license: StrictStr | None = None
    authors: tuple[StrictStr, ...] = Field(default_factory=tuple)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "authors")

    @model_validator(mode="after")
    def _validate_identity(self) -> "PackageInfo":
        _validate_segment(self.org, "package.org")
        _validate_segment(self.name, "package.name")
        _validate_segment(self.version, "package.version")
        if self.primary is not None:
            _validate_primary_ref(self.primary)
        return self

    @computed_field
    @property
    def identifier(self) -> str:
        return f"{self.org}/{self.name}"


class SchemaResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    entry: StrictStr | None = None
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "metadata")


class TacticResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    entry: StrictStr
    input: StrictStr | None = None
    output: StrictStr | None = None
    runtime: StrictStr = "python"
    description: StrictStr = ""
    examples: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "examples", "metadata")

    @field_validator("examples", mode="before")
    @classmethod
    def _validate_examples(cls, value: Any) -> Any:
        return mapping_sequence_field_value("examples", value, allow_tuple=True)

    @model_validator(mode="after")
    def _validate_metadata_tokens(self) -> "TacticResource":
        _validate_token(self.runtime, "tactic.runtime")
        return self


class ServiceResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    entry: StrictStr | None = None
    tactic: StrictStr | None = None
    tactics: tuple[StrictStr, ...] = Field(default_factory=tuple)
    transport: StrictStr = "fastapi"
    description: StrictStr = ""
    subscribes: tuple[StrictStr, ...] = Field(default_factory=tuple)
    publishes: tuple[StrictStr, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "tactics", "subscribes", "publishes", "metadata")

    @model_validator(mode="after")
    def _validate_metadata_tokens(self) -> "ServiceResource":
        _validate_token(self.transport, "service.transport")
        return self


class ChannelResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    schema_ref: StrictStr | None = Field(default=None, alias="schema")
    form: StrictStr = "log"
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def schema(self) -> str | None:
        return self.schema_ref

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "metadata")

    @model_validator(mode="after")
    def _validate_metadata_tokens(self) -> "ChannelResource":
        _validate_token(self.form, "channel.form")
        if self.form not in CHANNEL_FORM_VALUES:
            expected = ", ".join(sorted(CHANNEL_FORM_VALUES))
            raise ValueError(f"channel.form must be one of: {expected}.")
        return self


class SnapshotResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    schema_ref: StrictStr | None = Field(default=None, alias="schema")
    channel: StrictStr | None = None
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def schema(self) -> str | None:
        return self.schema_ref

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "metadata")


class RunResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    services: tuple[StrictStr, ...] = Field(default_factory=tuple)
    tactics: tuple[StrictStr, ...] = Field(default_factory=tuple)
    channels: tuple[StrictStr, ...] = Field(default_factory=tuple)
    snapshots: tuple[StrictStr, ...] = Field(default_factory=tuple)
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(
            self,
            "services",
            "tactics",
            "channels",
            "snapshots",
            "metadata",
        )


class ConfigResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    config_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    defaults: dict[str, Any] = Field(default_factory=dict)
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def schema(self) -> dict[str, Any]:
        return copy_boundary_value(self.config_schema)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "config_schema", "defaults", "metadata")

    @field_validator("config_schema", "defaults", mode="before")
    @classmethod
    def _validate_public_maps(cls, value: Any, info: ValidationInfo) -> Any:
        return mapping_field_value(info.field_name, value)


class DocResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    path: StrictStr
    title: StrictStr = ""
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "metadata")


class ExampleResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    path: StrictStr | None = None
    command: StrictStr | None = None
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "metadata")


class AssetResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    path: StrictStr
    media_type: StrictStr = "application/octet-stream"
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "metadata")


class CardResource(_MetadataModel):
    model_config = ConfigDict(extra="allow")

    summary: StrictStr = ""
    tags: tuple[StrictStr, ...] = Field(default_factory=tuple)
    safety: StrictStr = ""
    latency: StrictStr = ""
    suggested_commands: tuple[StrictStr, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "tags", "suggested_commands", "metadata")


class RequirementResource(BaseModel):
    """Package-level launch requirements."""

    model_config = ConfigDict(extra="forbid")

    api_keys: dict[StrictStr, StrictStr] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "api_keys")

    @field_validator("api_keys", mode="before")
    @classmethod
    def _validate_api_keys(cls, value: Any) -> Any:
        return mapping_field_value("requirements.api_keys", value)

    @model_validator(mode="after")
    def _validate_key_names(self) -> "RequirementResource":
        for key in self.api_keys:
            _validate_env_key(key, "requirements.api_keys")
        return self


#: Native runtime sections that only describe folder discovery.
NATIVE_DISCOVERY_SECTIONS = frozenset({"prompts", "proxies", "tools", "configs"})
#: Sections that can hold both native ``paths`` discovery and named resources.
SHARED_DISCOVERY_SECTIONS = frozenset({"tactics", "services", "assets"})
_PATH_ENTRY_KEYS = frozenset({"path", "prefix", "under"})
_DEPENDENCY_ENTRY_KEYS = frozenset({"path", "alias", "as"})


class DiscoverySection(BaseModel):
    """Native folder discovery: ``[section] paths = ["./dir", ...]``.

    Entries are either ``"./dir"`` / ``"./dir under prefix"`` strings or
    ``{path = "./dir", prefix = "..."}`` tables, matching the native runtime.
    Custom sections may carry extra keys; the native runtime ignores them.
    """

    model_config = ConfigDict(extra="allow")

    paths: tuple[StrictStr | dict[StrictStr, StrictStr], ...] = Field(
        default_factory=tuple
    )

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "paths")

    @field_validator("paths", mode="before")
    @classmethod
    def _validate_paths(cls, value: Any) -> Any:
        return _entry_sequence(value, "paths", _PATH_ENTRY_KEYS)


class DependencySection(BaseModel):
    """Native ``[dependencies] packages = ["./pkg", "./pkg as alias"]``."""

    model_config = ConfigDict(extra="forbid")

    packages: tuple[StrictStr | dict[StrictStr, StrictStr], ...] = Field(
        default_factory=tuple
    )

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "packages")

    @field_validator("packages", mode="before")
    @classmethod
    def _validate_packages(cls, value: Any) -> Any:
        return _entry_sequence(value, "dependencies.packages", _DEPENDENCY_ENTRY_KEYS)


class PackageManifest(BaseModel):
    """Machine-readable `lllm.toml` package contract.

    One file serves both readers: the hub reads ``[package]`` identity and
    named resources such as ``[tactics.echo]``; the native runtime reads
    ``[dependencies]`` and folder discovery (``[prompts] paths = [...]``,
    ``[tactics] paths = [...]``, and custom sections). Discovery sections are
    collected into :attr:`discovery`, keyed by section name.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    package: PackageInfo
    dependencies: DependencySection = Field(default_factory=DependencySection)
    discovery: dict[StrictStr, DiscoverySection] = Field(default_factory=dict)
    schemas: dict[StrictStr, SchemaResource] = Field(default_factory=dict)
    tactics: dict[StrictStr, TacticResource] = Field(default_factory=dict)
    services: dict[StrictStr, ServiceResource] = Field(default_factory=dict)
    channels: dict[StrictStr, ChannelResource] = Field(default_factory=dict)
    snapshots: dict[StrictStr, SnapshotResource] = Field(default_factory=dict)
    runs: dict[StrictStr, RunResource] = Field(default_factory=dict)
    config: ConfigResource | None = None
    docs: dict[StrictStr, DocResource] = Field(default_factory=dict)
    examples: dict[StrictStr, ExampleResource] = Field(default_factory=dict)
    assets: dict[StrictStr, AssetResource] = Field(default_factory=dict)
    requirements: RequirementResource = Field(default_factory=RequirementResource)
    card: CardResource | None = None
    base_dir: Path | None = None

    @property
    def identifier(self) -> str:
        return self.package.identifier

    @model_validator(mode="before")
    @classmethod
    def _collect_discovery_sections(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "discovery" in data:
            raise ValueError(
                "discovery is not a manifest section; use [section] paths = [...]."
            )
        data = dict(data)
        declared = set(cls.model_fields) - {"discovery", "base_dir"}
        discovery: dict[str, Any] = {}
        for name, value in list(data.items()):
            if not isinstance(value, dict):
                continue
            if name in NATIVE_DISCOVERY_SECTIONS or name not in declared:
                discovery[name] = data.pop(name)
            elif name in SHARED_DISCOVERY_SECTIONS and "paths" in value:
                section = dict(value)
                discovery[name] = {"paths": section.pop("paths")}
                data[name] = section
        data["discovery"] = discovery
        return data

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(
            self,
            "dependencies",
            "discovery",
            "schemas",
            "tactics",
            "services",
            "channels",
            "snapshots",
            "runs",
            "config",
            "docs",
            "examples",
            "assets",
            "requirements",
            "card",
        )

    def ref(self, kind: ResourceKind, name: str) -> str:
        _validate_resource_kind(kind)
        _validate_segment(name, f"{kind}.name")
        plural = f"{kind}s" if kind != "schema" else "schemas"
        return f"lllm://{self.package.org}/{self.package.name}/{plural}/{name}"


class HubResource(_MetadataModel):
    """Indexed package resource."""

    model_config = ConfigDict(frozen=True)

    kind: ResourceKind
    name: StrictStr
    ref: StrictStr
    entry: StrictStr | None = None
    description: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "metadata")

    @model_validator(mode="after")
    def _validate_identity(self) -> "HubResource":
        _validate_segment(self.name, f"{self.kind}.name")
        parsed = parse_resource_ref(self.ref)
        expected_section = "schemas" if self.kind == "schema" else f"{self.kind}s"
        if parsed.resource_kind != expected_section or parsed.name != self.name:
            raise ValueError("resource ref must match resource kind and name.")
        return self


class ValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: Literal["error", "warning"]
    code: StrictStr
    message: StrictStr
    resource: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_code(self) -> "ValidationIssue":
        _validate_token(self.code, "issue.code")
        return self


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: StrictBool
    issues: tuple[ValidationIssue, ...] = Field(default_factory=tuple)

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "issues")


class PackageRecord(BaseModel):
    """A package stored in the local hub index."""

    model_config = ConfigDict(frozen=True)

    org: StrictStr
    name: StrictStr
    version: StrictStr
    kind: PackageKind
    description: StrictStr = ""
    root: Path
    manifest_path: Path
    resources: tuple[HubResource, ...] = Field(default_factory=tuple)
    requirements: RequirementResource = Field(default_factory=RequirementResource)
    validation: ValidationReport = Field(
        default_factory=lambda: ValidationReport(ok=False)
    )
    card: CardResource | None = None

    def model_post_init(self, __context: Any) -> None:
        _isolate_fields(self, "resources", "requirements", "validation", "card")

    @model_validator(mode="after")
    def _validate_identity(self) -> "PackageRecord":
        _validate_segment(self.org, "record.org")
        _validate_segment(self.name, "record.name")
        _validate_segment(self.version, "record.version")
        for resource in self.resources:
            parsed = parse_resource_ref(resource.ref)
            if parsed.org != self.org or parsed.package != self.name:
                raise ValueError("resource ref must match record identifier.")
        return self

    @computed_field
    @property
    def identifier(self) -> str:
        return f"{self.org}/{self.name}"

    @computed_field
    @property
    def key(self) -> str:
        return f"{self.identifier}@{self.version}"

    def resources_by_kind(self, kind: ResourceKind) -> tuple[HubResource, ...]:
        _validate_resource_kind(kind)
        return tuple(
            resource.model_copy(deep=True)
            for resource in self.resources
            if resource.kind == kind
        )


def _validate_segment(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\;%")
    ):
        raise ValueError(f"{field_name} must be a non-empty path segment.")


def _validate_primary_ref(value: str) -> None:
    if not isinstance(value, str) or "." not in value:
        raise ValueError("package.primary must have shape section.name.")
    section, name = value.split(".", 1)
    if section not in PRIMARY_SECTION_VALUES:
        expected = ", ".join(sorted(PRIMARY_SECTION_VALUES))
        raise ValueError(f"package.primary section must be one of: {expected}.")
    _validate_segment(name, "package.primary name")


def _validate_token(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\;%")
    ):
        raise ValueError(f"{field_name} must be a non-empty token.")


def _validate_env_key(value: str, field_name: str) -> None:
    if not isinstance(value, str) or re.match(r"^[A-Z][A-Z0-9_]*$", value) is None:
        raise ValueError(f"{field_name} keys must be uppercase environment names.")


def _entry_sequence(value: Any, label: str, allowed_keys: frozenset[str]) -> Any:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of paths.")
    for entry in value:
        if isinstance(entry, str):
            if not entry.strip():
                raise ValueError(f"{label} entries must be non-empty strings.")
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"{label} entries must be strings or tables with path.")
        unknown = set(entry) - allowed_keys
        if unknown:
            expected = ", ".join(sorted(allowed_keys))
            raise ValueError(f"{label} entry keys must be one of: {expected}.")
    return value


def _validate_resource_kind(value: str) -> None:
    if not isinstance(value, str) or value not in RESOURCE_KIND_VALUES:
        expected = ", ".join(sorted(RESOURCE_KIND_VALUES))
        raise ValueError(f"resource kind must be one of: {expected}.")


def _isolate_fields(model: BaseModel, *field_names: str) -> None:
    for field_name in field_names:
        object.__setattr__(
            model,
            field_name,
            copy_boundary_value(getattr(model, field_name)),
        )
    extra = getattr(model, "__pydantic_extra__", None)
    if extra is not None:
        object.__setattr__(model, "__pydantic_extra__", copy_boundary_value(extra))
