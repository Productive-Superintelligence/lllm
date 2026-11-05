"""Package card and config-template rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from ._metadata import (
    is_public_sensitive_metadata_key as _is_sensitive_metadata_key,
)
from .endpoints import (
    ENDPOINT_MODES,
    ENDPOINT_SCOPES,
    HTTP_METHODS,
    valid_endpoint_path,
)
from .models import PackageRecord


def render_agent_card(record: PackageRecord) -> str:
    """Render concise package instructions for coding agents."""

    card = record.card
    summary = (
        card.summary
        if card is not None and card.summary
        else record.description
        or f"LLLM package `{record.identifier}`."
    )
    lines = [
        f"# Agent Card: {record.identifier}",
        "",
        summary,
        "",
        "## Operating Notes",
        "",
        "- LLLM hub describes packages, refs, config, and metadata; it does not launch services.",
        "- Resolve required services/channels through local `.lllm/config.toml` bindings.",
        "- Prefer declared refs and resource metadata over guessing paths or URLs.",
        "",
    ]
    lines.extend(_requirements_lines(record))
    if card is not None:
        if card.safety:
            lines.append(f"- Safety: {card.safety}")
        if card.latency:
            lines.append(f"- Latency: {card.latency}")
        if card.tags:
            lines.append(f"- Tags: {', '.join(card.tags)}")
        if card.suggested_commands:
            lines.extend(["", "## Suggested Commands", ""])
            for command in card.suggested_commands:
                lines.append(f"- `{command}`")
            lines.append("")
    lines.extend(["## Resources", ""])
    if not record.resources:
        lines.append("No resources declared.")
    for resource in record.resources:
        lines.append(f"- {resource.kind} `{resource.name}`: `{resource.ref}`")
        if resource.entry:
            lines.append(f"  - entry/path: `{resource.entry}`")
        if resource.description:
            lines.append(f"  - {resource.description}")
        metadata = _metadata_summary(resource.metadata)
        if metadata:
            lines.append(f"  - Metadata: {metadata}")
        lines.extend(_endpoint_lines(resource.metadata))
        lines.extend(_example_lines(resource.metadata))
    template = render_config_template(record).rstrip()
    if template:
        lines.extend(["", "## Config Template", "", "```toml", template, "```"])
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_package_card(record: PackageRecord) -> str:
    summary = (
        record.card.summary
        if record.card is not None and record.card.summary
        else record.description
    )
    lines = [
        f"# {record.identifier}",
        "",
        summary or f"LLLM package `{record.identifier}`.",
        "",
        f"- Version: `{record.version}`",
        f"- Kind: `{record.kind}`",
        f"- Validation: `{'ok' if record.validation.ok else 'failed'}`",
        "",
    ]
    lines.extend(_card_metadata_lines(record))
    lines.extend(_requirements_lines(record))
    lines.extend([
        "## Resources",
        "",
    ])
    if not record.resources:
        lines.append("No resources declared.")
    for resource in record.resources:
        lines.append(f"- `{resource.ref}` ({resource.kind})")
        if resource.description:
            lines.append(f"  - {resource.description}")
        metadata = _metadata_summary(resource.metadata)
        if metadata:
            lines.append(f"  - Metadata: {metadata}")
        lines.extend(_endpoint_lines(resource.metadata))
        lines.extend(_example_lines(resource.metadata))
        if resource.kind == "config":
            defaults = _mapping_summary(resource.metadata.get("defaults"))
            if defaults:
                lines.append(f"  - Defaults: {defaults}")
    lines.extend(["", "## Local Config Template", "", "```toml"])
    lines.append(render_config_template(record).rstrip())
    lines.extend(["```", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_config_template(record: PackageRecord) -> str:
    lines: list[str] = []
    settings = _settings_lines(record)
    if settings:
        lines.extend(settings)
        lines.append("")
    resources = tuple(record.resources)
    service_ports = _service_ports(resources)
    tactic_ports = _tactic_ports(resources, service_ports)
    service_settings = _service_settings_lines(service_ports)
    if service_settings:
        lines.extend(service_settings)
        lines.append("")
    store_settings = _store_settings_lines(resources)
    if store_settings:
        lines.extend(store_settings)
        lines.append("")
    for resource in record.resources:
        if resource.kind == "service":
            lines.extend(
                [
                    f"[refs.{_toml_key(resource.ref)}]",
                    f'url = "http://127.0.0.1:{service_ports[resource.name]}"',
                ]
            )
            lines.extend(_ref_metadata_lines(resource.ref, resource.metadata))
            lines.append("")
        if resource.kind == "tactic":
            tactic_port = tactic_ports.get(resource.name, 8000)
            lines.extend(
                [
                    f"[refs.{_toml_key(resource.ref)}]",
                    f'url = "http://127.0.0.1:{tactic_port}/tactics/'
                    f'{resource.name}"',
                ]
            )
            lines.extend(_ref_metadata_lines(resource.ref, resource.metadata))
            lines.append("")
        if resource.kind in {"channel", "snapshot"}:
            lines.extend(
                [
                    f"[refs.{_toml_key(resource.ref)}]",
                    'store = ".sssn"',
                ]
            )
            lines.extend(_ref_metadata_lines(resource.ref, resource.metadata))
            lines.append("")
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def _service_settings_lines(service_ports: dict[str, int]) -> list[str]:
    lines: list[str] = []
    for name, port in service_ports.items():
        lines.extend([f"[services.{_toml_key(name)}]", f"port = {port}", ""])
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _store_settings_lines(resources: tuple[Any, ...]) -> list[str]:
    if not any(resource.kind in {"channel", "snapshot"} for resource in resources):
        return []
    return ["[stores.default]", 'path = ".sssn"']


def _service_ports(resources: tuple[Any, ...]) -> dict[str, int]:
    ports: dict[str, int] = {}
    used: set[int] = set()
    next_port = 8000
    for resource in resources:
        if resource.kind != "service":
            continue
        port = _preferred_service_port(resource)
        if port is None or port in used:
            while next_port in used:
                next_port += 1
            port = next_port
        ports[resource.name] = port
        used.add(port)
        next_port += 1
    return ports


def _preferred_service_port(resource: Any) -> int | None:
    port = resource.metadata.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        return None
    if not (1 <= port <= 65535):
        return None
    return port


def _tactic_ports(
    resources: tuple[Any, ...], service_ports: dict[str, int]
) -> dict[str, int]:
    ports: dict[str, int] = {}
    for resource in resources:
        if resource.kind != "service":
            continue
        tactic = resource.metadata.get("tactic")
        if isinstance(tactic, str) and tactic:
            ports.setdefault(tactic, service_ports[resource.name])
        tactics = resource.metadata.get("tactics")
        if isinstance(tactics, list):
            for tactic_name in tactics:
                if isinstance(tactic_name, str) and tactic_name:
                    ports.setdefault(tactic_name, service_ports[resource.name])
    return ports


def _card_metadata_lines(record: PackageRecord) -> list[str]:
    card = record.card
    if card is None:
        return []
    lines: list[str] = []
    if card.tags:
        lines.append(f"- Tags: {', '.join(f'`{tag}`' for tag in card.tags)}")
    if card.safety:
        lines.append(f"- Safety: {card.safety}")
    if card.latency:
        lines.append(f"- Latency: {card.latency}")
    if card.suggested_commands:
        lines.extend(["", "## Suggested Commands", ""])
        for command in card.suggested_commands:
            lines.append(f"- `{command}`")
    if lines and lines[-1] != "":
        lines.append("")
    return lines


def _requirements_lines(record: PackageRecord) -> list[str]:
    requirements = getattr(record, "requirements", None)
    api_keys = getattr(requirements, "api_keys", None)
    if not isinstance(api_keys, dict) or not api_keys:
        return []
    lines = ["## Required API Keys", ""]
    for key, description in sorted(api_keys.items()):
        suffix = f": {description}" if description else ""
        lines.append(f"- `{key}`{suffix}")
    lines.append("")
    return lines


def _settings_lines(record: PackageRecord) -> list[str]:
    settings: list[str] = []
    for resource in record.resources:
        if resource.kind != "config":
            continue
        defaults = resource.metadata.get("defaults")
        if not isinstance(defaults, dict):
            continue
        rendered: list[str] = []
        for key, item in sorted(defaults.items()):
            if _is_sensitive_metadata_key(key):
                continue
            value = _toml_public_value(item)
            if value is not None:
                rendered.append(f"{_toml_key(key)} = {value}")
        if rendered:
            if not settings:
                settings.append("[settings]")
            settings.extend(rendered)
    return settings


def _metadata_lines(metadata: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in sorted(metadata.items()):
        if key in {"url", "store", "path", "endpoints", "examples"} or value in (
            None,
            "",
            [],
            {},
        ) or _is_sensitive_metadata_key(key):
            continue
        rendered = _toml_public_value(value)
        if rendered is not None:
            lines.append(f"{_toml_key(key)} = {rendered}")
    return lines


def _ref_metadata_lines(ref: str, metadata: dict[str, Any]) -> list[str]:
    metadata_lines = _metadata_lines(metadata)
    if not metadata_lines:
        return []
    return ["", f"[refs.{_toml_key(ref)}.metadata]", *metadata_lines]


def _toml_key(key: str) -> str:
    if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", key):
        return key
    return json.dumps(key)


def _toml_value(value: Any) -> str | None:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        rendered = [_toml_value(item) for item in value]
        if any(item is None for item in rendered):
            return None
        return "[" + ", ".join(rendered_item for rendered_item in rendered if rendered_item) + "]"
    if isinstance(value, Mapping):
        rendered_items: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str):
                return None
            rendered = _toml_value(item)
            if rendered is None:
                return None
            rendered_items.append(f"{_toml_key(key)} = {rendered}")
        return "{ " + ", ".join(rendered_items) + " }"
    return None


def _toml_public_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        rendered_items: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str) or _is_sensitive_metadata_key(key):
                continue
            rendered = _toml_public_value(item)
            if rendered is None:
                return None
            rendered_items.append(f"{_toml_key(key)} = {rendered}")
        if not rendered_items:
            return None
        return "{ " + ", ".join(rendered_items) + " }"
    if isinstance(value, (list, tuple)):
        rendered = [_toml_public_value(item) for item in value]
        if any(item is None for item in rendered):
            return None
        return "[" + ", ".join(rendered_item for rendered_item in rendered if rendered_item) + "]"
    return _toml_value(value)


def _metadata_summary(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in sorted(metadata.items()):
        if key in {"defaults", "endpoints", "examples", "schema"}:
            continue
        if _is_sensitive_metadata_key(key):
            continue
        rendered = _toml_public_value(value)
        if rendered is not None:
            parts.append(f"`{key}={rendered}`")
    return ", ".join(parts)


def _endpoint_lines(metadata: dict[str, Any]) -> list[str]:
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, list):
        return []
    lines: list[str] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        rendered = _endpoint_line(endpoint)
        if rendered is None:
            continue
        lines.append(rendered)
    return lines


def _endpoint_line(endpoint: dict[str, Any]) -> str | None:
    method_value = endpoint.get("method")
    method = method_value.upper() if isinstance(method_value, str) else ""
    path = endpoint.get("path")
    if method not in HTTP_METHODS or not valid_endpoint_path(path):
        return None
    suffix_parts: list[str] = []
    name = endpoint.get("name")
    if name not in (None, ""):
        if not _valid_endpoint_label(name):
            return None
        suffix_parts.append(name)
    mode = endpoint.get("mode")
    scope = endpoint.get("scope")
    if mode not in (None, ""):
        if mode not in ENDPOINT_MODES:
            return None
        suffix_parts.append(mode)
    elif scope not in (None, ""):
        if scope not in ENDPOINT_SCOPES:
            return None
        suffix_parts.append(scope)
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"  - Endpoint: `{method} {path}`{suffix}"


def _valid_endpoint_label(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "%" not in value
        and not any(ch.isspace() for ch in value)
    )


def _example_lines(metadata: dict[str, Any]) -> list[str]:
    examples = metadata.get("examples")
    if not isinstance(examples, list):
        return []
    lines: list[str] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        description = _example_description(example)
        if description is None:
            continue
        parts: list[str] = []
        if "input" in example:
            rendered = _json_public_inline(example["input"])
            if rendered is not None:
                parts.append(f"`input={rendered}`")
        if "output" in example:
            rendered = _json_public_inline(example["output"])
            if rendered is not None:
                parts.append(f"`output={rendered}`")
        if "command" in example:
            rendered = _json_public_inline(example["command"])
            if rendered is not None:
                parts.append(f"`command={rendered}`")
        if not parts:
            continue
        if description and description.endswith((".", "!", "?", ":")):
            prefix = f"{description} "
        else:
            prefix = f"{description}: " if description else ""
        lines.append(f"  - Example: {prefix}{' -> '.join(parts)}")
    return lines


def _example_description(example: dict[str, Any]) -> str | None:
    value = example.get("description")
    if value in (None, ""):
        value = example.get("name")
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        return None
    return value.strip()


def _json_inline(value: Any) -> str | None:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return None


_JSON_OMIT = object()


def _json_public_inline(value: Any) -> str | None:
    public_value = _json_public_value(value)
    if public_value is _JSON_OMIT:
        return None
    return _json_inline(public_value)


def _json_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        rendered: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str):
                return _JSON_OMIT
            if _is_sensitive_metadata_key(key):
                continue
            public_item = _json_public_value(item)
            if public_item is _JSON_OMIT:
                return _JSON_OMIT
            rendered[key] = public_item
        if value and not rendered:
            return _JSON_OMIT
        return rendered
    if isinstance(value, (list, tuple)):
        rendered_items: list[Any] = []
        for item in value:
            public_item = _json_public_value(item)
            if public_item is _JSON_OMIT:
                return _JSON_OMIT
            rendered_items.append(public_item)
        return rendered_items
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return _JSON_OMIT


def _mapping_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts: list[str] = []
    for key, item in sorted(value.items()):
        if _is_sensitive_metadata_key(key):
            continue
        rendered = _toml_public_value(item)
        if rendered is not None:
            parts.append(f"`{key}={rendered}`")
    return ", ".join(parts)
