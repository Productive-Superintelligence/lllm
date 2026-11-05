"""Endpoint metadata validation shared by validators and cards."""

from __future__ import annotations

from typing import Any

from .models import ValidationIssue

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
ENDPOINT_MODES = {"run", "stream", "events"}
ENDPOINT_SCOPES = {"store", "channel", "subscription", "artifact", "snapshot"}


def validate_endpoint_metadata(
    resource_model: Any,
    resource: str,
) -> list[ValidationIssue]:
    endpoints = resource_extra(resource_model).get("endpoints")
    metadata = getattr(resource_model, "metadata", None)
    if endpoints is None and isinstance(metadata, dict):
        endpoints = metadata.get("endpoints")
    if endpoints is None:
        return []
    if not isinstance(endpoints, list):
        return [
            ValidationIssue(
                level="error",
                code="endpoint_metadata_invalid",
                message="Endpoint metadata must be a list.",
                resource=resource,
            )
        ]
    issues: list[ValidationIssue] = []
    for index, endpoint in enumerate(endpoints, start=1):
        if not isinstance(endpoint, dict):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="endpoint_metadata_invalid",
                    message=f"Endpoint #{index} must be a table/object.",
                    resource=resource,
                )
            )
            continue
        method_value = endpoint.get("method")
        method = (
            method_value.upper()
            if isinstance(method_value, str)
            and method_value
            and not any(ch.isspace() for ch in method_value)
            else ""
        )
        if method not in HTTP_METHODS:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="endpoint_method_invalid",
                    message=f"Endpoint #{index} declares invalid method {method!r}.",
                    resource=resource,
                )
            )
        path = endpoint.get("path")
        if not valid_endpoint_path(path):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="endpoint_path_invalid",
                    message=(
                        f"Endpoint #{index} path must be an absolute route path "
                        "without whitespace, percent escapes, URL syntax, "
                        "queries, fragments, network-path prefixes, empty or "
                        "dot segments, backslashes, colons, or path params."
                    ),
                    resource=resource,
                )
            )
        name = endpoint.get("name")
        if name is not None and not valid_endpoint_label(name):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="endpoint_name_invalid",
                    message=(
                        f"Endpoint #{index} declares invalid name {name!r}; "
                        "names must not contain whitespace or percent escapes."
                    ),
                    resource=resource,
                )
            )
        mode = endpoint.get("mode")
        if mode is not None and mode not in ENDPOINT_MODES:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="endpoint_mode_invalid",
                    message=f"Endpoint #{index} declares invalid mode {mode!r}.",
                    resource=resource,
                )
            )
        scope = endpoint.get("scope")
        if scope is not None and scope not in ENDPOINT_SCOPES:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="endpoint_scope_invalid",
                    message=f"Endpoint #{index} declares invalid scope {scope!r}.",
                    resource=resource,
                )
            )
        description = endpoint.get("description")
        if description is not None and not isinstance(description, str):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="endpoint_description_invalid",
                    message=f"Endpoint #{index} description must be a string.",
                    resource=resource,
                )
            )
        tags = endpoint.get("tags")
        if tags is not None and not valid_endpoint_tags(tags):
            issues.append(
                ValidationIssue(
                    level="error",
                    code="endpoint_tags_invalid",
                    message=(
                        f"Endpoint #{index} tags must be non-empty strings "
                        "without whitespace or percent escapes."
                    ),
                    resource=resource,
                )
            )
    return issues


def valid_endpoint_path(path: Any) -> bool:
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or path.startswith("//")
        or "%" in path
        or any(ch.isspace() for ch in path)
        or "?" in path
        or "#" in path
        or "://" in path
        or any(ch in path for ch in "\\:;")
    ):
        return False
    return not any(part in {"", ".", ".."} for part in path.split("/")[1:])


def valid_endpoint_label(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "%" not in value
        and not any(ch.isspace() for ch in value)
    )


def valid_endpoint_tags(tags: Any) -> bool:
    return (
        isinstance(tags, list)
        and all(valid_endpoint_label(tag) for tag in tags)
    )


def resource_extra(resource_model: Any) -> dict[str, Any]:
    return dict(getattr(resource_model, "model_extra", None) or {})
