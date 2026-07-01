"""Resource refs used for tactic composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

from .errors import TacticRefError


@dataclass(frozen=True)
class TacticRef:
    """A stable `psi://org/package/tactics/name` tactic reference."""

    value: str
    org: str = field(init=False)
    package: str = field(init=False)
    resource_kind: str = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise TacticRefError("Tactic ref must be a non-empty string.")
        parsed = urlparse(self.value)
        if parsed.scheme != "psi":
            raise TacticRefError(f"Tactic ref must use psi:// scheme: {self.value}")
        if parsed.params or parsed.query or parsed.fragment:
            raise TacticRefError(
                "Tactic ref must not include params, query, or fragment: "
                f"{self.value}"
            )
        org = parsed.netloc
        raw_parts = parsed.path.split("/")
        if (
            len(raw_parts) != 4
            or raw_parts[0] != ""
            or any(not part for part in raw_parts[1:])
        ):
            raise TacticRefError(
                "Tactic ref must have shape psi://org/package/tactics/name: "
                f"{self.value}"
            )
        package, resource_kind, name = raw_parts[1:]
        if not org or not package.strip() or not name.strip():
            raise TacticRefError(f"Tactic ref contains an empty segment: {self.value}")
        for segment in (org, package, resource_kind, name):
            decoded_segment = unquote(segment)
            if any(ch.isspace() for ch in decoded_segment):
                raise TacticRefError(
                    "Tactic ref contains a whitespace-bearing segment: "
                    f"{self.value}"
                )
        for segment in (org, package, name):
            decoded_segment = unquote(segment)
            if (
                decoded_segment in {".", ".."}
                or any(ch in decoded_segment for ch in "/:\\")
                or "%" in segment
            ):
                raise TacticRefError(
                    f"Tactic ref contains an invalid segment: {self.value}"
                )
        if resource_kind != "tactics":
            raise TacticRefError(
                f"Tactic ref must point at /tactics/, got /{resource_kind}/: {self.value}"
            )
        object.__setattr__(self, "org", org)
        object.__setattr__(self, "package", package)
        object.__setattr__(self, "resource_kind", resource_kind)
        object.__setattr__(self, "name", name)

    @classmethod
    def parse(cls, value: str | "TacticRef") -> "TacticRef":
        if isinstance(value, cls):
            return value
        return cls(value)

    def __str__(self) -> str:
        return self.value


def tactic_ref_value(value: str, label: str = "tactic ref") -> str:
    """Return a normalized tactic ref string or raise ``ValueError``."""

    try:
        return str(TacticRef(value))
    except TacticRefError as exc:
        raise ValueError(f"{label} must be a valid tactic ref: {exc}") from exc


def optional_tactic_ref_value(
    value: str | None,
    label: str = "tactic ref",
) -> str | None:
    if value is None:
        return None
    return tactic_ref_value(value, label)


def service_ref_value(value: str, label: str = "service_ref") -> str:
    """Return a normalized service ref string or raise ``ValueError``."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty service ref.")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{label} must not contain whitespace.")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            raise ValueError(f"{label} must be an absolute HTTP(S) URL.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{label} must not include embedded credentials.")
        if (
            ";" in parsed.netloc
            or parsed.params
            or ";" in parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"{label} must not include URL params, query, or fragment."
            )
        _validate_http_service_url_path(parsed.netloc, parsed.path, label)
        return value
    if parsed.scheme != "psi":
        raise ValueError(f"{label} must be a psi:// service ref or HTTP(S) URL.")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not include params, query, or fragment.")
    raw_parts = parsed.path.split("/")
    if (
        not parsed.netloc
        or len(raw_parts) != 4
        or raw_parts[0] != ""
        or any(not part for part in raw_parts[1:])
    ):
        raise ValueError(f"{label} must have shape psi://org/package/services/name.")
    package, resource_kind, name = raw_parts[1:]
    if resource_kind != "services":
        raise ValueError(f"{label} must point at /services/.")
    for segment in (parsed.netloc, package, resource_kind, name):
        decoded_segment = unquote(segment)
        if any(ch.isspace() for ch in decoded_segment):
            raise ValueError(f"{label} contains a whitespace-bearing segment.")
    for segment in (parsed.netloc, package, name):
        decoded_segment = unquote(segment)
        if (
            decoded_segment in {".", ".."}
            or any(ch in decoded_segment for ch in "/:\\")
            or "%" in segment
        ):
            raise ValueError(f"{label} contains an invalid segment.")
    return value


def _validate_http_service_url_path(netloc: str, path: str, label: str) -> None:
    if "%" in netloc or "%" in path:
        raise ValueError(f"{label} must not contain percent escapes.")
    if any(ch in path for ch in "\\:"):
        raise ValueError(
            f"{label} URL path must not contain backslashes or colons."
        )
    if path in {"", "/"}:
        return
    if "//" in path:
        raise ValueError(f"{label} URL path must not contain empty segments.")
    trimmed = path.rstrip("/")
    if any(part in {".", ".."} for part in trimmed.split("/")[1:]):
        raise ValueError(f"{label} URL path must not contain dot segments.")


def optional_service_ref_value(
    value: str | None,
    label: str = "service_ref",
) -> str | None:
    if value is None:
        return None
    return service_ref_value(value, label)
