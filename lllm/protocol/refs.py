"""Resource refs used for tactic composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

from .errors import TacticRefError


#: Package sections a ref may point at: lllm://org/package/<section>/name.
REF_SECTIONS = frozenset(
    {
        "schemas",
        "tactics",
        "services",
        "channels",
        "snapshots",
        "runs",
        "configs",
        "docs",
        "examples",
        "assets",
    }
)


@dataclass(frozen=True)
class ResourceRef:
    """Parsed parts of an ``lllm://org/package/section/name`` ref."""

    org: str
    package: str
    resource_kind: str
    name: str

    def __str__(self) -> str:
        return f"lllm://{self.org}/{self.package}/{self.resource_kind}/{self.name}"


def parse_resource_ref(
    value: str,
    *,
    noun: str = "Ref",
    sections: tuple[str, ...] | None = None,
    error: type[Exception] = ValueError,
) -> ResourceRef:
    """Parse any ``lllm://`` resource ref.

    ``sections`` restricts the ref to particular package sections (for example
    ``("tactics",)``); by default any of :data:`REF_SECTIONS` is accepted.
    Errors are raised as ``error`` with messages that start with ``noun``.
    """

    if not isinstance(value, str) or not value.strip():
        raise error(f"{noun} must be a non-empty string.")
    parsed = urlparse(value)
    if parsed.scheme != "lllm":
        raise error(f"{noun} must use lllm:// scheme: {value}")
    if parsed.params or parsed.query or parsed.fragment:
        raise error(f"{noun} must not include params, query, or fragment: {value}")
    org = parsed.netloc
    raw_parts = parsed.path.split("/")
    if len(raw_parts) != 4 or raw_parts[0] != "" or any(not part for part in raw_parts[1:]):
        raise error(f"{noun} must have shape {_ref_shape(sections)}: {value}")
    package, resource_kind, name = raw_parts[1:]
    if not org or not package.strip() or not name.strip():
        raise error(f"{noun} contains an empty segment: {value}")
    for segment in (org, package, resource_kind, name):
        if any(ch.isspace() for ch in unquote(segment)):
            raise error(f"{noun} contains a whitespace-bearing segment: {value}")
    if sections is None and resource_kind not in REF_SECTIONS:
        raise error(f"{noun} uses unknown resource section {resource_kind!r}: {value}")
    for segment in (org, package, name):
        decoded_segment = unquote(segment)
        if (
            decoded_segment in {".", ".."}
            or any(ch in decoded_segment for ch in "/:\\;")
            or "%" in segment
        ):
            raise error(f"{noun} contains an invalid segment: {value}")
    if sections is not None and resource_kind not in sections:
        expected = " or ".join(f"/{section}/" for section in sections)
        raise error(f"{noun} must point at {expected}, got /{resource_kind}/: {value}")
    return ResourceRef(org=org, package=package, resource_kind=resource_kind, name=name)


def _ref_shape(sections: tuple[str, ...] | None) -> str:
    if sections is None:
        return "lllm://org/package/resources/name"
    return " or ".join(f"lllm://org/package/{section}/name" for section in sections)


@dataclass(frozen=True)
class TacticRef:
    """A stable `lllm://org/package/tactics/name` tactic reference."""

    value: str
    org: str = field(init=False)
    package: str = field(init=False)
    resource_kind: str = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        ref = parse_resource_ref(
            self.value,
            noun="Tactic ref",
            sections=("tactics",),
            error=TacticRefError,
        )
        object.__setattr__(self, "org", ref.org)
        object.__setattr__(self, "package", ref.package)
        object.__setattr__(self, "resource_kind", ref.resource_kind)
        object.__setattr__(self, "name", ref.name)

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
    if parsed.scheme != "lllm":
        raise ValueError(f"{label} must be a lllm:// service ref or HTTP(S) URL.")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not include params, query, or fragment.")
    raw_parts = parsed.path.split("/")
    if (
        not parsed.netloc
        or len(raw_parts) != 4
        or raw_parts[0] != ""
        or any(not part for part in raw_parts[1:])
    ):
        raise ValueError(f"{label} must have shape lllm://org/package/services/name.")
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
            or any(ch in decoded_segment for ch in "/:\\;")
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
