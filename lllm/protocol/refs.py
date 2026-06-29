"""Resource refs used for tactic composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

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
        org = parsed.netloc.strip()
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
        if not org or not package or not name:
            raise TacticRefError(f"Tactic ref contains an empty segment: {self.value}")
        for segment in (org, package, name):
            if segment in {".", ".."} or any(ch in segment for ch in ":\\"):
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
