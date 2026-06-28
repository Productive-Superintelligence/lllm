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
        parsed = urlparse(self.value)
        if parsed.scheme != "psi":
            raise TacticRefError(f"Tactic ref must use psi:// scheme: {self.value}")
        org = parsed.netloc.strip()
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 3:
            raise TacticRefError(
                "Tactic ref must have shape psi://org/package/tactics/name: "
                f"{self.value}"
            )
        package, resource_kind, name = parts
        if not org or not package or not name:
            raise TacticRefError(f"Tactic ref contains an empty segment: {self.value}")
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
        return cls(str(value))

    def __str__(self) -> str:
        return self.value
