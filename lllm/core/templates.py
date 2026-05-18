from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


MANIFEST_NAME = "lllm-template.toml"
DEFAULT_TEMPLATE = "minimal"

TEXT_EXTENSIONS = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

TEXT_FILENAMES = {
    ".env",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "Makefile",
    "_env",
    "_env.example",
    "_gitignore",
}

DOTFILE_NAMES = {
    "_env": ".env",
    "_env.example": ".env.example",
    "_gitignore": ".gitignore",
}


@dataclass(frozen=True)
class TemplateVariable:
    """Variable declared by a template manifest."""

    required: bool = False
    default: Any = None
    has_default: bool = False


@dataclass(frozen=True)
class TemplateSpec:
    """Parsed metadata from ``lllm-template.toml``."""

    name: str
    version: str = ""
    description: str = ""
    entry: str = "template"
    variables: Mapping[str, TemplateVariable] | None = None

    @classmethod
    def from_toml(cls, data: Mapping[str, Any]) -> "TemplateSpec":
        template = data.get("template", {})
        if not isinstance(template, Mapping):
            raise ValueError("[template] must be a table")

        name = str(template.get("name", "")).strip()
        if not name:
            raise ValueError("[template] name is required")

        raw_variables = data.get("variables", {})
        if raw_variables is None:
            raw_variables = {}
        if not isinstance(raw_variables, Mapping):
            raise ValueError("[variables] must be a table")

        variables: dict[str, TemplateVariable] = {}
        for var_name, raw in raw_variables.items():
            if isinstance(raw, Mapping):
                variables[str(var_name)] = TemplateVariable(
                    required=bool(raw.get("required", False)),
                    default=raw.get("default"),
                    has_default="default" in raw,
                )
            else:
                variables[str(var_name)] = TemplateVariable(
                    default=raw,
                    has_default=True,
                )

        return cls(
            name=name,
            version=str(template.get("version", "")),
            description=str(template.get("description", "")),
            entry=str(template.get("entry", "template")),
            variables=variables,
        )


@dataclass(frozen=True)
class TemplateSource:
    """Resolved template source independent of where it came from."""

    ref: str
    kind: str
    root: Any
    spec: TemplateSpec

    @property
    def template_root(self) -> Any:
        return self.root.joinpath(self.spec.entry)


def resolve_template(ref: str = DEFAULT_TEMPLATE) -> TemplateSource:
    """Resolve a built-in template name or local template directory."""

    ref = ref or DEFAULT_TEMPLATE

    if _is_simple_template_name(ref):
        builtin_root = resources.files("lllm").joinpath("templates", ref)
        if _has_manifest(builtin_root):
            return TemplateSource(
                ref=ref,
                kind="builtin",
                root=builtin_root,
                spec=_read_spec(builtin_root),
            )

    local_root = Path(ref).expanduser()
    if _has_manifest(local_root):
        return TemplateSource(
            ref=str(local_root),
            kind="local",
            root=local_root.resolve(),
            spec=_read_spec(local_root),
        )

    raise FileNotFoundError(
        f"Template '{ref}' not found. Use a built-in template name or a local "
        f"directory containing {MANIFEST_NAME}."
    )


def list_builtin_templates() -> list[TemplateSpec]:
    """Return specs for built-in templates bundled with LLLM."""

    templates_root = resources.files("lllm").joinpath("templates")
    specs: list[TemplateSpec] = []
    for child in templates_root.iterdir():
        if child.is_dir() and _has_manifest(child):
            specs.append(_read_spec(child))
    return sorted(specs, key=lambda spec: spec.name)


def list_builtin_template_names() -> list[str]:
    """Return names of built-in templates bundled with LLLM."""

    return [spec.name for spec in list_builtin_templates()]


def _is_simple_template_name(ref: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9_.-]+", ref) is not None


def render_template(
    source: TemplateSource,
    destination: str | Path,
    variables: Mapping[str, Any],
) -> Path:
    """Render *source* into *destination* using declared manifest variables."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Path '{destination}' already exists.")

    template_root = source.template_root
    if not template_root.is_dir():
        raise FileNotFoundError(
            f"Template '{source.spec.name}' entry directory not found: "
            f"{source.spec.entry}"
        )

    resolved_variables = _resolve_variables(source.spec, variables)
    destination.mkdir(parents=True, exist_ok=False)

    for relative_parts, item in _walk_files(template_root):
        rendered_parts = [
            _render_text(DOTFILE_NAMES.get(part, part), resolved_variables)
            for part in relative_parts
        ]
        target = destination.joinpath(*rendered_parts)
        target.parent.mkdir(parents=True, exist_ok=True)

        if _is_text_file(item.name):
            content = item.read_text(encoding="utf-8")
            target.write_text(_render_text(content, resolved_variables), encoding="utf-8")
        else:
            target.write_bytes(item.read_bytes())

    return destination


def _has_manifest(root: Any) -> bool:
    try:
        return root.joinpath(MANIFEST_NAME).is_file()
    except (FileNotFoundError, NotADirectoryError, AttributeError, OSError):
        return False


def _read_spec(root: Any) -> TemplateSpec:
    manifest = root.joinpath(MANIFEST_NAME)
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    return TemplateSpec.from_toml(data)


def _resolve_variables(
    spec: TemplateSpec,
    provided: Mapping[str, Any],
) -> dict[str, str]:
    variables = spec.variables or {}
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for name, variable in variables.items():
        if name in provided:
            resolved[name] = str(provided[name])
        elif variable.has_default:
            resolved[name] = str(variable.default)
        elif variable.required:
            missing.append(name)
        else:
            resolved[name] = ""

    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing required template variables for '{spec.name}': {missing_list}"
        )

    return resolved


def _walk_files(root: Any, parts: tuple[str, ...] = ()):
    for child in root.iterdir():
        child_parts = parts + (child.name,)
        if child.is_dir():
            yield from _walk_files(child, child_parts)
        elif child.is_file():
            yield child_parts, child


def _is_text_file(filename: str) -> bool:
    return filename in TEXT_FILENAMES or Path(filename).suffix.lower() in TEXT_EXTENSIONS


def _render_text(text: str, variables: Mapping[str, str]) -> str:
    for name, value in variables.items():
        text = text.replace("{{" + name + "}}", value)
    return text


def normalize_package_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower()).strip("_")
    if not normalized:
        normalized = "lllm_app"
    if normalized[0].isdigit():
        normalized = f"app_{normalized}"
    return normalized


def normalize_pyproject_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", name.strip().lower()).strip("-")
    return normalized or "lllm-app"


def title_from_name(name: str) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name.strip()) if w]
    return " ".join(word[:1].upper() + word[1:] for word in words) or "LLLM App"
