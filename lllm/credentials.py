"""API-key requirements and local credential storage for launching packages."""

from __future__ import annotations

import getpass
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal


CONVENTIONAL_API_KEYS: dict[str, str] = {
    "OPENAI_API_KEY": "OpenAI API key used by OpenAI and OpenAI-compatible clients.",
    "ANTHROPIC_API_KEY": "Anthropic API key used by Claude clients.",
    "TOGETHER_API_KEY": "Together AI API key.",
    "GEMINI_API_KEY": "Google Gemini API key.",
    "GROQ_API_KEY": "Groq API key.",
    "MISTRAL_API_KEY": "Mistral API key.",
    "OPENROUTER_API_KEY": "OpenRouter API key.",
    "COHERE_API_KEY": "Cohere API key.",
    "DEEPSEEK_API_KEY": "DeepSeek API key.",
    "PERPLEXITY_API_KEY": "Perplexity API key.",
    "FIREWORKS_API_KEY": "Fireworks API key.",
    "XAI_API_KEY": "xAI API key.",
    "HF_TOKEN": "Hugging Face access token.",
    "TAVILY_API_KEY": "Tavily search API key.",
    "EXA_API_KEY": "Exa search API key.",
}

MetadataKey = Literal[
    "required_api_keys",
    "requires_api_keys",
    "required_env",
]
RESOURCE_METADATA_KEYS: tuple[MetadataKey, ...] = (
    "required_api_keys",
    "requires_api_keys",
    "required_env",
)
CredentialBackend = Literal["auto", "keyring", "env"]
KEYRING_SERVICE = "lllm"


class CredentialError(ValueError):
    """Raised when credentials cannot be read or stored."""


@dataclass(frozen=True)
class ApiKeyRequirement:
    """A named environment credential required by a package."""

    name: str
    description: str = ""
    source: str = "package"


@dataclass(frozen=True)
class ApiKeyStatus:
    """Credential readiness without exposing secret values."""

    requirement: ApiKeyRequirement
    ready: bool
    source: str | None = None


@dataclass(frozen=True)
class CredentialResolution:
    """Resolved requirement set."""

    statuses: tuple[ApiKeyStatus, ...]

    @property
    def missing(self) -> tuple[ApiKeyRequirement, ...]:
        return tuple(status.requirement for status in self.statuses if not status.ready)

    @property
    def ok(self) -> bool:
        return not self.missing

    def public_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "name": status.requirement.name,
                "description": status.requirement.description,
                "source": status.requirement.source,
                "ready": status.ready,
                "credential_source": status.source,
            }
            for status in self.statuses
        ]


def collect_manifest_api_key_requirements(
    manifest: Any,
    *,
    section: str | None = None,
    name: str | None = None,
) -> tuple[ApiKeyRequirement, ...]:
    """Collect package and selected-resource API-key requirements."""

    requirements: list[ApiKeyRequirement] = []
    package_requirements = getattr(manifest, "requirements", None)
    requirements.extend(
        _requirements_from_mapping(
            getattr(package_requirements, "api_keys", None),
            source="requirements.api_keys",
        )
    )
    if section is not None and name is not None:
        requirements.extend(_resource_requirements(manifest, section, name))
    return dedupe_requirements(requirements)


def dedupe_requirements(
    requirements: Iterable[ApiKeyRequirement],
) -> tuple[ApiKeyRequirement, ...]:
    merged: dict[str, ApiKeyRequirement] = {}
    for requirement in requirements:
        _validate_env_name(requirement.name)
        existing = merged.get(requirement.name)
        if existing is None:
            merged[requirement.name] = requirement
            continue
        description = existing.description or requirement.description
        if (
            existing.description
            and requirement.description
            and existing.description != requirement.description
        ):
            description = f"{existing.description} {requirement.description}"
        source = existing.source
        if requirement.source and requirement.source not in source.split(", "):
            source = f"{source}, {requirement.source}"
        merged[requirement.name] = ApiKeyRequirement(
            name=requirement.name,
            description=description,
            source=source,
        )
    return tuple(merged[key] for key in sorted(merged))


def resolve_api_keys(
    requirements: Iterable[ApiKeyRequirement],
    *,
    env_file: str | Path | None = None,
    use_keyring: bool = True,
) -> CredentialResolution:
    """Resolve keys from process env, local env file, and optional keyring."""

    statuses: list[ApiKeyStatus] = []
    env_values = read_env_file(env_file) if env_file is not None else {}
    for requirement in dedupe_requirements(requirements):
        env_value = os.environ.get(requirement.name)
        if env_value:
            statuses.append(ApiKeyStatus(requirement, ready=True, source="env"))
            continue
        file_value = env_values.get(requirement.name)
        if file_value:
            os.environ[requirement.name] = file_value
            statuses.append(ApiKeyStatus(requirement, ready=True, source=str(env_file)))
            continue
        keyring_value = keyring_get(requirement.name) if use_keyring else None
        if keyring_value:
            os.environ[requirement.name] = keyring_value
            statuses.append(ApiKeyStatus(requirement, ready=True, source="keyring"))
            continue
        statuses.append(ApiKeyStatus(requirement, ready=False))
    return CredentialResolution(tuple(statuses))


def store_api_keys(
    values: dict[str, str],
    *,
    credentials: CredentialBackend = "auto",
    env_file: str | Path = ".env.local",
) -> str:
    """Store values in keyring or a local env file and return the backend name."""

    clean_values = _validated_secret_values(values)
    if not clean_values:
        return "none"
    backend = choose_credential_backend(credentials)
    if backend == "keyring":
        for name, value in clean_values.items():
            keyring_set(name, value)
        return "keyring"
    write_env_file(env_file, clean_values)
    return str(env_file)


def configure_missing_api_keys(
    requirements: Iterable[ApiKeyRequirement],
    *,
    credentials: CredentialBackend = "auto",
    env_file: str | Path = ".env.local",
    provided: dict[str, str] | None = None,
    input_func: Callable[[str], str] = input,
    secret_func: Callable[[str], str] = getpass.getpass,
) -> str:
    """Prompt for missing keys and persist them through the selected backend."""

    provided_values = _validated_secret_values(provided or {})
    requirements = dedupe_requirements(requirements)
    values: dict[str, str] = {}
    for requirement in requirements:
        if requirement.name in provided_values:
            values[requirement.name] = provided_values[requirement.name]
            continue
        prompt = _prompt_for_requirement(requirement)
        answer = secret_func(prompt)
        if answer:
            values[requirement.name] = answer.strip()
        else:
            confirm = input_func(f"Skip {requirement.name}? [y/N] ").strip().lower()
            if confirm not in {"y", "yes"}:
                raise CredentialError(f"missing value for {requirement.name}")
    return store_api_keys(values, credentials=credentials, env_file=env_file)


def read_env_file(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}
    if env_path.is_symlink() or not env_path.is_file():
        raise CredentialError("env file must be a regular file")
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is not None:
            key, value = parsed
            values[key] = value
    return values


def write_env_file(path: str | Path, values: dict[str, str]) -> None:
    env_path = Path(path).expanduser()
    if env_path.exists() and (env_path.is_symlink() or not env_path.is_file()):
        raise CredentialError("env file must be a regular file")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = (
        env_path.read_text(encoding="utf-8").splitlines()
        if env_path.exists()
        else []
    )
    pending = dict(_validated_secret_values(values))
    rendered: list[str] = []
    for line in existing_lines:
        parsed = _parse_env_line(line)
        if parsed is None:
            rendered.append(line)
            continue
        key, _ = parsed
        if key in pending:
            rendered.append(f"{key}={_quote_env_value(pending.pop(key))}")
        else:
            rendered.append(line)
    if pending and rendered and rendered[-1] != "":
        rendered.append("")
    for key, value in sorted(pending.items()):
        rendered.append(f"{key}={_quote_env_value(value)}")
    fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rendered).rstrip() + "\n")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass


def keyring_available() -> bool:
    try:
        import keyring  # noqa: F401
    except Exception:
        return False
    return True


def keyring_get(name: str) -> str | None:
    _validate_env_name(name)
    try:
        import keyring

        value = keyring.get_password(KEYRING_SERVICE, name)
    except Exception:
        return None
    if isinstance(value, str) and value:
        return value
    return None


def keyring_set(name: str, value: str) -> None:
    _validate_env_name(name)
    if not value:
        raise CredentialError(f"{name} cannot be empty")
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, name, value)
    except Exception as exc:
        raise CredentialError(
            "keyring is not available; use --credentials env to write a local env file"
        ) from exc


def choose_credential_backend(credentials: CredentialBackend) -> Literal["keyring", "env"]:
    if credentials == "keyring":
        return "keyring"
    if credentials == "env":
        return "env"
    if credentials != "auto":
        raise CredentialError("credentials must be one of: auto, keyring, env")
    return "keyring" if keyring_available() else "env"


def parse_key_values(raw_values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_value in raw_values:
        key, sep, value = raw_value.partition("=")
        if not sep:
            raise CredentialError("--set values must have shape NAME=value")
        key = key.strip()
        _validate_env_name(key)
        if not value:
            raise CredentialError(f"{key} cannot be empty")
        parsed[key] = value
    return parsed


def conventional_requirements() -> tuple[ApiKeyRequirement, ...]:
    return tuple(
        ApiKeyRequirement(name=key, description=description, source="convention")
        for key, description in sorted(CONVENTIONAL_API_KEYS.items())
    )


def format_missing_requirements(requirements: Iterable[ApiKeyRequirement]) -> str:
    lines = []
    for requirement in dedupe_requirements(requirements):
        suffix = f" - {requirement.description}" if requirement.description else ""
        lines.append(f"- {requirement.name}{suffix}")
    return "\n".join(lines)


def _resource_requirements(
    manifest: Any,
    section: str,
    name: str,
) -> tuple[ApiKeyRequirement, ...]:
    resources = getattr(manifest, section, {})
    resource = resources.get(name) if isinstance(resources, dict) else None
    requirements: list[ApiKeyRequirement] = []
    requirements.extend(_metadata_requirements(getattr(resource, "metadata", {}), source=f"{section}.{name}.metadata"))
    if section == "services" and resource is not None:
        tactic = getattr(resource, "tactic", None)
        if isinstance(tactic, str) and tactic:
            requirements.extend(_resource_requirements(manifest, "tactics", tactic))
        for tactic_name in getattr(resource, "tactics", ()) or ():
            if isinstance(tactic_name, str) and tactic_name:
                requirements.extend(_resource_requirements(manifest, "tactics", tactic_name))
    if section == "runs" and resource is not None:
        for child_section in ("services", "tactics", "channels", "snapshots"):
            for child_name in getattr(resource, child_section, ()) or ():
                if isinstance(child_name, str) and child_name:
                    requirements.extend(
                        _resource_requirements(manifest, child_section, child_name)
                    )
    return tuple(requirements)


def _metadata_requirements(
    metadata: Any,
    *,
    source: str,
) -> tuple[ApiKeyRequirement, ...]:
    if not isinstance(metadata, dict):
        return ()
    requirements: list[ApiKeyRequirement] = []
    for key in RESOURCE_METADATA_KEYS:
        requirements.extend(
            _requirements_from_mapping(
                metadata.get(key),
                source=f"{source}.{key}",
            )
        )
    return tuple(requirements)


def _requirements_from_mapping(
    value: Any,
    *,
    source: str,
) -> tuple[ApiKeyRequirement, ...]:
    if value in (None, "", [], {}):
        return ()
    if isinstance(value, dict):
        requirements = []
        for key, description in value.items():
            if not isinstance(key, str):
                raise CredentialError("API key requirement names must be strings")
            _validate_env_name(key)
            requirements.append(
                ApiKeyRequirement(
                    name=key,
                    description=_requirement_description(key, description),
                    source=source,
                )
            )
        return tuple(requirements)
    if isinstance(value, (list, tuple)):
        requirements = []
        for item in value:
            if not isinstance(item, str):
                raise CredentialError("API key requirement lists must contain strings")
            _validate_env_name(item)
            requirements.append(
                ApiKeyRequirement(
                    name=item,
                    description=CONVENTIONAL_API_KEYS.get(item, ""),
                    source=source,
                )
            )
        return tuple(requirements)
    raise CredentialError("API key requirements must be a map or list")


def _requirement_description(name: str, value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        description = value.get("description")
        if isinstance(description, str):
            return description
    return CONVENTIONAL_API_KEYS.get(name, "")


def _validated_secret_values(values: dict[str, str]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in values.items():
        _validate_env_name(key)
        if not isinstance(value, str) or not value:
            raise CredentialError(f"{key} cannot be empty")
        clean[key] = value
    return clean


def _validate_env_name(name: str) -> None:
    if not isinstance(name, str) or re.match(r"^[A-Z][A-Z0-9_]*$", name) is None:
        raise CredentialError("API key names must be uppercase environment names")


def _prompt_for_requirement(requirement: ApiKeyRequirement) -> str:
    detail = f" ({requirement.description})" if requirement.description else ""
    return f"{requirement.name}{detail}: "


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()
    key, sep, value = stripped.partition("=")
    if not sep:
        return None
    key = key.strip()
    try:
        _validate_env_name(key)
    except CredentialError:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = _unquote_env_value(value)
    return key, value


def _quote_env_value(value: str) -> str:
    return json.dumps(value)


def _unquote_env_value(value: str) -> str:
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
        return decoded if isinstance(decoded, str) else value[1:-1]
    return value[1:-1]
