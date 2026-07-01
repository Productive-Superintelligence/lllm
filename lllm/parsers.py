"""Runtime-agnostic parser utilities for tactic and prompt outputs."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .protocol._validation import copy_boundary_value, mapping_field_value


class ParseError(ValueError):
    """Raised when a parser cannot satisfy its declared contract."""


class BaseParser:
    """Minimal parser interface."""

    def parse(self, content: str, **runtime_args: Any) -> dict[str, Any]:
        raise NotImplementedError


def find_xml_blocks(text: str, tag: str) -> list[str]:
    """Return contents of all ``<tag>...</tag>`` blocks."""

    text = _require_text(text)
    tag = _require_tag(tag)
    escaped = re.escape(tag)
    return re.findall(rf"<{escaped}>(.*?)</{escaped}>", text, flags=re.DOTALL)


def find_md_blocks(text: str, tag: str) -> list[str]:
    """Return contents of all fenced markdown blocks whose fence starts with tag."""

    text = _require_text(text)
    tag = _require_tag(tag)
    escaped = re.escape(tag)
    pattern = rf"```{escaped}(?:[ \t]*\n|\s)(.*?)```"
    return [match.strip() for match in re.findall(pattern, text, flags=re.DOTALL)]


class DefaultTagParser(BaseParser, BaseModel):
    """Extract XML blocks, fenced markdown blocks, and signal tags."""

    xml_tags: list[str] = Field(default_factory=list)
    md_tags: list[str] = Field(default_factory=list)
    signal_tags: list[str] = Field(default_factory=list)
    required_xml_tags: list[str] = Field(default_factory=list)
    required_md_tags: list[str] = Field(default_factory=list)
    parser_args: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator(
        "xml_tags",
        "md_tags",
        "signal_tags",
        "required_xml_tags",
        "required_md_tags",
        mode="before",
    )
    @classmethod
    def _validate_tag_list(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            return value
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{info.field_name} must be a list of tag strings")
        validated = []
        for tag in value:
            try:
                validated.append(_require_tag(tag, label=f"{info.field_name} item"))
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
        return validated

    @field_validator("parser_args", mode="before")
    @classmethod
    def _validate_parser_args(cls, value: Any) -> Any:
        return mapping_field_value("parser_args", value)

    def model_post_init(self, __context: Any) -> None:
        self.xml_tags = copy_boundary_value(self.xml_tags)
        self.md_tags = copy_boundary_value(self.md_tags)
        self.signal_tags = copy_boundary_value(self.signal_tags)
        self.required_xml_tags = copy_boundary_value(self.required_xml_tags)
        self.required_md_tags = copy_boundary_value(self.required_md_tags)
        self.parser_args = copy_boundary_value(self.parser_args)

    def parse(self, content: str, **runtime_args: Any) -> dict[str, Any]:
        content = _require_text(content, label="content")
        xml_tags = _unique([*self.xml_tags, *self.required_xml_tags])
        md_tags = _unique([*self.md_tags, *self.required_md_tags])
        xml_blocks = {tag: find_xml_blocks(content, tag) for tag in xml_tags}
        md_blocks = {tag: find_md_blocks(content, tag) for tag in md_tags}
        errors = []
        for tag in self.required_xml_tags:
            if not xml_blocks.get(tag):
                errors.append(f"Missing required XML tag: <{tag}>...</{tag}>")
        for tag in self.required_md_tags:
            if not md_blocks.get(tag):
                errors.append(f"Missing required markdown block: ```{tag}")
        if errors:
            raise ParseError("Parsing errors:\n" + "\n".join(errors))
        return {
            "raw": content,
            "xml_tags": xml_blocks,
            "md_tags": md_blocks,
            "signal_tags": {
                tag: f"<{tag}>" in content for tag in self.signal_tags
            },
        }


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _require_text(value: Any, *, label: str = "text") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _require_tag(value: Any, *, label: str = "tag") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if any(character.isspace() for character in value):
        raise ValueError(f"{label} must not contain whitespace")
    return value


__all__ = [
    "BaseParser",
    "DefaultTagParser",
    "ParseError",
    "find_md_blocks",
    "find_xml_blocks",
]
