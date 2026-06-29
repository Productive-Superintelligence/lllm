"""Runtime-agnostic parser utilities for tactic and prompt outputs."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParseError(ValueError):
    """Raised when a parser cannot satisfy its declared contract."""


class BaseParser:
    """Minimal parser interface."""

    def parse(self, content: str, **runtime_args: Any) -> dict[str, Any]:
        raise NotImplementedError


def find_xml_blocks(text: str, tag: str) -> list[str]:
    """Return contents of all ``<tag>...</tag>`` blocks."""

    escaped = re.escape(tag)
    return re.findall(rf"<{escaped}>(.*?)</{escaped}>", text, flags=re.DOTALL)


def find_md_blocks(text: str, tag: str) -> list[str]:
    """Return contents of all fenced markdown blocks whose fence starts with tag."""

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

    def model_post_init(self, __context: Any) -> None:
        self.xml_tags = deepcopy(self.xml_tags)
        self.md_tags = deepcopy(self.md_tags)
        self.signal_tags = deepcopy(self.signal_tags)
        self.required_xml_tags = deepcopy(self.required_xml_tags)
        self.required_md_tags = deepcopy(self.required_md_tags)
        self.parser_args = deepcopy(self.parser_args)

    def parse(self, content: str, **runtime_args: Any) -> dict[str, Any]:
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


__all__ = [
    "BaseParser",
    "DefaultTagParser",
    "ParseError",
    "find_md_blocks",
    "find_xml_blocks",
]
