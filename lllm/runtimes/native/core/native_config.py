"""Native LLLM runtime configuration.

This module owns config keys consumed by LLLM's native Agent/Prompt/Dialog
runtime. Generic package loading, resource discovery, and tactic profile
inheritance stay in :mod:`lllm.core.config`.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import _deep_merge
from .runtime import Runtime

logger = logging.getLogger(__name__)

__all__ = [
    "AgentSpec",
    "ContextManagerConfig",
    "ProxyConfig",
    "SkillsConfig",
    "parse_agent_configs",
]

# ---------------------------------------------------------------------------
# AgentSpec — config → agent intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class ProxyConfig:
    """
    Configuration for proxy-based tool calling on an agent.

    Settable globally (under the ``global`` key in tactic config) and
    overridable per-agent.  When both are present, the per-agent dict is
    deep-merged on top of the global one, so agents can override individual
    fields (e.g. swap ``exec_env``) without repeating everything.

    Config format (YAML)::

        proxy:
          activate_proxies: [fmp, fred]   # which proxies to load; empty = all
          deploy_mode: false               # passed through to proxy instances
          cutoff_date: "2024-01-01"        # ISO date; restricts data range
          exec_env: interpreter      # "interpreter" | "jupyter" | null
          max_output_chars: 5000           # truncate run_python output (interpreter only)
          truncation_indicator: "... (truncated)"
          timeout: 60.0                    # seconds before TimeoutError (interpreter only)
          prompt_template: null            # override auto-selected system-prompt block

    **exec_env values**

    ``"interpreter"`` (default)
        Agent calls ``run_python`` tool.  LLLM runs code in a lightweight
        in-process :class:`~lllm.proxies.interpreter.AgentInterpreter` with a
        persistent namespace.  Parallel-safe, zero subprocess overhead.

    ``"jupyter"``
        Agent writes ``<python_cell>`` / ``<markdown_cell>`` XML tags.  Your
        tactic extracts these and runs them via
        :class:`~lllm.sandbox.jupyter.JupyterSession`.  Only ``query_api_doc``
        is injected as a tool; ``run_python`` is **not** added.

    ``null`` (or any unrecognised string)
        No execution tool injected.  Useful when the agent only needs API
        awareness (``query_api_doc`` + directory in prompt) but execution is
        handled externally or not needed at all.

    Future sandbox types (e.g. ``"docker"``, ``"wasm"``) can be added by
    extending the tactic — the string passes through without validation.
    """

    activate_proxies: List[str] = field(default_factory=list)
    deploy_mode: bool = False
    cutoff_date: Optional[str] = None  # ISO date string e.g. "2024-01-01"
    exec_env: Optional[str] = "interpreter"  # "interpreter" | "jupyter" | None
    max_output_chars: int = 5000
    truncation_indicator: str = "... (truncated)"
    timeout: float = 60.0
    prompt_template: Optional[str] = None  # None → auto-select based on exec_env

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProxyConfig":
        return cls(
            activate_proxies=d.get("activate_proxies", []),
            deploy_mode=d.get("deploy_mode", False),
            cutoff_date=d.get("cutoff_date", None),
            exec_env=d.get("exec_env", "interpreter"),
            max_output_chars=d.get("max_output_chars", 5000),
            truncation_indicator=d.get("truncation_indicator", "... (truncated)"),
            timeout=d.get("timeout", 60.0),
            prompt_template=d.get("prompt_template", None),
        )


@dataclass
class ContextManagerConfig:
    """
    Configuration for context-window management on an agent.

    Settable globally (under the ``global`` key in tactic config) and
    overridable per-agent.  When both are present the per-agent dict is
    deep-merged on top of the global one.

    Config format (YAML)::

        context_manager:
          type: default       # "default" → DefaultContextManager; null → disabled
          max_tokens: 128000  # optional hard cap; omit to auto-detect from litellm

    **type values**

    ``"default"`` (built-in)
        Uses :class:`~lllm.core.dialog.DefaultContextManager`: drops/truncates
        old messages so total tokens stay within the model's context window.

    Custom string (e.g. ``"summary"``)
        Looked up in the runtime via
        :meth:`~lllm.core.runtime.Runtime.get_context_manager`.  Register your
        class first::

            runtime.register_context_manager("summary", SummaryCompressor)

    ``null`` / omitted
        Context management disabled for this agent.
    """

    type: str = "default"
    max_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextManagerConfig":
        return cls(
            type=d.get("type", "default"),
            max_tokens=d.get("max_tokens", None),
        )

    def build(self, model_name: str, runtime: Runtime):
        """Instantiate and return the configured :class:`~lllm.core.dialog.ContextManager`."""
        from .dialog import DefaultContextManager

        if self.type in (None, "null", "none"):
            return None
        if self.type == "default":
            return DefaultContextManager(
                model_name=model_name, max_tokens=self.max_tokens
            )
        # Custom type registered in the runtime
        cm_cls = runtime.get_context_manager(self.type)
        return cm_cls(model_name=model_name, max_tokens=self.max_tokens)


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split *text* into ``(frontmatter_dict, body)``."""
    import re

    import yaml as _yaml

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not fm_match:
        return {}, text
    try:
        frontmatter = _yaml.safe_load(fm_match.group(1)) or {}
    except Exception:
        frontmatter = {}
    body = text[fm_match.end() :]
    return frontmatter, body


def _parse_skill_md(path: Path) -> Dict[str, Any]:
    """Parse a ``SKILL.md`` file.

    Returns a dict with keys:
      ``name``          — from frontmatter, falls back to directory name
      ``description``   — from frontmatter (empty string if missing)
      ``allowed_tools`` — list parsed from the ``allowed-tools`` space-delimited field
      ``body``          — Markdown body with frontmatter stripped
      ``skill_dir``     — ``Path`` of the skill's root directory
    """
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(text)

    raw_tools = frontmatter.get("allowed-tools", "") or ""
    allowed_tools = raw_tools.split() if isinstance(raw_tools, str) else list(raw_tools)

    return {
        "name": frontmatter.get("name") or path.parent.name,
        "description": frontmatter.get("description", ""),
        "allowed_tools": allowed_tools,
        "body": body.strip(),
        "skill_dir": path.parent,
    }


def _discover_skills(project_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """Scan standard skill directories and return ``{name: skill_dict}``."""
    if project_dir is None:
        project_dir = Path.cwd()
    home = Path.home()

    search_dirs = [
        project_dir / ".agents" / "skills",
        project_dir / ".claude" / "skills",
        home / ".agents" / "skills",
        home / ".claude" / "skills",
    ]

    skills: Dict[str, Dict] = {}
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for skill_dir in sorted(search_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                skill = _parse_skill_md(skill_md)
                name = skill["name"]
                if name not in skills:  # project-level takes precedence
                    skills[name] = skill
            except Exception as exc:
                logger.warning("Failed to parse skill at %s: %s", skill_md, exc)
    return skills


def _is_skill_id(s: str) -> bool:
    """Return True if *s* looks like an Anthropic-hosted skill ID (``skill_…``)."""
    return s.startswith("skill_")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _fetch_skill_from_url(url: str) -> Optional[Dict[str, Any]]:
    """Download a ``SKILL.md`` from *url* and parse it.  Returns None on failure."""
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
        stem = Path(url.rstrip("/").rsplit("/", 1)[-1]).stem
        frontmatter, body = _parse_frontmatter(text)
        raw_tools = frontmatter.get("allowed-tools", "") or ""
        allowed_tools = (
            raw_tools.split() if isinstance(raw_tools, str) else list(raw_tools)
        )
        return {
            "name": frontmatter.get("name") or stem,
            "description": frontmatter.get("description", ""),
            "allowed_tools": allowed_tools,
            "body": body.strip(),
            "skill_dir": None,  # no local directory for URL skills
        }
    except Exception as exc:
        logger.warning("Failed to fetch skill from %s: %s", url, exc)
        return None


def _list_skill_resources(skill_dir: Optional[Path]) -> List[str]:
    """Return relative paths of bundled resource files under *skill_dir*."""
    if skill_dir is None or not skill_dir.is_dir():
        return []
    resources = []
    for p in sorted(skill_dir.rglob("*")):
        if p.name == "SKILL.md" or not p.is_file():
            continue
        rel = p.relative_to(skill_dir)
        # Skip hidden files and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        resources.append(str(rel))
    return resources


def make_activate_skill_tool(skills: Dict[str, Dict]) -> "Function":
    """Return an ``activate_skill`` :class:`~lllm.core.prompt.Function` tool.

    When the model calls ``activate_skill(name="pdf")``, it receives the full
    ``SKILL.md`` body plus a listing of any bundled resource files.  The model
    can then load those files using its standard file-reading capability.

    *skills* is the ``{name: skill_dict}`` map built by
    :meth:`SkillsConfig._resolve_text_skills`.
    """
    from .prompt import Function

    skill_names = sorted(skills.keys())

    def activate_skill(name: str) -> str:
        """Load the full instructions for a skill by name."""
        if name not in skills:
            return f"Skill '{name}' not found. Available skills: {skill_names}"
        skill = skills[name]
        parts = [f'<skill_content name="{name}">']
        parts.append(skill["body"])

        if skill["skill_dir"] is not None:
            parts.append(f"\nSkill directory: {skill['skill_dir']}")
            parts.append(
                "Relative paths in this skill resolve against the skill directory."
            )

        resources = _list_skill_resources(skill["skill_dir"])
        if resources:
            parts.append("\n<skill_resources>")
            for r in resources:
                parts.append(f"  <file>{r}</file>")
            parts.append("</skill_resources>")

        if skill["allowed_tools"]:
            parts.append(
                f"\nNote: This skill declares tool requirements: "
                f"{' '.join(skill['allowed_tools'])}. "
                "Ensure the agent has access to these tools."
            )

        parts.append("</skill_content>")
        return "\n".join(parts)

    description = (
        "Load the full instructions for an available skill. "
        "Call this when a task matches a skill's description — before attempting "
        "to perform the task — to get detailed step-by-step guidance. "
        f"Available skills: {skill_names}."
    )

    return Function.from_callable(
        activate_skill,
        description=description,
        prop_desc={"name": f"Skill name to load. One of: {skill_names}."},
    )


@dataclass
class SkillsConfig:
    """
    Configuration for agent skills following the `agentskills.io <https://agentskills.io>`_ standard.

    Settable globally (under the ``global`` key in tactic config) and
    overridable per-agent.  When both are present the per-agent list
    replaces the global one (skills lists are not merged).

    Config format (YAML)::

        skills: [pdf, commit, review-pr]   # local names, auto-discovered
        # OR
        skills: "*"                         # all locally discovered skills
        # OR — mixed, each entry is auto-classified:
        skills:
          - local-skill            # local name → scan .agents/skills/ etc.
          - skill_01abc123         # Anthropic-hosted ID (starts with "skill_")
          - https://example.com/SKILL.md   # remote URL → fetched at build time

    **Entry types (auto-detected):**

    ``skill_<id>`` (starts with ``skill_``)
        Anthropic-hosted skill.  Passed directly to the API as
        ``skills=[{"id": "skill_01abc"}]`` with the required
        ``anthropic-beta: skills-2025-10-02`` header.  Anthropic injects
        the skill content server-side, including ``allowed-tools`` grants.
        Only works with Anthropic models.

    ``https://`` / ``http://``
        Remote SKILL.md URL.  Fetched once at build time.  Only the **catalog
        entry** (name + description) is injected into the system prompt; the
        full body is served on demand via the ``activate_skill`` tool.

    Anything else
        Local skill name.  Discovered from standard directories (project paths
        take precedence over user-level paths):

        * ``<project>/.agents/skills/<name>/SKILL.md``
        * ``<project>/.claude/skills/<name>/SKILL.md``
        * ``~/.agents/skills/<name>/SKILL.md``
        * ``~/.claude/skills/<name>/SKILL.md``

    **How local and URL skills are surfaced to the model:**

    At agent build time, only a compact **skill catalog** (name + description,
    ~50–100 tokens per skill) is appended to the system prompt.  An
    ``activate_skill`` tool is also injected.  When the model decides a skill
    is relevant it calls ``activate_skill(name="…")`` to load the full
    ``SKILL.md`` body into context on demand — following the progressive
    disclosure pattern from the agentskills.io specification.
    """

    names: Union[List[str], str]  # list of entries or "*" for all local

    @classmethod
    def from_config(cls, value) -> "SkillsConfig":
        """Parse from a YAML value (list of entries or the string ``"*"``)."""
        if value == "*":
            return cls(names="*")
        if isinstance(value, list):
            return cls(names=[str(n) for n in value])
        if isinstance(value, str):
            return cls(names=[value])
        raise ValueError(
            f"'skills' must be a list of skill entries or '*', got: {value!r}"
        )

    # ------------------------------------------------------------------
    # Partition entries into local/url/id buckets
    # ------------------------------------------------------------------

    def _partition(self) -> Tuple[List[str], List[str], List[str]]:
        """Return ``(local_names, urls, skill_ids)`` for the configured entries."""
        if self.names == "*":
            return [], [], []  # handled separately in callers
        local, urls, ids = [], [], []
        for entry in self.names:
            if _is_skill_id(entry):
                ids.append(entry)
            elif _is_url(entry):
                urls.append(entry)
            else:
                local.append(entry)
        return local, urls, ids

    # ------------------------------------------------------------------
    # Resolve text skills (local + URL) → skill dicts
    # ------------------------------------------------------------------

    def resolve_text_skills(
        self, project_dir: Optional[Path] = None
    ) -> Dict[str, Dict]:
        """Discover and return ``{name: skill_dict}`` for all local and URL skills.

        Anthropic-hosted skill IDs are excluded; they are handled separately
        via :meth:`build_model_args_patch`.
        """
        if self.names == "*":
            return _discover_skills(project_dir)

        local_names, urls, _ids = self._partition()
        result: Dict[str, Dict] = {}

        all_local = _discover_skills(project_dir)
        for name in local_names:
            if name in all_local:
                result[name] = all_local[name]
            else:
                logger.warning(
                    "Skill '%s' not found in any search path "
                    "(.agents/skills/, .claude/skills/, ~/.agents/skills/, ~/.claude/skills/)",
                    name,
                )

        for url in urls:
            skill = _fetch_skill_from_url(url)
            if skill:
                result[skill["name"]] = skill

        return result

    # ------------------------------------------------------------------
    # Catalog block (tier-1 disclosure: name + description only)
    # ------------------------------------------------------------------

    @staticmethod
    def build_catalog_block(skills: Dict[str, Dict]) -> str:
        """Return the system-prompt block that discloses available skills.

        Injects only ``name`` and ``description`` per skill (~50–100 tokens
        each) plus a one-line instruction on how to activate them.
        Returns an empty string when *skills* is empty.
        """
        if not skills:
            return ""

        lines = [
            "",
            "When a task matches a skill's description, call the "
            "`activate_skill` tool with that skill's name to load its full "
            "instructions before proceeding.",
            "",
            "<available_skills>",
        ]
        for skill in skills.values():
            lines.append(f'  <skill name="{skill["name"]}">')
            lines.append(f"    <description>{skill['description']}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # API-level injection (Anthropic-hosted skill IDs)
    # ------------------------------------------------------------------

    def build_model_args_patch(self) -> Dict[str, Any]:
        """Return a dict to deep-merge into ``model_args`` for Anthropic skill IDs.

        If no skill IDs are configured returns an empty dict.
        Example result::

            {
                "skills": [{"id": "skill_01abc"}, {"id": "skill_02xyz"}],
                "extra_headers": {"anthropic-beta": "skills-2025-10-02"},
            }
        """
        if self.names == "*":
            return {}
        _, _, ids = self._partition()
        if not ids:
            return {}
        return {
            "skills": [{"id": sid} for sid in ids],
            "extra_headers": {"anthropic-beta": "skills-2025-10-02"},
        }


_KNOWN_AGENT_KEYS = frozenset(
    {
        "name",
        "model_name",
        "system_prompt",
        "system_prompt_path",
        "api_type",
        "model_args",
        "max_exception_retry",
        "max_interrupt_steps",
        "max_llm_recall",
        "extra_settings",
        "proxy",
        "context_manager",
        "skills",
        "tools",
    }
)


def _copy_prompt_with_update(prompt, update: Dict[str, Any]):
    copied = prompt.model_copy(update=update)
    for attr in ("_qualified_key", "_resource_namespace"):
        value = getattr(prompt, attr, None)
        if value is not None:
            setattr(copied, attr, value)
    return copied


def _parse_tool_refs(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, (list, tuple)):
        raise TypeError(
            "Agent config 'tools' must be a tool/proxy resource ref or a list of refs"
        )
    refs: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError(
                "Agent config 'tools' entries must be tool/proxy resource refs, "
                f"got {type(item).__name__}"
            )
        refs.append(item)
    return refs


@dataclass
class AgentSpec:
    """
    Parsed, validated description of one agent from config.

    Intermediate representation between raw YAML and live Agent instances.
    Config parsing fails here with clear errors; Agent construction is trivial.

    Config format (per-agent, after global merge)::

        name: analyzer
        model_name: gpt-4o
        system_prompt_path: analytica/analyzer_system   # OR
        system_prompt: "You are an analyst. ..."          # inline
        api_type: completion
        model_args:
            temperature: 0.1
            max_completion_tokens: 20000
        max_exception_retry: 3
        max_interrupt_steps: 5
        max_llm_recall: 0
        extra_settings: {}
        tools:
          - shared_pkg.tactics:code_review
          - shared_pkg.tools:search
          - shared_pkg.proxies:market_data
    """

    name: str
    model: str
    system_prompt_path: Optional[str] = None
    system_prompt: Any = None  # Prompt object or None
    api_type: str = "completion"  # stored as string, converted at build time
    model_args: Dict[str, Any] = field(default_factory=dict)
    max_exception_retry: int = 3
    max_interrupt_steps: int = 5
    max_llm_recall: int = 0
    extra_settings: Dict[str, Any] = field(default_factory=dict)
    proxy: Optional[ProxyConfig] = None
    context_manager: Optional[ContextManagerConfig] = None
    skills: Optional[SkillsConfig] = None
    tools: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, name: str, raw: Dict[str, Any]) -> "AgentSpec":
        """Parse a single agent config dict into an AgentSpec.

        *raw* is the per-agent dict **after** global defaults have been
        merged in.  Known keys are extracted; unknown keys are treated
        as additional model_args.
        """
        raw = raw.copy()

        # -- required: model -----------------------------------------------
        model = raw.pop("model_name", None)
        if model is None:
            raise ValueError(f"Agent '{name}' missing required 'model_name'")

        # -- required: system prompt (inline string or registry path) ------
        inline_prompt_str = raw.pop("system_prompt", None)
        system_prompt_path = raw.pop("system_prompt_path", None)
        if inline_prompt_str is None and system_prompt_path is None:
            raise ValueError(
                f"Agent '{name}' needs either 'system_prompt' or 'system_prompt_path'"
            )

        # Build a Prompt object from inline string if provided
        system_prompt = None
        if inline_prompt_str is not None:
            from .prompt import Prompt

            system_prompt = Prompt(
                path=f"_inline/{name}/system", prompt=inline_prompt_str
            )

        # -- optional typed fields -----------------------------------------
        api_type = raw.pop("api_type", "completion")
        max_exception_retry = raw.pop("max_exception_retry", 3)
        max_interrupt_steps = raw.pop("max_interrupt_steps", 5)
        max_llm_recall = raw.pop("max_llm_recall", 0)
        extra_settings = raw.pop("extra_settings", {})

        # -- proxy config (already deep-merged with global by the caller) --
        proxy_raw = raw.pop("proxy", None)
        proxy = ProxyConfig.from_dict(proxy_raw) if proxy_raw else None

        # -- context manager config ----------------------------------------
        cm_raw = raw.pop("context_manager", None)
        if isinstance(cm_raw, dict):
            # type: null in YAML arrives as None value inside the dict
            cm_type = cm_raw.get("type")
            context_manager_cfg = (
                None if cm_type is None else ContextManagerConfig.from_dict(cm_raw)
            )
        else:
            context_manager_cfg = None

        # -- skills config -------------------------------------------------
        skills_raw = raw.pop("skills", None)
        skills_cfg = (
            SkillsConfig.from_config(skills_raw) if skills_raw is not None else None
        )

        # -- tactic tools --------------------------------------------------
        tools = _parse_tool_refs(raw.pop("tools", None))

        # -- model_args: explicit dict + leftover unknown keys -------------
        model_args = raw.pop("model_args", {})
        raw.pop("name", None)
        if raw:
            logger.warning(
                "Agent '%s': unrecognised config keys %s will be passed as model_args. "
                "Known keys are: %s. Check for typos.",
                name,
                sorted(raw.keys()),
                sorted(_KNOWN_AGENT_KEYS),
            )
        model_args.update(raw)  # anything left is additional model_args

        return cls(
            name=name,
            model=model,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            api_type=api_type,
            model_args=model_args,
            max_exception_retry=max_exception_retry,
            max_interrupt_steps=max_interrupt_steps,
            max_llm_recall=max_llm_recall,
            extra_settings=extra_settings,
            proxy=proxy,
            context_manager=context_manager_cfg,
            skills=skills_cfg,
            tools=tools,
        )

    def build(self, runtime: Runtime, invoker):
        """Construct a live Agent from this spec."""
        from .agent import Agent
        from .const import APITypes

        if self.system_prompt is not None:
            prompt = self.system_prompt
        else:
            prompt = runtime.get_prompt(self.system_prompt_path)

        api_type = (
            self.api_type
            if isinstance(self.api_type, APITypes)
            else APITypes(self.api_type)
        )
        tool_function_refs = list(self.tools)
        tool_proxy_refs: List[str] = []
        if self.tools:
            from .tactic_tool import (
                namespace_from_qualified_key,
                partition_agent_tool_refs,
            )

            base_namespace = getattr(
                prompt, "_resource_namespace", None
            ) or namespace_from_qualified_key(getattr(prompt, "_qualified_key", None))
            partitioned_tools = partition_agent_tool_refs(
                self.tools,
                runtime=runtime,
                base_namespace=base_namespace,
            )
            tool_function_refs = partitioned_tools.function_refs
            tool_proxy_refs = partitioned_tools.proxy_refs

        # -- Proxy tool injection ------------------------------------------
        proxy_config = self.proxy
        if tool_proxy_refs:
            if proxy_config is None:
                proxy_config = ProxyConfig(activate_proxies=tool_proxy_refs)
            else:
                merged_activate_proxies: List[str] = []
                for ref in [*proxy_config.activate_proxies, *tool_proxy_refs]:
                    if ref not in merged_activate_proxies:
                        merged_activate_proxies.append(ref)
                proxy_config = ProxyConfig(
                    activate_proxies=merged_activate_proxies,
                    deploy_mode=proxy_config.deploy_mode,
                    cutoff_date=proxy_config.cutoff_date,
                    exec_env=proxy_config.exec_env,
                    max_output_chars=proxy_config.max_output_chars,
                    truncation_indicator=proxy_config.truncation_indicator,
                    timeout=proxy_config.timeout,
                    prompt_template=proxy_config.prompt_template,
                )

        if proxy_config is not None:
            from ..proxies.base import ProxyManager
            from ..proxies.interpreter import AgentInterpreter
            from ..proxies.prompt_template import render_proxy_prompt
            from ..proxies.proxy_tools import (
                make_query_api_doc_tool,
                make_run_python_tool,
            )

            cutoff = (
                dt.datetime.fromisoformat(proxy_config.cutoff_date)
                if proxy_config.cutoff_date
                else None
            )
            proxy_manager = ProxyManager(
                activate_proxies=proxy_config.activate_proxies,
                cutoff_date=cutoff,
                deploy_mode=proxy_config.deploy_mode,
                runtime=runtime,
            )
            interpreter = AgentInterpreter(
                proxy_manager,
                max_output_chars=proxy_config.max_output_chars,
                truncation_indicator=proxy_config.truncation_indicator,
                timeout=proxy_config.timeout,
            )

            query_doc_tool = make_query_api_doc_tool(proxy_manager)

            # Only interpreter mode injects run_python.
            # Other modes (jupyter, None, future sandboxes) leave execution to
            # the tactic — the agent writes cell tags or uses another mechanism.
            extra_tools = [query_doc_tool]
            if proxy_config.exec_env == "interpreter":
                extra_tools.append(make_run_python_tool(interpreter))

            proxy_block = render_proxy_prompt(
                api_directory=proxy_manager.retrieve_api_docs(),
                max_output_chars=proxy_config.max_output_chars,
                truncation_indicator=proxy_config.truncation_indicator,
                exec_env=proxy_config.exec_env,
                custom_template=proxy_config.prompt_template,
            )

            # Create a modified prompt without mutating the original.
            prompt = _copy_prompt_with_update(
                prompt,
                {
                    "prompt": prompt.prompt + proxy_block,
                    "function_list": list(prompt.function_list) + extra_tools,
                },
            )

        # -- Direct tool refs ----------------------------------------------
        if tool_function_refs:
            prompt = _copy_prompt_with_update(
                prompt,
                {
                    "function_list": list(prompt.function_list) + tool_function_refs,
                },
            )

        # -- Skills injection ----------------------------------------------
        model_args = dict(self.model_args)
        if self.skills is not None:
            # Tier-1: inject catalog (name + description) + activate_skill tool
            text_skills = self.skills.resolve_text_skills()
            if text_skills:
                catalog_block = SkillsConfig.build_catalog_block(text_skills)
                activate_tool = make_activate_skill_tool(text_skills)
                prompt = _copy_prompt_with_update(
                    prompt,
                    {
                        "prompt": prompt.prompt + catalog_block,
                        "function_list": list(prompt.function_list) + [activate_tool],
                    },
                )
            # Anthropic-hosted skill IDs → merge into model_args for the API call
            patch = self.skills.build_model_args_patch()
            if patch:
                model_args = _deep_merge(model_args, patch)

        # -- Context manager -----------------------------------------------
        context_manager = (
            self.context_manager.build(self.model, runtime)
            if self.context_manager is not None
            else None
        )

        return Agent(
            name=self.name,
            system_prompt=prompt,
            model=self.model,
            llm_invoker=invoker,
            runtime=runtime,
            api_type=api_type,
            model_args=model_args,
            max_exception_retry=self.max_exception_retry,
            max_interrupt_steps=self.max_interrupt_steps,
            max_llm_recall=self.max_llm_recall,
            context_manager=context_manager,
        )


def parse_agent_configs(
    config: Dict[str, Any],
    agent_group: List[str],
    tactic_name: str,
) -> Dict[str, "AgentSpec"]:
    """Parse ``global`` + ``agent_configs`` from a tactic config dict.

    Returns ``{agent_name: AgentSpec}`` for each name in *agent_group*.
    """
    global_cfg = config.get("global", {})
    raw_list = config.get("agent_configs", [])

    agent_by_name: Dict[str, Dict] = {}
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise TypeError(
                f"agent_configs entries must be dicts, got {type(entry).__name__}"
            )
        name = entry.get("name")
        if name is None:
            raise ValueError(f"Agent config entry missing 'name': {entry}")
        agent_by_name[name] = _deep_merge(global_cfg, entry)

    specs: Dict[str, AgentSpec] = {}
    for agent_name in agent_group:
        if agent_name not in agent_by_name:
            raise ValueError(
                f"Agent '{agent_name}' required by tactic '{tactic_name}' "
                f"not found in agent_configs. Available: {sorted(agent_by_name)}"
            )
        specs[agent_name] = AgentSpec.from_config(agent_name, agent_by_name[agent_name])

    return specs
