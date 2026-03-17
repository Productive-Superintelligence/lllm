# lllm/core/runtime.py
"""
Runtime — the central registry for an LLLM session.

All resources live in a unified ``ResourceNode``-based store keyed by
qualified URLs (``"package.section:resource_path"``).  Resolution is
namespace-aware: bare keys are resolved via the default namespace.

Named runtimes are supported for parallel experiments.
"""
from __future__ import annotations

import difflib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TYPE_CHECKING

from lllm.core.resource import ResourceNode, PackageInfo

if TYPE_CHECKING:
    from lllm.core.prompt import Prompt
    from lllm.proxies.base import BaseProxy

logger = logging.getLogger(__name__)


class Runtime:
    """
    Holds all registries and shared state for an LLLM runtime.

    Internally every resource is a :class:`ResourceNode` stored in
    ``_resources`` keyed by its qualified key.  Typed convenience
    methods (``register_prompt``, ``get_prompt``, etc.) are thin wrappers.
    """

    def __init__(self) -> None:
        self._resources: Dict[str, ResourceNode] = {}
        self._type_index: Dict[str, Set[str]] = {}
        self.packages: Dict[str, PackageInfo] = {}
        self._loaded_package_paths: Set[str] = set()
        self._default_namespace: Optional[str] = None
        self._discovery_done: bool = False

    # ==================================================================
    # Unified registry
    # ==================================================================

    def register(self, node: ResourceNode, overwrite: bool = True) -> None:
        """Register a ``ResourceNode``.

        Stored under ``node.qualified_key``.  Warns on collision if
        overwriting.
        """
        qk = node.qualified_key

        if qk in self._resources:
            if not overwrite:
                raise ValueError(f"Resource '{qk}' already registered")
            existing = self._resources[qk]
            if existing is not node:
                logger.debug("Resource '%s' overwritten (type=%s)", qk, node.resource_type)

        self._resources[qk] = node

        rtype = node.resource_type
        if rtype not in self._type_index:
            self._type_index[rtype] = set()
        self._type_index[rtype].add(qk)

    def get(self, key: str, resource_type: Optional[str] = None) -> Any:
        """Retrieve a resource value by key.

        Resolution:
            1. Exact match on *key*.
            2. If *key* has no ``:``, try ``default_ns.<type>s:key`` for
               the given *resource_type*, or scan all built-in sections.
        """
        node = self._resolve(key, resource_type)
        if resource_type and node.resource_type != resource_type:
            raise TypeError(
                f"Resource '{key}' is type '{node.resource_type}', expected '{resource_type}'"
            )
        return node.value

    def get_node(self, key: str, resource_type: Optional[str] = None) -> ResourceNode:
        """Like :meth:`get` but returns the node itself (not ``.value``)."""
        return self._resolve(key, resource_type)

    def _resolve(self, key: str, resource_type: Optional[str] = None) -> ResourceNode:
        # 1. Exact match
        if key in self._resources:
            return self._resources[key]

        # 2. Default namespace fallback (bare key, no ":")
        if self._default_namespace and ":" not in key:
            if resource_type:
                ns_key = f"{self._default_namespace}.{resource_type}s:{key}"
                if ns_key in self._resources:
                    return self._resources[ns_key]
            for section in ("prompts", "proxies", "tactics", "configs"):
                ns_key = f"{self._default_namespace}.{section}:{key}"
                if ns_key in self._resources:
                    return self._resources[ns_key]

        # 3. If "pkg:path" with no dot, inject section for typed lookups
        if ":" in key and resource_type:
            pkg_part, resource_part = key.split(":", 1)
            if "." not in pkg_part:
                full_key = f"{pkg_part}.{resource_type}s:{resource_part}"
                if full_key in self._resources:
                    return self._resources[full_key]

        all_keys = sorted(self._resources.keys())
        close = difflib.get_close_matches(key, all_keys, n=5, cutoff=0.4)
        hint = f" Did you mean: {close}?" if close else f" Registered ({len(all_keys)}): {all_keys[:20]}"
        raise KeyError(f"Resource '{key}' not found.{hint}")

    def has(self, key: str) -> bool:
        try:
            self._resolve(key)
            return True
        except KeyError:
            return False

    def keys(self, resource_type: Optional[str] = None) -> List[str]:
        if resource_type:
            return sorted(self._type_index.get(resource_type, set()))
        return sorted(self._resources.keys())

    # ==================================================================
    # Package management
    # ==================================================================

    def register_package(self, pkg: PackageInfo) -> None:
        eff = pkg.effective_name
        if eff in self.packages:
            existing = self.packages[eff]
            if existing.base_dir == pkg.base_dir:
                return
            logger.warning(
                "Package '%s' already registered (from %s), overwriting with %s",
                eff, existing.base_dir, pkg.base_dir,
            )
        self.packages[eff] = pkg

    # ==================================================================
    # Typed convenience — Prompts
    # ==================================================================

    def register_prompt(self, prompt: "Prompt", overwrite: bool = True,
                        namespace: str = "") -> None:
        node = ResourceNode.eager(prompt.path, prompt,
                                  namespace=namespace, resource_type="prompt")
        self.register(node, overwrite=overwrite)
        prompt._qualified_key = node.qualified_key  # type: ignore[attr-defined]

    def get_prompt(self, path: str) -> "Prompt":
        return self.get(path, resource_type="prompt")

    # ==================================================================
    # Typed convenience — Proxies
    # ==================================================================

    def register_proxy(self, name: str, proxy_cls: Type["BaseProxy"],
                       overwrite: bool = False, namespace: str = "") -> None:
        node = ResourceNode.eager(name, proxy_cls,
                                  namespace=namespace, resource_type="proxy")
        self.register(node, overwrite=overwrite)

    def get_proxy(self, path: str) -> Type["BaseProxy"]:
        return self.get(path, resource_type="proxy")

    # ==================================================================
    # Typed convenience — Tactics
    # ==================================================================

    def register_tactic(self, tactic_type: str, tactic_cls: Type,
                        overwrite: bool = False, namespace: str = "") -> None:
        node = ResourceNode.eager(tactic_type, tactic_cls,
                                  namespace=namespace, resource_type="tactic")
        self.register(node, overwrite=overwrite)

    def get_tactic(self, name: str) -> Type:
        return self.get(name, resource_type="tactic")

    # ==================================================================
    # Typed convenience — Configs
    # ==================================================================

    def register_config(self, name: str, config_data: Any = None,
                        overwrite: bool = True, namespace: str = "",
                        loader: Any = None) -> None:
        if loader is not None:
            node = ResourceNode.lazy(name, loader,
                                     namespace=namespace, resource_type="config")
        else:
            node = ResourceNode.eager(name, config_data,
                                      namespace=namespace, resource_type="config")
        self.register(node, overwrite=overwrite)

    def get_config(self, path: str) -> Any:
        return self.get(path, resource_type="config")

    # ==================================================================
    # Typed convenience — Context Managers
    # ==================================================================

    def register_context_manager(self, cm_cls: Type,
                                  overwrite: bool = True, namespace: str = "") -> None:
        """Register a :class:`~lllm.core.dialog.ContextManager` subclass.

        The registry key is taken from ``cm_cls.name``, which every concrete
        subclass must define.  The built-in ``"default"`` type
        (:class:`~lllm.core.dialog.DefaultContextManager`) is resolved
        automatically by the config layer and does **not** need to be registered.
        Use this method to register *custom* implementations::

            class SummaryCompressor(ContextManager):
                name = "summary"
                ...

            runtime.register_context_manager(SummaryCompressor)

        Then in config::

            context_manager:
              type: summary
        """
        cm_name = getattr(cm_cls, "name", None)
        if not cm_name:
            raise ValueError(
                f"Cannot register {cm_cls.__name__}: it must define a 'name' class attribute."
            )
        node = ResourceNode.eager(cm_name, cm_cls,
                                  namespace=namespace, resource_type="context_manager")
        self.register(node, overwrite=overwrite)

    def get_context_manager(self, name: str) -> Type:
        """Return a registered context-manager class by *name*."""
        return self.get(name, resource_type="context_manager")

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def reset(self) -> None:
        self._resources.clear()
        self._type_index.clear()
        self.packages.clear()
        self._loaded_package_paths.clear()
        self._default_namespace = None
        self._discovery_done = False


# ==================================================================
# Module-level singletons and named runtime registry
# ==================================================================

_default_runtime = Runtime()
_runtimes: Dict[str, Runtime] = {}


def get_default_runtime() -> Runtime:
    return _default_runtime


def set_default_runtime(rt: Runtime) -> None:
    global _default_runtime
    _default_runtime = rt


def get_runtime(name: Optional[str] = None) -> Runtime:
    if name is None:
        return _default_runtime
    if name not in _runtimes:
        raise KeyError(
            f"Runtime '{name}' not found. Available: {sorted(_runtimes)}. "
            f"Call load_runtime(path, name='{name}') first."
        )
    return _runtimes[name]


_SHARED_PACKAGES_DIR = "lllm_packages"


def _load_shared_packages(rt: "Runtime") -> None:
    """Auto-load packages found in standard ``lllm_packages/`` directories.

    Scans two locations (project-level first, then user-level) and loads
    every sub-folder that contains an ``lllm.toml``.  This is the drop-in
    sharing mechanism — analogous to ``.agents/skills/`` for agent skills.

    Scanned locations (in order):
      1. ``<project_root>/lllm_packages/``  — project-level shared packages
      2. ``~/.lllm/packages/``              — user-level shared packages

    A ``<project_root>`` is the directory that contains the project's
    ``lllm.toml``.  If no ``lllm.toml`` is found, ``cwd`` is used instead.
    """
    from pathlib import Path
    from lllm.core.config import load_package, find_config_file

    project_toml = find_config_file()
    project_root = project_toml.parent if project_toml else Path.cwd()

    search_dirs = [
        project_root / _SHARED_PACKAGES_DIR,
        Path.home() / ".lllm" / "packages",
    ]

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for pkg_dir in sorted(search_dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            toml = pkg_dir / "lllm.toml"
            if not toml.is_file():
                continue
            try:
                load_package(str(toml), runtime=rt)
                logger.debug("Auto-loaded shared package from %s", pkg_dir)
            except Exception as exc:
                warnings.warn(
                    f"Failed to auto-load shared package at {pkg_dir}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def load_runtime(
    toml_path: Optional[str] = None,
    name: Optional[str] = None,
) -> Runtime:
    """Create and populate a Runtime from a TOML file.

    *name=None* → replaces the default runtime.
    *name="something"* → stored as a named runtime.
    """
    from lllm.core.config import load_package, find_config_file

    rt = Runtime()
    if toml_path is not None:
        load_package(str(toml_path), runtime=rt)
    else:
        found = find_config_file()
        if found:
            load_package(str(found), runtime=rt)
        else:
            from lllm.core.config import load_cwd_fallback
            if load_cwd_fallback(rt):
                warnings.warn(
                    "No lllm.toml found — auto-discovered resource folders in the current "
                    "directory. Add an lllm.toml to remove this warning and enable "
                    "namespacing, dependencies, and full package features.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            else:
                logger.debug("No lllm.toml found. Running with empty runtime (fast mode).")

    # Always scan lllm_packages/ directories regardless of how the main package
    # was loaded — shared packages layer on top of the project package.
    _load_shared_packages(rt)

    rt._discovery_done = True

    if name is None:
        set_default_runtime(rt)
    else:
        _runtimes[name] = rt
    return rt


# ---------------------------------------------------------------------------
# Package install / export
# ---------------------------------------------------------------------------

def install_package(
    zip_path: "str | Path",
    *,
    alias: Optional[str] = None,
    scope: str = "user",
    load: bool = True,
    runtime: Optional[Runtime] = None,
) -> "Path":
    """Install an LLLM package from a zip file.

    Extracts *zip_path*, reads the package name from the top-level
    ``lllm.toml``, and copies the package to the appropriate shared-packages
    directory.

    Parameters
    ----------
    zip_path:
        Path to the ``.zip`` file produced by :func:`export_package` or any
        zip archive with a top-level folder containing an ``lllm.toml``.
    alias:
        Install the package under this name instead of the name declared in
        its ``lllm.toml``.  The folder is renamed and the ``name`` field in
        ``lllm.toml`` is updated so the namespace prefix changes too.  Use
        this when the package's original name collides with something already
        installed.
    scope:
        ``"user"`` (default) — installs to ``~/.lllm/packages/``, available
        across all projects for the current user.
        ``"project"`` — installs to ``<project_root>/lllm_packages/``,
        committed with the repo and shared with the team.
    load:
        If ``True`` (default) immediately loads the installed package into
        *runtime* so it is available without restarting Python.
    runtime:
        Runtime to load into.  Defaults to the current default runtime.

    Returns
    -------
    Path
        The installed package directory.

    Raises
    ------
    FileNotFoundError
        If *zip_path* does not exist.
    ValueError
        If the zip contains no recognisable package or the zip structure is
        ambiguous.
    FileExistsError
        If a package with the same name is already installed.  Pass
        ``alias=<new_name>`` to install under a different namespace.

    Examples
    --------
    Basic install::

        from lllm import install_package
        install_package("finance-toolkit-v1.2.zip")
        # Resources now accessible as finance-toolkit:...

    Install with alias to avoid a name collision::

        install_package("finance-toolkit-v1.2.zip", alias="ft")
        # Resources accessible as ft:...

    Install into the project (committed to repo)::

        install_package("finance-toolkit-v1.2.zip", scope="project")
    """
    import re
    import shutil
    import tempfile
    import tomllib
    import zipfile as _zipfile

    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Determine install root
    if scope == "user":
        install_root = Path.home() / ".lllm" / "packages"
    elif scope == "project":
        from lllm.core.config import find_config_file
        project_toml = find_config_file()
        project_root = project_toml.parent if project_toml else Path.cwd()
        install_root = project_root / _SHARED_PACKAGES_DIR
    else:
        raise ValueError(f"scope must be 'user' or 'project', got {scope!r}")

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        with _zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # Locate the package root inside the extracted tree: the first
        # directory that contains an lllm.toml (at most one level deep).
        pkg_dir: Optional[Path] = None
        if (tmp / "lllm.toml").is_file():
            pkg_dir = tmp
        else:
            candidates = sorted(
                p for p in tmp.iterdir()
                if p.is_dir() and (p / "lllm.toml").is_file()
            )
            if len(candidates) == 1:
                pkg_dir = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(
                    f"Multiple package directories found in {zip_path.name}: "
                    f"{[c.name for c in candidates]}. "
                    "The zip must contain exactly one top-level package folder."
                )

        if pkg_dir is None:
            raise ValueError(
                f"No lllm.toml found in {zip_path.name}. "
                "Package zips must contain a top-level folder with an lllm.toml."
            )

        # Read the declared package name
        with (pkg_dir / "lllm.toml").open("rb") as fh:
            toml_data = tomllib.load(fh)
        original_name: str = toml_data.get("package", {}).get("name", "")
        if not original_name:
            raise ValueError(
                f"lllm.toml in {zip_path.name} has no [package] name field."
            )

        final_name = alias if alias else original_name

        # Validate alias format (same rules as package names)
        if alias and not re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$|^[a-z0-9]$", alias):
            raise ValueError(
                f"alias '{alias}' is not a valid package name. "
                "Use lowercase letters, numbers, and hyphens only."
            )

        # Check for collision
        install_root.mkdir(parents=True, exist_ok=True)
        dest = install_root / final_name
        if dest.exists():
            if alias:
                raise FileExistsError(
                    f"A package named '{final_name}' is already installed at {dest}. "
                    "Choose a different alias."
                )
            else:
                raise FileExistsError(
                    f"Package '{original_name}' is already installed at {dest}. "
                    f"Use alias=<new_name> to install under a different namespace, "
                    "or remove the existing package directory first."
                )

        # If alias differs from the original name, patch lllm.toml in-place
        # before copying so the namespace prefix matches what the user expects.
        if alias and alias != original_name:
            toml_text = (pkg_dir / "lllm.toml").read_text(encoding="utf-8")
            # Replace: name = "original_name"  →  name = "alias"
            # Handles both single- and double-quoted values.
            toml_text = re.sub(
                r'^(name\s*=\s*)["\']' + re.escape(original_name) + r'["\']',
                f'\\1"{alias}"',
                toml_text,
                flags=re.MULTILINE,
            )
            (pkg_dir / "lllm.toml").write_text(toml_text, encoding="utf-8")

        shutil.copytree(str(pkg_dir), str(dest))
        logger.info("Installed package '%s' to %s", final_name, dest)

    if load:
        from lllm.core.config import load_package
        rt = runtime or get_default_runtime()
        load_package(str(dest / "lllm.toml"), runtime=rt)
        logger.info("Loaded package '%s' into runtime", final_name)

    return dest


def export_package(
    package_name: str,
    output_path: "str | Path",
    *,
    bundle_deps: bool = False,
    runtime: Optional[Runtime] = None,
) -> "Path":
    """Export an installed package to a zip file ready for sharing.

    The zip follows the standard structure expected by :func:`install_package`:
    a single top-level folder (named after the package) containing the full
    package directory tree including ``lllm.toml``.

    Parameters
    ----------
    package_name:
        The package name as registered in the runtime (the ``name`` field
        from its ``lllm.toml``).
    output_path:
        Destination file path.  A ``.zip`` extension is added automatically
        if omitted.
    bundle_deps:
        If ``True``, resolve all transitive dependencies declared in the
        package's ``lllm.toml`` ``[dependencies]`` section and bundle them
        inside a ``lllm_packages/`` sub-directory in the zip.  Recipients can
        install the package directly without installing each dependency
        separately.  Defaults to ``False``.
    runtime:
        Runtime to look up the package in.  Defaults to the current default
        runtime.

    Returns
    -------
    Path
        The created zip file.

    Raises
    ------
    ValueError
        If *package_name* is not found in the runtime.
    FileNotFoundError
        If the package's on-disk directory no longer exists.

    Examples
    --------
    ::

        from lllm import export_package
        export_package("acme-finance", "~/releases/acme-finance-v1.2.zip")
        # With bundled deps:
        export_package("acme-finance", "~/releases/acme-finance-v1.2.zip", bundle_deps=True)
    """
    import zipfile as _zipfile

    output_path = Path(output_path).expanduser()
    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")

    rt = runtime or get_default_runtime()
    pkg_info = rt.packages.get(package_name)
    if pkg_info is None:
        available = sorted(rt.packages)
        raise ValueError(
            f"Package '{package_name}' not found in runtime. "
            f"Available packages: {available}"
        )

    pkg_dir = Path(pkg_info.base_dir)
    if not pkg_dir.is_dir():
        raise FileNotFoundError(
            f"Package directory for '{package_name}' not found at {pkg_dir}."
        )

    # Resolve transitive dependencies if requested
    dep_dirs: Dict[str, Path] = {}
    if bundle_deps:
        dep_dirs = _collect_package_deps(package_name, rt)

    def _should_skip(rel_parts: tuple) -> bool:
        return any(
            part.startswith(".") or part in ("__pycache__", ".git")
            for part in rel_parts
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _zipfile.ZipFile(output_path, "w", _zipfile.ZIP_DEFLATED) as zf:
        # Write the main package files
        for file in sorted(pkg_dir.rglob("*")):
            if not file.is_file():
                continue
            rel = file.relative_to(pkg_dir)
            if _should_skip(rel.parts):
                continue
            arcname = Path(package_name) / rel
            zf.write(file, arcname)

        # Bundle dependency packages under <pkg>/lllm_packages/<dep>/
        for dep_name, dep_dir in dep_dirs.items():
            for file in sorted(dep_dir.rglob("*")):
                if not file.is_file():
                    continue
                rel = file.relative_to(dep_dir)
                if _should_skip(rel.parts):
                    continue
                arcname = Path(package_name) / _SHARED_PACKAGES_DIR / dep_name / rel
                zf.write(file, arcname)

    logger.info("Exported package '%s' to %s", package_name, output_path)
    return output_path


def _collect_package_deps(
    package_name: str,
    rt: Runtime,
    _seen: Optional[Set[str]] = None,
) -> Dict[str, Path]:
    """Recursively collect all transitive dependency directories.

    Returns a mapping of ``{dep_name: dep_dir_path}`` for every dependency
    of *package_name* (direct and transitive), excluding the package itself.
    """
    import tomllib

    if _seen is None:
        _seen = set()
    _seen.add(package_name)

    pkg_info = rt.packages.get(package_name)
    if pkg_info is None:
        logger.warning("Dependency '%s' not found in runtime — skipping.", package_name)
        return {}

    pkg_dir = Path(pkg_info.base_dir)
    toml_file = pkg_dir / "lllm.toml"
    if not toml_file.is_file():
        return {}

    try:
        with toml_file.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        logger.warning("Could not read lllm.toml for '%s': %s", package_name, exc)
        return {}

    # [dependencies] packages = ["../some/path", "../other as alias"]
    # Entries are path strings (possibly with "as alias" suffix), relative to pkg_dir.
    raw_deps: list = data.get("dependencies", {}).get("packages", [])

    # Build a reverse map: base_dir (resolved str) → (name, Path) for fast lookup
    dir_to_pkg: Dict[str, tuple] = {
        str(Path(info.base_dir).resolve()): (eff_name, Path(info.base_dir))
        for eff_name, info in rt.packages.items()
    }

    collected: Dict[str, Path] = {}
    for raw in raw_deps:
        # Strip optional " as alias" suffix (handled by the dependency loader)
        path_part = str(raw).split(" as ")[0].strip()
        dep_abs = str((pkg_dir / path_part).resolve())
        match = dir_to_pkg.get(dep_abs)
        if match is None:
            logger.warning(
                "Dependency path '%s' (from '%s') not found in runtime — skipping.",
                path_part, package_name,
            )
            continue
        dep_name, dep_dir = match
        if dep_name in _seen:
            continue
        if not dep_dir.is_dir():
            logger.warning("Dependency '%s' directory missing at %s — skipping.", dep_name, dep_dir)
            continue
        collected[dep_name] = dep_dir
        # Recurse into transitive deps
        transitive = _collect_package_deps(dep_name, rt, _seen)
        for trans_name, trans_dir in transitive.items():
            if trans_name not in collected:
                collected[trans_name] = trans_dir

    return collected


# ---------------------------------------------------------------------------
# list_packages / remove_package
# ---------------------------------------------------------------------------

def list_packages(
    scope: Optional[str] = None,
    *,
    runtime: Optional[Runtime] = None,
) -> List[Dict[str, str]]:
    """Return metadata for all installed packages.

    Parameters
    ----------
    scope:
        ``"user"``    — only packages from ``~/.lllm/packages/``.
        ``"project"`` — only packages from ``<project_root>/lllm_packages/``.
        ``None``      — both scopes (default).

    Returns
    -------
    list of dict
        Each dict has keys: ``name``, ``version``, ``description``,
        ``scope`` (``"user"`` | ``"project"`` | ``"unknown"``), ``path``.

    Examples
    --------
    ::

        from lllm import list_packages
        for pkg in list_packages():
            print(pkg["name"], pkg["version"], pkg["scope"])
    """
    import tomllib

    from lllm.core.config import find_config_file

    project_toml = find_config_file()
    project_root = project_toml.parent if project_toml else Path.cwd()

    scope_dirs: List[tuple[str, Path]] = []
    if scope in (None, "project"):
        scope_dirs.append(("project", project_root / _SHARED_PACKAGES_DIR))
    if scope in (None, "user"):
        scope_dirs.append(("user", Path.home() / ".lllm" / "packages"))

    results: List[Dict[str, str]] = []
    for scope_name, search_dir in scope_dirs:
        if not search_dir.is_dir():
            continue
        for pkg_dir in sorted(search_dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            toml_file = pkg_dir / "lllm.toml"
            if not toml_file.is_file():
                continue
            try:
                with toml_file.open("rb") as fh:
                    data = tomllib.load(fh)
                pkg_section = data.get("package", {})
                results.append({
                    "name": pkg_section.get("name", pkg_dir.name),
                    "version": pkg_section.get("version", ""),
                    "description": pkg_section.get("description", ""),
                    "scope": scope_name,
                    "path": str(pkg_dir),
                })
            except Exception as exc:
                logger.warning("Could not read package at %s: %s", pkg_dir, exc)

    # Also include packages in the runtime that were loaded from elsewhere
    # (e.g. the main project package itself) — deduplicated by name.
    rt = runtime or get_default_runtime()
    seen_paths = {r["path"] for r in results}
    for pkg_info in rt.packages.values():
        pkg_dir_str = str(Path(pkg_info.base_dir).resolve())
        if pkg_dir_str not in seen_paths:
            results.append({
                "name": pkg_info.effective_name,
                "version": pkg_info.version,
                "description": pkg_info.description,
                "scope": "unknown",
                "path": pkg_info.base_dir,
            })

    return results


def remove_package(
    name: str,
    *,
    scope: Optional[str] = None,
    runtime: Optional[Runtime] = None,
) -> Path:
    """Remove an installed package by name.

    The package folder is deleted from the install directory.  The package
    is also unregistered from *runtime* if it is currently loaded there.

    Parameters
    ----------
    name:
        Package name as it appears in its ``lllm.toml`` (or the installed
        folder name if the two differ due to aliasing).
    scope:
        ``"user"``    — remove from ``~/.lllm/packages/``.
        ``"project"`` — remove from ``<project_root>/lllm_packages/``.
        ``None``      — search both locations; raises if found in multiple.
    runtime:
        Runtime to unregister from.  Defaults to the current default runtime.

    Returns
    -------
    Path
        The deleted package directory.

    Raises
    ------
    FileNotFoundError
        If the package is not found in any searched location.
    ValueError
        If *scope* is None and the package is found in both locations.

    Examples
    --------
    ::

        from lllm import remove_package
        remove_package("finance-toolkit")
        remove_package("acme-pack", scope="project")
    """
    import shutil
    import tomllib

    from lllm.core.config import find_config_file

    project_toml = find_config_file()
    project_root = project_toml.parent if project_toml else Path.cwd()

    scope_dirs: List[tuple[str, Path]] = []
    if scope in (None, "project"):
        scope_dirs.append(("project", project_root / _SHARED_PACKAGES_DIR))
    if scope in (None, "user"):
        scope_dirs.append(("user", Path.home() / ".lllm" / "packages"))

    def _matches(pkg_dir: Path, target: str) -> bool:
        """True if the package in *pkg_dir* is named *target*."""
        if pkg_dir.name == target:
            return True
        toml_file = pkg_dir / "lllm.toml"
        if toml_file.is_file():
            try:
                with toml_file.open("rb") as fh:
                    data = tomllib.load(fh)
                return data.get("package", {}).get("name", "") == target
            except Exception:
                pass
        return False

    matches: List[tuple[str, Path]] = []
    for scope_name, search_dir in scope_dirs:
        if not search_dir.is_dir():
            continue
        for pkg_dir in search_dir.iterdir():
            if pkg_dir.is_dir() and _matches(pkg_dir, name):
                matches.append((scope_name, pkg_dir))

    if not matches:
        raise FileNotFoundError(
            f"Package '{name}' not found in any installed location. "
            f"Run list_packages() to see what is installed."
        )
    if len(matches) > 1:
        locations = [str(m[1]) for m in matches]
        raise ValueError(
            f"Package '{name}' found in multiple locations: {locations}. "
            "Specify scope='user' or scope='project' to select one."
        )

    _, pkg_dir = matches[0]
    shutil.rmtree(str(pkg_dir))
    logger.info("Removed package '%s' from %s", name, pkg_dir)

    # Unregister from runtime if loaded
    rt = runtime or get_default_runtime()
    if name in rt.packages:
        del rt.packages[name]
        # Remove all resources from this package (by base_dir prefix)
        pkg_dir_str = str(pkg_dir)
        keys_to_remove = [
            k for k, node in rt._resources.items()
            if getattr(node, "source_path", None) and
            str(node.source_path).startswith(pkg_dir_str)
        ]
        for k in keys_to_remove:
            del rt._resources[k]
            for rtype_set in rt._type_index.values():
                rtype_set.discard(k)

    return pkg_dir