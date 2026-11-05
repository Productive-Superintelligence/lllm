"""Load shell strategies from Python files, directories, or entrypoints."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .packages import strategy_from_package
from .strategy import Strategy


def load_strategy(target: str | Path) -> Strategy:
    """Load a Strategy from `module:attr`, a `.py` file, or a directory."""

    text = _target_text(target)
    path = Path(text).expanduser()
    if path.is_dir():
        for name in ("strategy.py", "shell.py"):
            candidate = path / name
            if candidate.is_file():
                return _load_file(candidate)
        if (path / "lllm.toml").is_file():
            return strategy_from_package(path)
        raise ValueError("strategy directory must contain strategy.py or shell.py")
    if path.is_file():
        if path.name == "lllm.toml":
            return strategy_from_package(path)
        return _load_file(path)
    if ":" in text:
        module_name, _, attr = text.partition(":")
        with _import_path(Path.cwd()):
            module = importlib.import_module(module_name)
        value: Any = module
        for part in attr.split("."):
            value = getattr(value, part)
        return _coerce_strategy(value)
    raise ValueError("strategy target must be a directory, Python file, or module:attr")


def _load_file(path: Path) -> Strategy:
    module_name = f"_lllm_shell_strategy_{abs(hash(path.resolve()))}"
    with _import_path(path.parent):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import strategy file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    if hasattr(module, "build_strategy"):
        return _coerce_strategy(module.build_strategy)
    if hasattr(module, "strategy"):
        return _coerce_strategy(module.strategy)
    raise ValueError("strategy file must define build_strategy() or strategy")


def _coerce_strategy(value: Any) -> Strategy:
    if isinstance(value, Strategy):
        return value
    if callable(value) and _can_call_without_args(value):
        produced = value()
        if isinstance(produced, Strategy):
            return produced
    raise TypeError("strategy entrypoint must be a Strategy or no-argument factory")


def _can_call_without_args(value: Any) -> bool:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.default is inspect.Parameter.empty and parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return False
    return True


def _target_text(target: str | Path) -> str:
    try:
        text = str(target)
    except TypeError as exc:
        raise ValueError("strategy target must be a non-empty string or path") from exc
    if not text or text != text.strip():
        raise ValueError("strategy target must be a non-empty string or path")
    return text


@contextmanager
def _import_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass
        importlib.invalidate_caches()
