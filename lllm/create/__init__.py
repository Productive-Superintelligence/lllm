"""Project scaffolding: native project templates and service scaffolds."""

from .scaffold import DEFAULT_TEMPLATE, SERVICE_SCAFFOLDS, ScaffoldResult, create_project
from .templates import list_builtin_template_names

__all__ = [
    "DEFAULT_TEMPLATE",
    "SERVICE_SCAFFOLDS",
    "ScaffoldResult",
    "create_project",
    "list_builtin_template_names",
]
