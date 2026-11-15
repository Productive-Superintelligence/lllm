"""Public validation helpers for values crossing LLLM boundaries.

Packages built on LLLM (such as ``sssn`` and ``aaax``) import these instead of
the private ``lllm.protocol._validation`` module.
"""

from ._validation import copy_boundary_value

__all__ = ["copy_boundary_value"]
