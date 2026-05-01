"""chessvec - reference and vectorized chess rules engines.

The vectorized engine requires torch and is imported lazily; the reference
engine has no heavy dependencies.
"""

from . import action_encoding, reference, types

__all__ = ["action_encoding", "reference", "types", "vectorized"]


def __getattr__(name: str):
    if name == "vectorized":
        from . import vectorized

        return vectorized
    raise AttributeError(f"module 'chessvec' has no attribute {name!r}")
