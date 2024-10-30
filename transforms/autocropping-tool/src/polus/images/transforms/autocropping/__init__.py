"""autocropping."""

__version__ = "2.0.0"

from . import entropy
from .autocropping import autocropping

__all__ = [
    "autocropping",
    "entropy",
]
