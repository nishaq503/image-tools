"""The Stitching Vector to MicroJSON plugin."""

__version__ = "0.1.0"

from .convert import output_path
from .convert import sv2mj

__all__ = [
    "__version__",
    "sv2mj",
    "output_path",
]
