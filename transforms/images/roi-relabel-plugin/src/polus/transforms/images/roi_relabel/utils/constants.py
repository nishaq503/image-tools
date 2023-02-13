"""Collection of constants for use across my plugins."""

import logging
import os

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

NUM_THREADS = int(max(1.0, int(1 if (c := os.cpu_count()) is None else c) * 0.5))
TILE_SIZE_2D = 1024
TILE_SIZE_3D = 128
EPSILON = 1e-12


class Unset:
    """For class members that cannot be set at creating time.

    This is a hack around type-hinting when a value cannot be set in the __init__ method for a class.

    See: https://peps.python.org/pep-0661/

    Usage:
    ```python

    class MyClass:

        def __init__(self, *args, **kwargs):
            ...
            self.__value: typing.Union[ValueType, Unset] = UNSET

        def value_setter(self, *args, **kwargs):
            ...
            self.__value = something
            return

        @property
        def value(self) -> ValueType:
            if self.__value is UNSET:
                raise ValueError(f'Please call `value_setter` on the object before using this property.')
            return self.__value
    ```
    """

    __unset = None

    def __new__(cls):
        """You should never call this."""
        if cls.__unset is None:
            cls.__unset = super().__new__(cls)
        return cls.__unset


UNSET = Unset()
