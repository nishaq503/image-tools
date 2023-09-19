"""The main functions for the Image Registration plugin."""

import enum

from .image_registration import main


class Method(str, enum.Enum):
    """The registration methods."""

    Projective = "Projective"
    Affine = "Affine"
    PartialAffine = "PartialAffine"
