"""Helpers for dealing with finding and merging bounding boxes."""

import numpy

from . import entropy
from . import gradients


class BoundingBox:
    """A bounding box in an image."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Create a new bounding box.

        Args:
            x1: The x-coordinate of the top-left corner of the bounding box.
            y1: The y-coordinate of the top-left corner of the bounding box.
            x2: The x-coordinate of the bottom-right corner of the bounding box.
            y2: The y-coordinate of the bottom-right corner of the bounding box.
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __add__(self, other: "BoundingBox") -> "BoundingBox":
        """Merge this bounding box with another bounding box.

        Args:
            other: The bounding box to merge with.

        Returns:
            A new bounding box that encompasses both bounding boxes.
        """
        return BoundingBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    @staticmethod
    def from_tile(
        tile: numpy.ndarray,
        rows: bool,
        cols: bool,
        threshold: float,
    ) -> "BoundingBox":
        """Create a bounding box from a tile.

        At least one of rows and cols must be True.

        Args:
            tile: A numpy array (at least 2D) to create a bounding box from.
            rows: Whether to consider rows when creating the bounding box.
            cols: Whether to consider columns when creating the bounding box.
            threshold: The threshold to use when finding spikes in the entropy
                gradients.

        Returns:
            A bounding box that encompasses the high-entropy regions of the tile.

        Raises:
            ValueError: If both rows and cols are False.
        """
        if not rows and not cols:
            msg = "At least one of rows and cols must be True."
            raise ValueError(msg)

        if rows:
            row_entropy = entropy.row_wise_entropy(tile)
            y1, y2 = gradients.find_spike_indices(row_entropy, threshold)
        else:
            y1, y2 = 0, tile.shape[0]

        if cols:
            col_entropy = entropy.col_wise_entropy(tile)
            x1, x2 = gradients.find_spike_indices(col_entropy, threshold)
        else:
            x1, x2 = 0, tile.shape[1]

        return BoundingBox(x1, y1, x2, y2)

    def crop_tile(self, tile: numpy.ndarray) -> numpy.ndarray:
        """Crop the given tile to this bounding box.

        Args:
            tile: A numpy array (at least 2D) to crop.

        Returns:
            The cropped tile.
        """
        return tile[self.y1 : self.y2, self.x1 : self.x2, ...]
