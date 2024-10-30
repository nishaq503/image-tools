"""Helpers for dealing with finding and merging bounding boxes."""

import enum
import pathlib

import bfio
import numpy

from . import entropy
from . import gradients
from . import utils


class BoundingBox:
    """A bounding box in an image.

    The intended user-facing methods are `from_image` and `crop_image`.
    """

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

    def __repr__(self) -> str:
        """Return a string representation of the bounding box."""
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2})"

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

    @property
    def is_valid(self) -> bool:
        """Check if the bounding box is valid.

        Returns:
            True if the bounding box is valid, False otherwise.
        """
        return self.x1 < self.x2 and self.y1 < self.y2

    @property
    def dx(self) -> int:
        """The width of the bounding box."""
        return self.x2 - self.x1

    @property
    def dy(self) -> int:
        """The height of the bounding box."""
        return self.y2 - self.y1

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the bounding box as a tuple.

        Returns:
            A tuple of the form (xmin, ymin, xmax, ymax).
        """
        return self.x1, self.y1, self.x2, self.y2

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

    @staticmethod
    def from_image(
        reader: bfio.BioReader,
        rows: bool,
        cols: bool,
        threshold: float,
    ) -> "BoundingBox":
        """Create a bounding box from an image.

        Args:
            reader: The BioReader to read the image from.
            rows: Whether to consider rows when creating the bounding box.
            cols: Whether to consider columns when creating the bounding box.
            threshold: The threshold to use when finding spikes in the entropy
                gradients.

        Returns:
            A bounding box that encompasses the high-entropy regions of the image.

        Raises:
            ValueError: If no valid bounding boxes are found.
        """
        boxes = []
        start = BoundingBox(reader.X, reader.Y, 0, 0)

        for x_min, y_min, x_max, y_max in utils.iter_tiles_bfio(reader):
            tile = numpy.squeeze(reader[y_min:y_max, x_min:x_max, ...])
            bounding_box = BoundingBox.from_tile(tile, rows, cols, threshold)
            boxes.append(bounding_box)

        b_box = sum(boxes, start)
        if not b_box.is_valid:
            msg = f"Found an invalid bounding box: {b_box}"
            raise ValueError(msg)

        return b_box

    def crop_image(self, reader: bfio.BioReader, out_path: pathlib.Path) -> None:
        """Crop an image to this bounding box.

        Args:
            reader: The BioReader to read the image from.
            out_path: The path to save the cropped image to.
        """
        with bfio.BioWriter(out_path, metadata=reader.metadata) as writer:
            writer.X = self.dx
            writer.Y = self.dy

            for r_box, w_box in zip(
                utils.iter_tiles_bbox(self.as_tuple),
                utils.iter_tiles_bfio(writer),
            ):
                rx1, ry1, rx2, ry2 = r_box
                wx1, wy1, wx2, wy2 = w_box

                writer[wy1:wy2, wx1:wx2, ...] = reader[ry1:ry2, rx1:rx2, ...]


class EdgeTile(enum.Enum):
    """The type of tile along the edge of an image."""

    TopLeft = 0
    """The top-left tile."""

    TopRight = 1
    """The top-right tile."""

    BottomLeft = 2
    """The bottom-left tile."""

    BottomRight = 3
    """The bottom-right tile."""

    TopEdge = 4
    """A tile along the top edge but not in a corner."""

    BottomEdge = 5
    """A tile along the bottom edge but not in a corner."""

    LeftEdge = 6
    """A tile along the left edge but not in a corner."""

    RightEdge = 7
    """A tile along the right edge but not in a corner."""

    Inner = 8
    """An inner tile, not along any edge or in a corner."""

    @staticmethod
    def from_coords(
        image: tuple[int, int, int, int],
        tile: tuple[int, int, int, int],
    ) -> "EdgeTile":
        """Determine the type of tile from its coordinates.

        Args:
            image: The coordinates of the image.
            tile: The coordinates of the tile.

        Returns:
            The type of tile.
        """
        x1, y1, x2, y2 = tile
        ix1, iy1, ix2, iy2 = image

        edge = EdgeTile.Inner

        if x1 == ix1 and y1 == iy1:
            edge = EdgeTile.TopLeft
        elif x2 == ix2 and y1 + utils.TILE_SIZE >= iy2:
            edge = EdgeTile.TopRight
        elif x1 + utils.TILE_SIZE >= ix2 and y1 == iy1:
            edge = EdgeTile.BottomLeft
        elif x2 == ix2 and y2 == iy2:
            edge = EdgeTile.BottomRight
        elif x1 == ix1:
            edge = EdgeTile.LeftEdge
        elif x2 + utils.TILE_SIZE >= ix2:
            edge = EdgeTile.RightEdge
        elif y1 == iy1:
            edge = EdgeTile.TopEdge
        elif y2 + utils.TILE_SIZE >= iy2:
            edge = EdgeTile.BottomEdge

        return edge
