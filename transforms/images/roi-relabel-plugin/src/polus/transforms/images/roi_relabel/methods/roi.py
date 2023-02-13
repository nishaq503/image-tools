"""RoI objects."""

import math

__all__ = [
    "RoI",
]


class Point:
    __slots__ = [
        "x",
        "y",
    ]

    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    def __lt__(self, other: "Point") -> bool:
        return self.x < other.x or self.y < other.y

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def min(self, other: "Point") -> "Point":
        return Point(
            x=min(self.x, other.x),
            y=min(self.y, other.y),
        )

    def max(self, other: "Point") -> "Point":
        return Point(
            x=max(self.x, other.x),
            y=max(self.y, other.y),
        )

    def mid(self, other: "Point") -> "Point":
        return Point(
            x=(self.x + other.x) / 2,
            y=(self.y + other.y) / 2,
        )

    def distance_to(self, other: "Point") -> float:
        x = self.x - other.x
        y = self.y - other.y
        return math.sqrt(x * x + y * y)

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y


class RoI:
    """Create and manage RoIs from a labeled image."""

    __slots__ = [
        "top_left",
        "bottom_right",
        "label",
        "center",
    ]

    def __init__(
        self,
        top_left: tuple[float, float],
        bottom_right: tuple[float, float],
        label: int,
    ):
        """Create a new RoI from the top-left and bottom-right corners of a rectangle.

        Args:
            top_left: corner of rectangle.
            bottom_right: corner of rectangle.
            label: integer label for RoI. Also used for hashing RoIs for use with graph-based algorithms.
        """
        self.top_left = Point(*top_left)
        self.bottom_right = Point(*bottom_right)
        self.label = label
        self.center = self.top_left.mid(self.bottom_right)

    def __lt__(self, other: "RoI") -> bool:
        """Compare and order RoIs and help duplicated code.

        Args:
            other: RoI for comparison.

        Returns:
            Whether `self` should be ordered before `other`.
        """
        if self.top_left == other.top_left:
            return self.bottom_right < other.bottom_right
        else:
            return self.top_left < other.top_left

    def __hash__(self) -> int:
        """Hash RoIs for use in networkx graphs."""
        return hash(self.label)

    @property
    def range(self) -> float:
        """Compute the radial range of the RoI."""
        return self.center.distance_to(self.top_left)

    def merge_with(self, other: "RoI") -> "RoI":
        """Merge with another RoI and return the new ROI.

        Args:
            other: an RoI.
        """
        return RoI(
            top_left=self.top_left.min(other.top_left).as_tuple(),
            bottom_right=self.bottom_right.max(other.bottom_right).as_tuple(),
            label=self.label,
        )

    def touches(self, other: "RoI") -> bool:
        """Check whether the bounding boxes of the RoIs touch each other.

        Args:
            other: an RoI.
        """
        if other < self:
            return other.touches(self)
        else:
            return (
                self.bottom_right.x >= other.top_left.x
                and self.bottom_right.y >= other.top_left.y
            )

    def in_range_of(self, other: "RoI", multiplier: float = 1.0) -> bool:
        """Check whether the two RoIs within range to have different labels.

        Args:
            other: an RoI.
            multiplier: on the default radial range.
        """
        d = self.center.distance_to(other.center)
        return d <= ((self.range + other.range) * multiplier)
