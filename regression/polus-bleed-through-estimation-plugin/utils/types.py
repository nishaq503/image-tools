from pathlib import Path
from typing import Union

"""
A dictionary of scores for each tile in an image.
key: (6-tuple of indices) (z_min, z_max, y_min, y_max, x_min, x_max)
value: (float) score
"""
ScoresDict = dict[tuple[int, int, int, int, int, int], float]

"""
A list of coordinates for each tile that was selected by a Selector.
Each item is a 6-tuple of indices: (z_min, z_max, y_min, y_max, x_min, x_max)
"""
TileIndices = list[tuple[int, int, int, int, int, int]]

"""
A dictionary for a file in `filepattern`.
"""
FPFileDict = dict[str, Union[int, Path]]
