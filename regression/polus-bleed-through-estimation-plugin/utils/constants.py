import logging
import os

POLUS_LOG = getattr(logging, os.environ.get('POLUS_LOG', 'INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT', '.ome.tif')

# NUM_THREADS = max(1, int(cpu_count() * 0.5))  # TODO: What's the point?
TILE_SIZE_2D = 1024 * 2
TILE_SIZE_3D = 128
