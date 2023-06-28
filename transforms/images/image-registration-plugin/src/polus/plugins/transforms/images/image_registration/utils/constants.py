"""Constants for the Image Registration plugin."""

import logging
import multiprocessing
import os

MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
