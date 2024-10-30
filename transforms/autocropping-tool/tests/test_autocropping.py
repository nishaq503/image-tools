"""Tests for autocropping."""

import pytest

from polus.images.transforms.autocropping.autocropping import autocrop_group


def test_autocropping():
    """Test autocropping."""
    # TODO: Add tests
    pass


@pytest.mark.skipif("not config.getoption('slow')")
def test_slow_autocropping():
    """Test that can take a long time to run."""
    # TODO: Add optional tests
    pass


@pytest.mark.skipif("not config.getoption('downloads')")
def test_download_autocropping():
    """Test thatdownload data from."""
    # TODO: Add optional tests
    pass
