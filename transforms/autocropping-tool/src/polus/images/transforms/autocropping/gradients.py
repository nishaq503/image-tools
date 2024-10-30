"""Helpers for calculating gradients and spikes in a sequence of entropy values."""

import numpy


def find_spike_indices(data: numpy.ndarray, threshold: float) -> tuple[int, int]:
    """Find the indices of the first and last spikes in a sequence of entropy values.

    Args:
        data: A numpy array (1D) representing a sequence of entropy values.
        threshold: A float (positive) representing the threshold for the gradient.

    Returns:
        A tuple of two integers, the first representing the index of the start
        of the first spike and the second representing the index of the end of
        the last spike.
    """
    first_spike = find_spike(data, threshold)
    last_spike = find_spike(data[::-1], threshold)
    return first_spike, len(data) - last_spike - 1


def find_spike(data: numpy.ndarray, threshold: float) -> int:
    """Return the index of the start of the first spike in the data.

    Args:
        data: A numpy array (1D) representing a sequence of entropy values.
        threshold: A float (positive) representing the threshold for the gradient.

    Returns:
        An integer representing the index of the start of the first spike.
    """
    # We expect gradients to be nearly zero before the spike and large when the
    # spike starts

    # Find the first spike
    return numpy.argmax(numpy.abs(numpy.gradient(data)) > threshold)


__all__ = ["find_spike_indices"]
