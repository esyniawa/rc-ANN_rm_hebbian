import numpy as np
from typing import Optional

def reshape_array(m: np.ndarray, dim: int = 2) -> np.ndarray:
    """
    Reduces the dimension of a tensor m into a desired dimension.
    :param m: numpy array
    :param dim: reshape m into a dim dimensional array
    :return: m with np.ndim(m) = dim
    """
    shape = m.shape

    for i in range(m.ndim, dim, -1):
        new_shape = list(shape[:-1])
        new_shape[-1] = shape[-1] * shape[-2]
        shape = new_shape

    return m.reshape(shape)


def ceil(a: float, precision: int = 0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def moving_average(data, window_size, dim: Optional[int] = None):
    if not isinstance(data, np.ndarray) or not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Input 'data' must be a numpy array and 'window_size' must be a positive integer.")

    data = np.asarray(data)
    ret = np.cumsum(data, dtype=float, axis=dim)

    if window_size > data.shape[0]:
        return np.array([])

    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1 :] / window_size