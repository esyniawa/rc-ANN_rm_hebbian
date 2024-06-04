import numpy as np


def reshape_array(m: np.ndarray, dim: int = 2):
    """
    Reduces the dimension of matrix m into a desired dimension.
    :param m:
    :param dim:
    :return:
    """
    shape = m.shape

    for i in range(m.ndim, dim, -1):
        new_shape = list(shape[:-1])
        new_shape[-1] = shape[-1] * shape[-2]
        shape = new_shape

    return m.reshape(shape)


def ceil(a: float, precision: int = 0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)
