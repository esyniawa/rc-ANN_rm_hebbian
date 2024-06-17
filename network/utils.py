import numpy as np
from typing import Optional


def gauss_kernel(window_size: int, sd: float, amplitude: float | None = None):
    # gauss kernel should have a maximum on 0
    if window_size % 2 == 0:
        n = window_size + 1
    else:
        n = window_size

    x = np.linspace(-window_size/2, window_size/2, n, endpoint=True)

    if amplitude is None:
        amplitude = 1/(sd * np.sqrt(2 * np.pi))

    return amplitude * np.exp(-0.5 * np.power(x/sd, 2))


def exponential_smoothing(m: np.ndarray, windows_size: int = 100, tau: float = 50., axis: int = 0):
    s = np.arange(windows_size)
    kernel = np.exp(-s/tau)
    return np.apply_along_axis(lambda x: np.convolve(x, kernel/np.sum(kernel), mode='same'), axis=axis, arr=m)


def gauss_convolve(m: np.ndarray, window_size: int, sd: float, amplitude: float | None = None, axis: int = 0):
    g = gauss_kernel(window_size, sd, amplitude)
    return np.apply_along_axis(lambda x: np.convolve(x, g, mode='same'), axis=axis, arr=m)


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


def memory_out(memory_trace: np.ndarray, rng_trace: np.ndarray) -> np.ndarray:
    T = len(memory_trace)
    result = np.zeros(T)
    for t in range(T):
        if memory_trace[t] < 0.5:
            result[t] = rng_trace[t, 0]
        else:
            result[t] = rng_trace[t, 1]
    return result

if __name__ == '__main__':
    a = np.random.normal(size=(10,2))

    print(a.shape)
    print(gauss_convolve(a, 11, sd=2, axis=0).shape)