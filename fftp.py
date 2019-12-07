import math
import numpy as np
from multiprocessing.pool import ThreadPool, Pool

"""
NOT USED
"""


# class NoDaemonProcess(Process):
#     def _get_daemon(self):
#         return False
#
#     def _set_daemon(self, value):
#         pass
#
#     daemon = property(_get_daemon, _set_daemon)
#
#
# class Pool(PoolParent):
#     Process = NoDaemonProcess


def fftp2(matrix):
    with Pool(20) as pool:
        image = np.array(pool.map(fftp, matrix))
        image = np.array(pool.map(fftp, image.T)).T
    return image


def ifftp2(matrix):
    with Pool(20) as pool:
        image = np.array(pool.map(fftp, [c.conjugate() / (len(matrix)) for c in matrix]))
        image = np.array(pool.map(fftp, [c.conjugate() / (len(image.T)) for c in image.T])).T
    return np.flip(image, 0)


def fftp(vector, N=None, w=None):
    if N == 1:
        return vector
    else:
        if N is None:
            N = len(vector)
        if w is None:
            w = complex(math.cos(math.tau / N), math.sin(math.tau / N))
        vector = padding(vector, nearest_power(N))
        with ThreadPool(2) as pool:
            result = pool.starmap(fftp, [(vector[0::2], nearest_power(N) // 2, w ** 2),
                                         (vector[1::2], nearest_power(N) // 2, w ** 2)])
        x = 1
        fourier = [0] * N
        for i in range(N // 2):
            fourier[i] = result[0][i] + x * result[1][i]
            fourier[i + N // 2] = result[0][i] - x * result[1][i]
            x *= w
        return fourier


def nearest_power(number):
    return 1 << (number - 1).bit_length()


def padding(vector, pad):
    return np.append(vector, np.zeros(pad - len(vector)))
