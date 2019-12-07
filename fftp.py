import math
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Manager, Process, Pool
"""
NOT USED
"""

def fftp2(matrix):
    image = np.zeros(matrix.shape, dtype=complex)
    for row in range(image.shape[0]):
        image[row, :] = fftp(matrix[row, :])
    for col in range(image.shape[1]):
        image[:, col] = fftp(image[:, col])
    return image


def ifftp2(matrix):
    image = np.zeros(matrix.shape, dtype=complex)
    for row in range(image.shape[0]):
        image[row, :] = fftp([c.conjugate() / (len(matrix[row, :])) for c in matrix[row, :]])
    for col in range(image.shape[1]):
        image[:, col] = fftp([c.conjugate() / (len(image[:, col])) for c in image[:, col]])
    return np.flip(image, 0)

def fftp(vector, N=None, w=None):
    if N == 1:
        return vector
    else:
        if N is None:
            N = len(vector)
        if w is None:
            w = complex(math.cos(math.tau / N), math.sin(math.tau / N))
        # if pool is None:
        #     print('New pool')
        #     pool = Pool(processes=16)
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
