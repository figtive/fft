import math
import numpy as np

def dft2(matrix):
    image = np.zeros(matrix.shape, dtype=complex)
    for col in range(image.shape[1]):
        image[:, col] = dft(matrix[:, col])
    for row in range(image.shape[0]):
        image[row, :] = dft(image[row, :])
    return image


def idft2(matrix):
    image = np.zeros(matrix.shape, dtype=complex)
    for col in range(image.shape[1]):
        image[:, col] = idft(matrix[:, col])
    for row in range(image.shape[0]):
        image[row, :] = idft(image[row, :])
    return image


def dft(vector, N=None):
    if N is None:
        N = len(vector)
    fourier = []
    for k in range(N):
        fourier.append(complex(0))
        for n, x in enumerate(vector):
            theta = math.tau * k * n / N
            fourier[k] += x * complex(math.cos(theta), math.sin(theta))
    return fourier


def idft(vector, N=None):
    if N is None:
        N = len(vector)
    fourier = []
    for k in range(N):
        fourier.append(complex(0))
        for n, x in enumerate(vector):
            theta = - math.tau * k * n / N
            fourier[k] += x * complex(math.cos(theta), math.sin(theta))
    return fourier
