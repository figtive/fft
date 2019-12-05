import math
import numpy as np


def fft2(matrix):
    image = np.zeros(matrix.shape, dtype=complex)
    for row in range(image.shape[0]):
        image[row, :] = fft(matrix[row, :])
    for col in range(image.shape[1]):
        image[:, col] = fft(image[:, col])
    return image


def ifft2(matrix):
    image = np.zeros(matrix.shape, dtype=complex)
    for row in range(image.shape[0]):
        image[row, :] = fft([c.conjugate() / (len(matrix[row, :])) for c in matrix[row, :]])
    for col in range(image.shape[1]):
        image[:, col] = fft([c.conjugate() / (len(image[:, col])) for c in image[:, col]])
    return image


def fft(vector, N=None, w=None):
    if N == 1:
        return vector
    else:
        if N is None:
            N = len(vector)
        if w is None:
            w = complex(math.cos(math.tau / N), math.sin(math.tau / N))
        vector = padding(vector, nearest_power(N))
        fourier_even = fft(vector[0::2], nearest_power(N) // 2, w ** 2)
        fourier_odd = fft(vector[1::2], nearest_power(N) // 2, w ** 2)
        fourier = [0] * N
        x = 1
        for i in range(N // 2):
            fourier[i] = fourier_even[i] + x * fourier_odd[i]
            fourier[i + N // 2] = fourier_even[i] - x * fourier_odd[i]
            x *= w
        return fourier


def nearest_power(number):
    return 1 << (number - 1).bit_length()


def padding(vector, pad):
    return np.append(vector, np.zeros(pad - len(vector)))


def split_even_odd(vector):
    even = []
    odd = []
    for i, element in enumerate(vector):
        if i % 2:
            odd.append(element)
        else:
            even.append(element)
    return even, odd
