import math


def dft(vector, n=None):
    if n is None:
        n = len(vector)
    fourier = []
    for k in range(n):
        fourier.append(complex(0))
        for n, x in enumerate(vector):
            theta = math.tau * k * n / n
            fourier[k] += x * complex(math.cos(theta), math.sin(theta))
    return fourier
