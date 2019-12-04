import math


def fft(vector, n=None, w=None):
    if n == 1:
        return vector
    else:
        if n is None:
            n = len(vector)
        if w is None:
            w = complex(math.cos(math.tau / n), math.sin(math.tau / n))
        vector = padding(vector, nearest_power(n))
        even, odd = split_even_odd(vector)
        fourier_even = fft(even, nearest_power(n) // 2, w ** 2)
        fourier_odd = fft(odd, nearest_power(n) // 2, w ** 2)
        fourier = [0] * n
        x = 1
        for i in range(n // 2):
            print(i)
            fourier[i] = fourier_even[i] + x * fourier_odd[i]
            fourier[i + n // 2] = fourier_even[i] - x * fourier_odd[i]
            x *= w
        return fourier


def nearest_power(number):
    return 1 << (number - 1).bit_length()


def padding(vector, pad):
    return vector + [0] * (pad - len(vector))


def split_even_odd(vector):
    even = []
    odd = []
    for i, element in enumerate(vector):
        if i % 2:
            odd.append(element)
        else:
            even.append(element)
    return even, odd
