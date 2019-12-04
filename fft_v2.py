import numpy as np
import math


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


def fft(vector, N=None, w=None):
    if N == 1:
        return vector
    else:
        if N is None:
            N = len(vector)
        if w is None:
            w = complex(math.cos(math.tau / N), math.sin(math.tau / N))
        vector = padding(vector, nearest_power(N))
        even, odd = split_even_odd(vector)
        fourier_even = fft(even, nearest_power(N) // 2, w ** 2)
        fourier_odd = fft(odd, nearest_power(N) // 2, w ** 2)
        fourier = [0] * N
        x = 1
        for i in range(N // 2):
            print(i)
            fourier[i] = fourier_even[i] + x * fourier_odd[i]
            fourier[i + N // 2] = fourier_even[i] - x * fourier_odd[i]
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
    return (even, odd)


def print_poly(poly):
    for i, elem in enumerate(poly):
        if i == 0:
            print(f"{elem.real}x{i:.1f}", end='')
        elif elem.real != 0:
            print(f"{' + ' if elem.real > 0 else ' - '}{abs(elem.real):.1f}x{i}", end='')
    print()


if __name__ == "__main__":
    A = [1, 2, 3, 4, 5, 6]
    print(split_even_odd(padding(A, len(A))))
    print(fft(A))
    # print(np.fft.ifft(len(A)*A))

    print()

    # A = [-3, 0.5, 4, 0, 1,1]
    # B = [-4, 0, 1, 5, 1, 4, 10, -8]

    A = [-3, 0.5, 3]
    B = [-4, 0, 1]
    print_poly(A)
    print_poly(B)

    # fA = np.fft.ifft(4*A,4)
    # fB = np.fft.ifft(4*B,4)
    # print(f"A: {fA}")
    # print(f"B: {fB}")
    # print(f"C: {fA*fB}")
    # print(np.fft.ifft(fA*fB))

    # print("\nFFT")

    m = len(A) + len(B) - 1
    fA = fft(A, m)
    fB = fft(B, m)
    fC = [fA[i] * fB[i] for i in range(len(fA))]
    print(f"A {len(fA)} : {fA}")
    # print(f"B: {fB}")
    # print(f"C: {fC}")
    C = fft([c.conjugate() / (len(fC)) for c in fC], m)
    # print(C)
    print_poly(C)

    print()

    m = len(A) + len(B) - 1
    fA = dft(A, m)
    fB = dft(B, m)
    fC = [fA[i] * fB[i] for i in range(len(fA))]
    print(f"A {len(fA)} : {fA}")
    # print(f"B: {fB}")
    # print(f"C: {fC}")
    C = dft([c.conjugate() / (len(fC)) for c in fC], m)
    # print(C)
    print_poly(C)

    print()

    x = complex(1)
    w = complex(math.cos(2 * math.pi / 8), math.sin(2 * math.pi / 8))
    for i in range(16):
        print(f"{i} -> {x}")
        x *= w
