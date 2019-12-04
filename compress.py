from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.image as pltimage
import numpy as np


def main():
    full_image = Image.open('samples/bird.png').convert('L')
    img = np.array(full_image)
    pltimage.imsave('ori.jpg', img)
    plt.imshow(img, cmap='gray')
    plt.show()

    imgf = np.fft.fft2(img)

    plt.imshow(np.abs(imgf), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.show()

    imgf.real[abs(imgf.real) < np.max(abs(imgf.real)) * .005] = 0
    imgf.imag[abs(imgf.imag) < np.max(abs(imgf.imag)) * .005] = 0

    plt.imshow(np.abs(imgf), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.show()

    imgc = np.fft.ifft2(imgf)
    plt.imshow(abs(imgc), cmap='gray')
    plt.show()

    # result = Image.fromarray(abs(imgc)).convert('L')
    # result.save('comp.jpg')
    pltimage.imsave('comp.jpg', abs(imgc), cmap='gray')


if __name__ == '__main__':
    main()
