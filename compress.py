from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.image as pltimage
import numpy as np
from dft import dft2, idft2
from fft import fft2, ifft2
from time import perf_counter


def compress(trans, itrans, source, compression, desc):
    fig, axs = plt.subplots(2, 2)
    full_image = Image.open(source).convert('L')
    img = np.array(full_image)
    img = crop_power(img)
    print(f"Cropping image to {img.shape[0]}x{img.shape[1]}")

    pltimage.imsave('original.jpg', img, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].imshow(img, cmap='gray')

    start = perf_counter()
    imgf = trans(img)
    stop = perf_counter()
    print(f"Transform time: {stop - start} s")

    axs[0, 1].set_title(f"Time: {stop - start:.5f}s")
    axs[0, 1].axis('off')
    im1 = axs[0, 1].imshow(np.abs(np.fft.fftshift(imgf)), norm=LogNorm(vmin=5))
    fig.colorbar(im1, ax=axs[0, 1])

    print(f"Compression: {compression * 100}%")
    imgf[abs(imgf) < np.percentile(abs(imgf), compression * 100)] = 0

    axs[1, 1].axis('off')
    im2 = axs[1, 1].imshow(np.abs(np.fft.fftshift(imgf)), norm=LogNorm(vmin=5))
    fig.colorbar(im2, ax=axs[1, 1])

    start = perf_counter()
    imgc = itrans(imgf)
    stop = perf_counter()
    print(f"Transform time: {stop - start} s")

    axs[1, 1].set_title(f"Time: {stop - start:.5f}s")
    axs[1, 0].axis('off')
    axs[1, 0].imshow(abs(imgc), cmap='gray')

    fig.suptitle(desc, fontsize=16)
    plt.show()

    # result = Image.fromarray(abs(imgc)).convert('L')
    # result.save('comp.jpg')
    pltimage.imsave('compressed.jpg', abs(imgc), cmap='gray')


def crop_power(img):
    lower_x = lower_power(img.shape[0]) // 2
    lower_y = lower_power(img.shape[1]) // 2
    return img[img.shape[0] // 2 - lower_x:img.shape[0] // 2 + lower_x,
           img.shape[1] // 2 - lower_y:img.shape[1] // 2 + lower_y]


def lower_power(number):
    return 1 << (number).bit_length() - 1


if __name__ == '__main__':
    print("\n------ Numpy FFT Compression ------")
    compress(np.fft.fft2, np.fft.ifft2, 'samples/lena_color_512.tif', .95, "Numpy FFT")
    print("\n------ FFT Compression ------")
    compress(fft2, ifft2, 'samples/lena_color_512.tif', .95, "FFT")
    print("\n------ DFT Compression ------")
    compress(dft2, idft2, 'samples/lena_color_512.tif', .99, "DFT")
