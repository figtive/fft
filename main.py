import numpy as np
from skimage.transform import resize
import matplotlib.image as pltimage
import matplotlib.pyplot as plt
from PIL import Image

from compress import compress
from dft import dft2, idft2
from fft import fft2, ifft2

compression = .95
source = 'samples/lena_color_512.tif'
resolutions = (2, 4, 8)


def main():
    print(f"\n====== Generating images for {source} with resolutions {resolutions}")
    generate_images(source, resolutions)

    np_times = []
    fft_times = []
    dft_times = []
    for res in resolutions:
        print(f"\n====== Compressing for size {res}x{res}")
        print("--- Numpy FFT Compression... ", end='')
        pre, pos = compress(np.fft.fft2, np.fft.ifft2, f'images/{res}.png', compression, f"Numpy-{res}")
        np_times.append((pre + pos))
        print(f"Done! {pre + pos}s")
        print("--- FFT Compression... ", end='')
        pre, pos = compress(fft2, ifft2, f'images/{res}.png', compression, f"FFT-{res}")
        fft_times.append((pre + pos))
        print(f"Done! {pre + pos}s")
        print("--- DFT Compression... ", end='')
        pre, pos = compress(dft2, idft2, f'images/{res}.png', compression, f"DFT-{res}")
        dft_times.append((pre + pos))
        print(f"Done! {pre + pos}s")

    plt.show()
    plt.plot(resolutions, np_times)
    plt.plot(resolutions, fft_times)
    plt.plot(resolutions, dft_times)
    plt.title("Compression Time Summary")
    plt.legend(['Numpy FFT', 'Cooley-Tukey FFT', 'Naive DFT'], loc='upper left')
    plt.xticks(resolutions)
    plt.xlabel('image width and height (pixel)')
    plt.ylabel('time (s)')
    plt.savefig(f'output/summary.png', bbox_inches='tight')
    plt.show(bbox_inches='tight')

    plt.plot(resolutions, np_times)
    plt.plot(resolutions, fft_times)
    plt.plot(resolutions, dft_times)
    plt.title("Compression Time Summary")
    plt.legend(['Numpy FFT', 'Cooley-Tukey FFT', 'Naive DFT'], loc='upper left')
    plt.xticks(resolutions)
    plt.xlabel('image width and height (pixel)')
    plt.ylabel('time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'output/summary-log.png', bbox_inches='tight')
    plt.show(bbox_inches='tight')


def generate_images(source, resolutions):
    full_image = Image.open(source).convert('L')
    img = np.array(full_image)
    img = crop_power(img)
    print(f"Cropping image {source} to {img.shape[0]}x{img.shape[1]}")
    for res in resolutions:
        scaled = resize(img, (res, res), anti_aliasing=True)
        pltimage.imsave(f'images/{res}.png', abs(scaled), cmap='gray')


def crop_power(img):
    lower_x = lower_power(img.shape[0]) // 2
    lower_y = lower_power(img.shape[1]) // 2
    return img[img.shape[0] // 2 - lower_x:img.shape[0] // 2 + lower_x,
           img.shape[1] // 2 - lower_y:img.shape[1] // 2 + lower_y]


def lower_power(number):
    return 1 << number.bit_length() - 1


if __name__ == "__main__":
    main()
