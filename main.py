import numpy as np
from skimage.transform import resize
import matplotlib.image as pltimage
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate

from compress import compress
from dft import dft2, idft2
from fft import fft2, ifft2, fftp2, ifftp2, ffth2, iffth2

compression = .95
source = 'samples/lena_color_512.tif'
resolutions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

np_times = []
fft_times = []
fftp_times = []
ffth_times = []
dft_times = []


def main():
    print(f"\n====== Generating images for {source} with resolutions {resolutions}")
    generate_images(source, resolutions)

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

        print("--- FFTP Compression... ", end='')
        pre, pos = compress(fftp2, ifftp2, f'images/{res}.png', compression, f"FFTP-{res}")
        fftp_times.append((pre + pos))
        print(f"Done! {pre + pos}s")

        print("--- FFTH Compression... ", end='')
        pre, pos = compress(ffth2, iffth2, f'images/{res}.png', compression, f"FFTH-{res}")
        ffth_times.append((pre + pos))
        print(f"Done! {pre + pos}s")

        print("--- DFT Compression... ", end='')
        pre, pos = compress(dft2, idft2, f'images/{res}.png', compression, f"DFT-{res}")
        dft_times.append((pre + pos))
        print(f"Done! {pre + pos}s")

    plt.clf()
    inp = interpolate.splrep(resolutions, np_times, s=0)
    ifft = interpolate.splrep(resolutions, fft_times, s=0)
    ifftp = interpolate.splrep(resolutions, fftp_times, s=0)
    iffth = interpolate.splrep(resolutions, ffth_times, s=0)
    idft = interpolate.splrep(resolutions, dft_times, s=0)
    res = np.arange(0, 512, 1)
    np_inter = interpolate.splev(res, inp, der=0)
    fft_inter = interpolate.splev(res, ifft, der=0)
    fftp_inter = interpolate.splev(res, ifftp, der=0)
    ffth_inter = interpolate.splev(res, iffth, der=0)
    dft_inter = interpolate.splev(res, idft, der=0)

    plt.plot(res, np_inter, "-b", res, fft_inter, "-g", res, fftp_inter, "-c", res, ffth_inter, "-m", res, dft_inter,
             "-r")
    plt.legend(['NumPy FFT', 'Cooley-Tukey FFT', 'Parallel FFT', 'Hybrid FFT', 'Naive DFT'], loc='upper left')
    plt.plot(resolutions[:-1], np_times[:-1], "bx", resolutions[:-1], fft_times[:-1], "gx", resolutions[:-1],
             fftp_times[:-1], "cx", resolutions[:-1], ffth_times[:-1], "mx", resolutions[:-1], dft_times[:-1], "rx")

    plt.title("Compression Time Summary")
    plt.xlabel('image width and height (pixel)')
    plt.ylabel('time (s)')
    plt.savefig(f'output/summary.png', bbox_inches='tight', dpi=300)
    plt.clf()

    plt.plot(res, np_inter, "-b", res, fft_inter, "-g", res, fftp_inter, "-c", res, ffth_inter, "-m", res, dft_inter,
             "-r")
    plt.legend(['NumPy FFT', 'Cooley-Tukey FFT', 'Parallel FFT', 'Hybrid FFT', 'Naive DFT'], loc='upper left')
    plt.plot(resolutions[:-1], np_times[:-1], "bx", resolutions[:-1], fft_times[:-1], "gx", resolutions[:-1],
             fftp_times[:-1], "cx", resolutions[:-1], ffth_times[:-1], "mx", resolutions[:-1], dft_times[:-1], "rx")

    plt.title("Compression Time Summary")
    plt.xlabel('image width and height (pixel)')
    plt.ylabel('time (s)')
    plt.yscale('log')
    plt.savefig(f'output/summary-log.png', bbox_inches='tight', dpi=300)
    plt.clf()

    plt.plot(res, fft_inter, "-g", res, fftp_inter, "-c", res, ffth_inter, "-m")
    plt.legend(['Cooley-Tukey FFT', 'Parallel FFT', 'Hybrid FFT'], loc='upper left')
    plt.plot(resolutions[:-1], fft_times[:-1], "gx", resolutions[:-1],
             fftp_times[:-1], "cx", resolutions[:-1], ffth_times[:-1], "mx")

    plt.title("Compression Time Summary")
    plt.xlabel('image width and height (pixel)')
    plt.ylabel('time (s)')
    plt.savefig(f'output/summary-focus.png', bbox_inches='tight', dpi=300)
    plt.clf()

    plt.plot(res, fft_inter, "-g", res, fftp_inter, "-c", res, ffth_inter, "-m")
    plt.legend(['Cooley-Tukey FFT', 'Parallel FFT', 'Hybrid FFT'], loc='upper left')
    plt.plot(resolutions[:-1], fft_times[:-1], "gx", resolutions[:-1],
             fftp_times[:-1], "cx", resolutions[:-1], ffth_times[:-1], "mx")

    plt.title("Compression Time Summary")
    plt.xlabel('image width and height (pixel)')
    plt.ylabel('time (s)')
    plt.yscale('log')
    plt.savefig(f'output/summary-focus-log.png', bbox_inches='tight', dpi=300)
    plt.clf()


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
