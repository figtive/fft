from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.image as pltimage
import numpy as np
from time import perf_counter


def compress(trans, itrans, source, compression, desc):
    fig, axs = plt.subplots(2, 2)
    full_image = Image.open(source).convert('L')
    img = np.array(full_image)

    # pltimage.imsave(f'output/original-{desc}.png', img, cmap='gray')
    axs[0, 0].imshow(img, cmap='gray')

    start = perf_counter()
    imgf = trans(img)
    stop = perf_counter()
    time1 = stop - start
    # print(f"Transform time: {time1} s")

    axs[0, 1].set_title(f"Time: {time1:.5f}s")
    im1 = axs[0, 1].imshow(np.abs(np.fft.fftshift(imgf)), norm=LogNorm(vmin=5))
    fig.colorbar(im1, ax=axs[0, 1])

    # print(f"Compression: {compression * 100}%")
    imgf[abs(imgf) < np.percentile(abs(imgf), compression * 100)] = 0

    im2 = axs[1, 1].imshow(np.abs(np.fft.fftshift(imgf)), norm=LogNorm(vmin=5))
    fig.colorbar(im2, ax=axs[1, 1])

    start = perf_counter()
    imgc = itrans(imgf)
    stop = perf_counter()
    time2 = stop - start
    # print(f"Transform time: {time2} s")

    axs[1, 1].set_title(f"Time: {time2:.5f}s")
    axs[1, 0].imshow(abs(imgc), cmap='gray')

    axs[0, 0].xaxis.set_major_locator(plt.NullLocator())
    axs[0, 0].yaxis.set_major_locator(plt.NullLocator())
    axs[1, 0].xaxis.set_major_locator(plt.NullLocator())
    axs[1, 0].yaxis.set_major_locator(plt.NullLocator())
    axs[0, 1].xaxis.set_major_locator(plt.NullLocator())
    axs[0, 1].yaxis.set_major_locator(plt.NullLocator())
    axs[1, 1].xaxis.set_major_locator(plt.NullLocator())
    axs[1, 1].yaxis.set_major_locator(plt.NullLocator())
    fig.suptitle(desc)
    plt.savefig(f'output/summary-{desc}.png', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()

    # result = Image.fromarray(abs(imgc)).convert('L')
    # result.save('comp.jpg')
    pltimage.imsave(f'output/compressed-{desc}.png', abs(imgc), cmap='gray')

    return time1, time2
