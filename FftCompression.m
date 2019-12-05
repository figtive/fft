function [ a ] = FFTCompression()
    pkg load image;

    image = imread('samples/image1', 'jpg');
    figure(1), imshow(image);
    imwrite(image, 'output/image1-original.jpg')
    tic

    imageBw = rgb2gray(image);
    [nx, ny] = size(imageBw);
    figure(2), imshow(imageBw);
    imwrite(imageBw, 'output/image1-bw.jpg')
    toc

    imageFft = fft2(imageBw);
    fourierCoefficient = log(abs(fftshift(imageFft)) + 1);
    fourierCoefficient = mat2gray(fourierCoefficient);
    figure(3), imshow(fourierCoefficient);
    imwrite(fourierCoefficient, 'output/image1-fft.jpg')
    toc

    threshold = 0.001 * max(max(abs(imageFft)));
    indices = abs(imageFft) > threshold;
    imageCompressed = imageFft .* indices;
    imageCompressedConverted = uint8(real(ifft2(imageCompressed)));
    count = nx * ny - sum(sum(indices));
    percentage = 100 - count / (nx * ny) * 100;
    figure(4), imshow(imageCompressedConverted);
    title(num2str(percentage));
    imwrite(imageCompressedConverted, 'output/image1-compressed.jpg')
    toc

    pkg unload image;
endfunction
