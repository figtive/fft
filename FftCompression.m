function [ a ] = FFTCompression()
    pkg load image;

    image = imread('image1', 'jpg');
    figure(1), imshow(image);

    imageBw = rgb2gray(image);
    [nx, ny] = size(imageBw);
    figure(2), imshow(imageBw);

    imageFft = fft2(imageBw);
    fourierCoefficient = log(abs(fftshift(imageFft)) + 1);
    fourierCoefficient = mat2gray(fourierCoefficient);
    figure(3), imshow(fourierCoefficient);

    threshold = 0.001 * max(max(abs(imageFft)));
    indices = abs(imageFft) > threshold;
    imageCompressed = imageFft .* indices;
    imageCompressedConverted = uint8(real(ifft2(imageCompressed)));
    count = nx * ny - sum(sum(indices));
    percentage = 100 - count / (nx * ny) * 100;
    figure(4), imshow(imageCompressedConverted);
    title(num2str(percentage));

    pkg unload image;
endfunction
