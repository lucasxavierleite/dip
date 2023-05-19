#!/usr/bin/env python

'''
Image Processing course assignment 2.

University of São Paulo (USP)
Institute of Mathematics and Computing Sciences (ICMC)
SCC0251 - Image Processing 2023.1
Assignment 2: Fourier transform
Lucas Xavier Leite, USP number: 10783347

Task
----
In this assignment, your task is to match a set of given images using filters
in the frequency domain.
'''

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

DEBUG = False
PAD = False
GRAY_LEVELS = 256


def main():
    input_image_filename = input().strip()
    expected_image_filename = input().strip()
    filter_index = int(input())

    input_image = iio.imread(input_image_filename)
    expected_image = iio.imread(expected_image_filename)

    output_image = np.array(input_image.shape, dtype=np.float32)

    if filter_index in (0, 1):

        radius = int(input())

        _filter = 'ideal'
        selection = 'low-pass' if filter_index == 0 else 'high-pass'
        args = [radius]

    elif filter_index == 2:

        r0 = int(input())
        r1 = int(input())

        _filter = 'ideal'
        selection = 'band-pass'
        args = [r0, r1]

    elif filter_index == 3:

        _filter = 'laplacian'
        selection = 'high-pass'
        args = []

    elif filter_index == 4:

        sr = float(input())
        sc = float(input())

        _filter = 'gaussian'
        selection = 'low-pass'
        args = [(sr, sc)]

    elif filter_index in (5, 6):

        D0 = float(input())
        n = float(input())

        _filter = 'butterworth'
        selection = 'low-pass' if filter_index == 5 else 'high-pass'
        args = [(D0, n)]

    elif filter_index in (7, 8):

        D0 = float(input())
        D1 = float(input())

        n0 = float(input())
        n1 = float(input())

        _filter = 'butterworth'
        selection = 'band-reject' if filter_index == 7 else 'band-pass'
        args = [(D0, n0), (D1, n1)]

    output_image = apply_filter(input_image, _filter, selection, *args)

    print(f'{round(rmse(expected_image, output_image), 4):.4f}')

    if DEBUG is True:
        plt.subplot(131)
        plt.axis('off')
        plt.title('input')
        plt.imshow(input_image, cmap='gray')

        plt.subplot(132)
        plt.axis('off')
        plt.title('output')
        plt.imshow(output_image, cmap='gray')

        plt.subplot(133)
        plt.axis('off')
        plt.title('expected output')
        plt.imshow(expected_image, cmap='gray')

        plt.show()


def crop_center(image, region):
    '''
    Crop a specific region from the center of an image.

    Parameters
    ----------
    image : array-like
        The image to be cropped.
    region : tuple
        Tuple of int values containing the two dimensions of the region.

    Returns
    -------
    array-like
        The cropped region.
    '''
    crop_y, crop_x = region

    y, x = image.shape

    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)

    return image[start_y:start_y + crop_y, start_x:start_x + crop_x]


def get_filter(shape, _filter, selection, *args):
    '''
    Obtain the specified filter in the frequency domain.

    Parameters
    ----------
    shape : tuple
        Tuple of int values containing the 2 dimensions of the filter.
    _filter : string
        The filter to be obtained. Possible values are:
            - `ideal`
            - `laplacian`
            - `gaussian`
            - `butterworth`
    selection : string
        The area to be selected. Possible values are:
            - `low-pass`
            - `high-pass`
            - `band-pass`
            - `band-reject`
    *args
        The parameters to be used in each combination of filter and selection:
            - `ideal`, `low-pass` or `high-pass` : radius
            - `ideal`, `band-pass` or `band-reject` : r0, r1
            - `laplacian`, any : none (no arguemnts required)
            - `gaussian`, `low-pass` or `high-pass` : (sr, sc)
            - `gaussian`, `band-pass` or `band-reject` : (sr0, sc0), (sr1, sc1)
            - `butterworth`, `low-pass` or `high-pass` : (D0, n)
            - `butterworth`, `band-pass` or `band-reject` : (D0, n0), (D1, n1)

    Returns
    -------
    ndarray
        The corresponding filter in the frequency domain.

    Notes
    -----
    For `band-pass` and `band-reject`, parameter values passed in `args` should
    be grouped together in tuples, i.g., for `butterworth`: (D0, n0), (D1, n1).

    Raises
    ------
    ValueError
        If either `_filter` or `selection` are not valid options.
    '''
    P, Q = shape

    H = np.zeros((P, Q), dtype=np.float32)

    if selection in ('high-pass', 'low-pass'):
        if _filter == 'ideal':
            radius = args[0]
        if _filter == 'gaussian':
            sr, sc = args[0]
        if _filter == 'butterworth':
            D0, n = args[0]

        for u in range(P):
            for v in range(Q):
                if _filter == 'ideal':

                    D_uv = np.sqrt((u - P / 2) ** 2 + (v - Q / 2) ** 2)
                    H[u, v] = (D_uv <= radius)

                elif _filter == 'laplacian':

                    k = -4 * np.pi ** 2
                    H[u, v] = k * ((u - P / 2) ** 2 + (v - Q / 2) ** 2)

                elif _filter == 'gaussian':

                    x1 = ((u - P / 2) ** 2) / (2 * sr ** 2)
                    x2 = ((v - Q/2) ** 2) / (2 * sc ** 2)
                    H[u, v] = np.exp(-(x1 + x2))

                elif _filter == 'butterworth':

                    D_uv = np.sqrt((u - P / 2) ** 2 + (v - Q / 2) ** 2)
                    H[u, v] = 1 / (1 + (D_uv / D0) ** (2 * n))

                else:
                    raise ValueError('Invalid filter argument')

    elif selection in ('band-pass', 'band-reject'):

        a0, a1 = args

        H0 = get_filter(shape, _filter, 'high-pass', a0)
        H1 = get_filter(shape, _filter, 'high-pass', a1)

        H = np.maximum(H0, H1) - np.minimum(H0, H1)

    else:
        raise ValueError('Invalid selection argument')

    # Complement
    if selection in ('high-pass', 'band-reject'):
        H = 1 - H

    return H


def apply_filter(image, _filter, selection, *args):
    '''
    Apply the specified filter to an image.

    Parameters
    ----------
    image : array-like
        The image to apply the filter.
    _filter : string
        The filter to be applied.
    selection : string
        The area to be selected.
    *args
        The parameters to be used in each combination of filter and selection.

    Returns
    -------
    array-like
        The processed image.

    Notes
    -----
    Refer to `get_filter` documentation for possible values of `_filter` and
    `selection`.

    See Also
    --------
    get_filter : obtain the specified filter in the frequency domain.
    '''
    M, N = image.shape

    if PAD is True:
        original_shape = image.shape
        image = np.pad(image, (M // 2, N // 2), 'constant', constant_values=0)
        padded_shape = image.shape

        if DEBUG is True:
            print(f'{original_shape} -> {padded_shape}')
            print(image)

    F = np.fft.fftshift(np.fft.fft2(image))
    H = get_filter(image.shape, _filter, selection, *args)
    G_shift = np.multiply(F, H)
    G = np.fft.ifftshift(G_shift)

    # Normalization
    g = np.fft.ifft2(G).real
    output = ((g - g.min()) * (GRAY_LEVELS - 1)) / (g.max() - g.min())

    if PAD is True:
        output = crop_center(output, (M, N))

        if DEBUG is True:
            print(f'{padded_shape} -> {output.shape}')

    if DEBUG is True:
        plt.subplot(151)
        plt.axis('off')
        plt.title('original image in spatial domain')
        plt.imshow(image, cmap='gray')

        plt.subplot(152)
        plt.axis('off')
        plt.title('original image in frequency domain')
        plt.imshow(np.log1p(np.abs(F)), cmap='gray')

        plt.subplot(153)
        plt.axis('off')
        plt.title(f'{_filter} {selection} {args}')
        plt.imshow(H, cmap='gray')

        plt.subplot(154)
        plt.axis('off')
        plt.title('filter applied in frequency domain')
        plt.imshow(np.log1p(np.abs(G_shift)), cmap='gray')

        plt.subplot(155)
        plt.axis('off')
        plt.title('result in spatial domain')
        plt.imshow(output, cmap='gray')

        plt.show()

    return output


def rmse(reference, image):
    '''
    Calculate the root mean squared error of two images.

    Parameters
    ----------
    reference : array-like
        The reference image.
    image : array-like
        The image to compare against the reference image.

    Returns
    -------
    float
        The root mean squared error.
    '''
    return np.sqrt(np.square(reference - image).mean())


if __name__ == '__main__':
    main()
