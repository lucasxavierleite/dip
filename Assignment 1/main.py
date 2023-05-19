#!/usr/bin/env python

'''
Image Processing course assignment 1.

University of São Paulo (USP)
Institute of Mathematics and Computing Sciences (ICMC)
SCC0251 - Image Processing 2023.1
Assignment 1: enhancement and super-resolution
Lucas Xavier Leite, USP number: 10783347

Task
----
In this assignment you have to implement 3 distinct image enhancement
techniques, as well as a super-resolution method based on multiple views of the
same image.
'''

import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

DEBUG = False
GRAY_LEVELS = 256


def main():
    filename = input()
    ref_image = input()
    option = int(input())
    gamma = float(input())

    images = np.array([iio.imread(file)
                       for file in sorted(glob.glob(f'{filename}*.png'))])

    if option == 1:
        images = single_histogram_equalization(images)
    elif option == 2:
        images = joint_histogram_equalization(images)
    elif option == 3:
        images = gamma_correction(images, gamma)

    H_hat = super_resolution(images)
    H_ref = iio.imread(ref_image)

    print(f'{round(rmse(H_ref, H_hat), 4):.4f}')


def histogram(image, levels):
    '''
    Compute an image histogram.

    Parameters
    ----------
    image : array-like
        The image from which to obtain the histogram.
    levels : int
        Number of gray levels to consider when counting frequencies.

    Returns
    -------
    numpy.ndarray
        The histogram of ``image``.
    '''
    h = np.zeros(levels).astype(int)

    for i in range(levels):
        h[i] = np.sum(image == i)

    return h


def cumulative_histogram(image, levels):
    '''
    Compute a cumulative image histogram.

    Parameters
    ----------
    image : array-like
        The image from which to obtain the cumulative histogram.
    levels : int
        Number of gray levels to consider when counting frequencies.

    Returns
    -------
    numpy.ndarray
        The cumulative histogram of ``image``.
    '''
    h = histogram(image, levels)

    hc = np.zeros(levels).astype(int)
    hc[0] = h[0]

    for i in range(1, levels):
        hc[i] = h[i] + hc[i-1]

    return hc


def joint_cumulative_histogram(images, levels):
    '''
    Compute a joint cumulative image histogram using the images mean values.

    Parameters
    ----------
    images : array-like
        The images from which to obtain the joint cumulative histogram.
    levels : int
        Number of gray levels to consider when counting frequencies.

    Returns
    -------
    numpy.ndarray
        The joint cumulative histogram of ``images``.
    '''
    jh = np.mean([histogram(img, levels) for img in images], axis=0)

    if DEBUG is True:
        print([histogram(i, levels) for i in images])
        print(jh)

    jhc = np.zeros(levels).astype(int)
    jhc[0] = jh[0]

    for i in range(1, levels):
        jhc[i] = jh[i] + jhc[i-1]

    return jhc


def histogram_equalization(image, levels, cumulative_histogram):
    '''
    Equalize an image histogram.

    Parameters
    ----------
    image : array-like
        The image whose histogram to equalize.
    levels : int
        Number of gray levels to consider when counting frequencies.
    cumulative_histogram : array-like
        Cumulative histogram of ``image``.

    Returns
    -------
    s : numpy.ndarray
        The equalized histogram of ``image``.
    T : numpy.ndarray
        The transformation applied.
    '''
    s = np.zeros(image.shape).astype(np.uint8)
    T = np.zeros(levels).astype(np.uint8)

    N, M = image.shape

    for z in range(levels):
        si = ((levels - 1) / float(M * N)) * cumulative_histogram[z]
        s[np.where(image == z)] = si
        T[z] = si

    return (s, T)


def single_histogram_equalization(images):
    '''
    Perform histogram equalization on a set of images.

    Parameters
    ----------
    images : array-like
        The images to be enhanced using histogram equalization.

    Returns
    -------
    numpy.ndarray
        The enhanced images.
    '''
    if DEBUG is True:
        fig, ax = plt.subplots(images.shape[0], 5, figsize=(3, 3))

    images_eq = np.zeros(images.shape)

    for i, img in enumerate(images):
        hc = cumulative_histogram(img, GRAY_LEVELS)
        img_eq, T = histogram_equalization(img, GRAY_LEVELS, hc)
        images_eq[i] = img_eq

        if DEBUG is True:
            print(f'\n{i}:\n')
            print(img_eq)
            print_debug(ax, i, img, img_eq, T)

    if DEBUG is True:
        plt.show()

    return images_eq


def joint_histogram_equalization(images):
    '''
    Perform histogram equalization on a set of images, using a joint histogram.

    The same joint cumulative histogram is used to equalize the whole set of
    image histograms, instead of using each image's own histogram.

    Parameters
    ----------
    images : array-like
        The images to be enhanced.

    Returns
    -------
    numpy.ndarray
        The enhanced images.
    '''
    if DEBUG is True:
        fig, ax = plt.subplots(images.shape[0], 5, figsize=(3, 3))

    jhc = joint_cumulative_histogram(images, GRAY_LEVELS)

    images_eq = np.zeros(images.shape)

    for i, img in enumerate(images):
        img_eq, T = histogram_equalization(img, GRAY_LEVELS, jhc)
        images_eq[i] = img_eq

        if DEBUG is True:
            print(f'\n{i}:\n')
            print(img_eq)
            print_debug(ax, i, img, img_eq, T)

    if DEBUG is True:
        plt.show()

    return images_eq


def gamma_correction_single(image, gamma):
    '''
    Perform gamma correction on a single image.

    Parameters
    ----------
    image : array-like
        The image to be enhanced.
    gamma : float
        The gamma correction factor.

    Returns
    -------
    numpy.ndarray
        The enhanced image.
    '''
    output = image.copy()
    max_value = float(GRAY_LEVELS - 1)

    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            output[x][y] = (max_value * ((output[x][y] / max_value)
                                         ** (1 / gamma))).astype(np.uint8)

    return output


def gamma_correction(images, gamma):
    '''
    Perform gamma correction on a set of images.

    Parameters
    ----------
    images : array-like
        The images to be enhanced.
    gamma : float
        The gamma correction factor.

    Returns
    -------
    numpy.ndarray
        The enhanced image.
    '''
    if DEBUG is True:
        fig, ax = plt.subplots(images.shape[0], 4, figsize=(3, 3))

    G = np.array([gamma_correction_single(img, gamma) for img in images])

    if DEBUG is True:
        for i, (img, gi) in enumerate(zip(images, G)):
            print(f'\nOriginal ({i}):\n')
            print(img)
            print(f'\nEqualized ({i}):\n')
            print(gi)
            print_debug(ax, i, img, gi)

        plt.show()

    return G


def super_resolution(images):
    '''
    Use a set of low resolution images to compose one of higher resolution.

    Parameters
    ----------
    images : array-like
        The set of low resolution images.

    Returns
    -------
    numpy.ndarray
        The higher resolution image.
    '''
    assert images.shape[0] == 4

    N, M = images[0].shape

    H = np.zeros((N * 2, M * 2))

    H[::2, ::2] = images[0]
    H[::2, 1::2] = images[1]
    H[1::2, ::2] = images[2]
    H[1::2, 1::2] = images[3]

    if DEBUG is True:
        print(f'{images[0].shape} -> {H.shape}\n')
        print('L[0]:\n')
        print(images[0])

        print('\nL[1]:\n')
        print(images[1])

        print('\nL[2]:\n')
        print(images[2])

        print('\nL[3]:\n')
        print(images[3])

        print('\nH:\n')
        print(H)

        plt.imshow(images[0], cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(H, cmap='gray')
        plt.axis('off')
        plt.show()

    return H


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


def print_debug(ax, row, original_image, enhanced_image, transformation=None):
    '''
    Print info and plot graphs for debugging.

    This function considers a fixed number of columns (3 or 4) for plotting,
    and plots all graphs in a single row, i.e., this function should be called
    once for each row. After calling this function for the last row,
    matplotlib.pyplot.show should be called.

    The plots include both the original input image and the enhanced image, as
    well as their histograms and the transformation function - when using the
    `transformation` optional parameter.

    Parameters
    ----------
    ax : array-like
        2D array of `matplotlib.axes.Axes` used for plotting. The number of
        columns may only be 3 or 4 (to include the plot of `transformation`).
    row : int
        Subplot row number.
    original_image : array-like
        The original input image, without any enhancement.
    enhanced_image : array-like
        The enhanced image.
    transformation : array-like
        The transformation function applied to the image.
    '''
    h = histogram(original_image, GRAY_LEVELS)
    heq = histogram(enhanced_image, GRAY_LEVELS)

    ax[row][0].imshow(original_image, cmap='gray')
    ax[row][0].axis('off')

    ax[row][1].bar(range(GRAY_LEVELS), h)
    ax[row][1].set_xlabel('Graylevel / intensity')
    ax[row][1].set_ylabel('Frequency')

    ax[row][2].imshow(enhanced_image, cmap='gray')
    ax[row][2].axis('off')

    ax[row][3].bar(range(GRAY_LEVELS), heq)
    ax[row][3].set_xlabel('Graylevel / intensity')
    ax[row][3].set_ylabel('Frequency')

    if transformation is not None:
        ax[row][4].plot(range(GRAY_LEVELS), transformation)
        ax[row][4].set_xlabel('Input pixel value')
        ax[row][4].set_ylabel('Output pixel value')


if __name__ == '__main__':
    main()
