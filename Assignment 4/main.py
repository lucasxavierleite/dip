#!/usr/bin/env python

'''
Image Processing course assignment 4.

University of São Paulo (USP)
Institute of Mathematics and Computing Sciences (ICMC)
SCC0251 - Image Processing 2023.1
Assignment 4: mathematical morphology
Lucas Xavier Leite, USP number: 10783347

Task
----
The objective of this task is to create and execute the Flood Fill Algorithm
for the purpose of painting a specific region and detecting connected
components within an image. The algorithm must be capable of producing the
pixels belonging to the region’s connected components as output.
'''

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

DEBUG = False


def main():
    filename = input().strip()
    seed_x = int(input())
    seed_y = int(input())
    c = int(input())

    # Read and binarize the image, then apply the flood fill algorithm
    image = (iio.imread(filename) > 127).astype(np.uint8)
    filled_image, area = flood_fill(image, (seed_x, seed_y), c)

    # Print the filled area coordinates
    print(' '.join([f'({i} {j})' for i, j in area]), ' \n')

    # Display the original and filled images for debugging
    if DEBUG is True:
        plt.subplot(121)
        plt.title('input')
        plt.axis('off')
        plt.imshow(image, cmap='gray')

        plt.subplot(122)
        plt.title('output')
        plt.axis('off')
        plt.imshow(filled_image, cmap='gray')

        plt.show()


def flood_fill(image, seed, c):
    '''
    Apply the flood fill algorithm to the given image.

    Parameters
    ----------
    image : array-like
        Input binary image.
    seed : tuple
        Seed coordinates (x, y) for flood fill.
    c : int
        Connectivity type, either 4 or 8.

    Returns
    -------
    filled_image : numpy.ndarray
        Image with the filled region.
    area : list
        List of pixel coordinates belonging to the region.
    '''
    assert c in [4, 8]

    # Copy the input image and set fill color
    fill = np.copy(image)
    seed_x, seed_y = seed
    fill_color = int((fill[seed] == 0))

    # Print seed and its color for debugging
    if DEBUG is True:
        print('Original image:\n')
        print(image)
        print(f'\nSeed is {seed} = {fill[seed]}, painting with {fill_color}\n')

    # Initialize list of coordinates belonging to the filled region
    area = []

    # Call recursive flood fill algorithm
    _flood_fill(image, seed_y, seed_x, c, fill_color, area)

    # Print filled image for debugging
    if DEBUG is True:
        print('\nFilled image:\n')
        print(fill, '\n\n')

    # Sort pixel coordinates first by x and then by y
    area = sorted(area)

    return fill, area


def _flood_fill(image, x, y, c, color, area):
    '''
    Recursive function to perform flood fill algorithm.

    Parameters
    ----------
    image : numpy.ndarray
        Input binary image.
    x : int
        x-coordinate of the current pixel.
    y : int
        y-coordinate of the current pixel.
    c : int
        Connectivity type, either 4 or 8.
    color : int
        Color to fill the region with.
    area : list
        List of pixel coordinates belonging to the region.
    '''
    # Stop at the borders
    if image[y, x] == color:
        if DEBUG is True:
            print(f'Stopping at image[{(x, y)}] = {image[x, y]}')
        return

    if DEBUG is True:
        print(f'Visiting {(x, y)}')

    # Paint the pixel and append its coordinates to the list
    image[y, x] = color
    area.append((y, x))

    # Recursive calls for 4-neighborhood
    _flood_fill(image, x+1, y, c, color, area)
    _flood_fill(image, x-1, y, c, color, area)
    _flood_fill(image, x, y+1, c, color, area)
    _flood_fill(image, x, y-1, c, color, area)

    if c == 8:
        # Recursive calls for 8-neighborhood
        _flood_fill(image, x+1, y+1, c, color, area)
        _flood_fill(image, x+1, y-1, c, color, area)
        _flood_fill(image, x-1, y+1, c, color, area)
        _flood_fill(image, x-1, y-1, c, color, area)


if __name__ == '__main__':
    main()
