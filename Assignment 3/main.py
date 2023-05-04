#!/usr/bin/env python

'''
Image Processing course assignment 3.

University of São Paulo (USP)
Institute of Mathematics and Computing Sciences (ICMC)
SCC0251 - Image Processing 2023.1
Assignment 3: image descriptors
Lucas Xavier Leite, USP number: 10783347

Task
----
In this assignment you have to implement a classic feature extractor called
Histogram of Oriented Gradients (HOG). It was first introduced by Dalal and
Triggs for human detection in CCTV images. We will follow in their steps and
implement a solution for the same task using both the descriptor and the
Machine Learning algorithm K-nearest-neighbours.
'''

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy import ndimage

DEBUG = True
GRAY_LEVELS = 256

# Ignore division by zero warnings
np.seterr(divide='ignore', invalid='ignore')


def main():
    # Read input data
    X0 = input().split(' ')
    X1 = input().split(' ')
    X_test = input().split(' ')

    # Load training and test sets
    training_set_no_humans = [luminance(iio.imread(xi)) for xi in X0]
    training_set_humans = [luminance(iio.imread(xi)) for xi in X1]
    test_set = [luminance(iio.imread(xi)) for xi in X_test]

    # Display some example images
    if DEBUG is True:
        plt.subplot(131)
        plt.title('sample from training set X0')
        plt.axis('off')
        plt.imshow(training_set_no_humans[0], cmap='gray')

        plt.subplot(132)
        plt.title('sample from training set X1')
        plt.axis('off')
        plt.imshow(training_set_humans[0], cmap='gray')

        plt.subplot(133)
        plt.title('sample from test set X_test')
        plt.axis('off')
        plt.imshow(test_set[0], cmap='gray')

        plt.show()

    # Compute HOG descriptors for training and test sets
    desc_no_humans = [hog(image) for image in training_set_no_humans]
    desc_humans = [hog(image) for image in training_set_humans]

    # Classify test set images
    pred = [str(knn(hog(x), desc_no_humans, desc_humans)) for x in test_set]

    # Print predictions
    print(' '.join(pred))

    # Calculate the accuracy
    if DEBUG is True:
        true_classes = np.array([x.split('/')[1] for x in X_test])
        print(f'\nAccuracy: {accuracy(true_classes, pred) * 100}%')


def normalize_minmax(f, factor):
    '''
    Normalize an input array using min-max normalization.

    Parameters
    ----------
    f : array-like
        Input array to be normalized.
    factor : int
        Scale factor to be applied after normalization.

    Returns
    -------
    numpy.ndarray
        Normalized and scaled array.

    Examples
    --------
    >>> f = np.array([1, 2, 3, 4, 5])
    >>> normalize_minmax(f, 255)
    array([  0,  63, 127, 191, 255], dtype=uint8)
    '''
    f_min = np.min(f)
    f_max = np.max(f)

    f = (f - f_min) / (f_max - f_min)

    return (f * factor)


def luminance(image):
    '''
    Converts an RGB image to grayscale using the luminance method.

    Parameters
    ----------
    image : numpy.ndarray
        The input RGB image.

    Returns
    -------
    numpy.ndarray
        The resulting grayscale image.
    '''
    M, N, _ = image.shape

    image_copy = image.copy().astype(np.float32)

    R = image_copy[:, :, 0]
    G = image_copy[:, :, 1]
    B = image_copy[:, :, 2]

    output = np.zeros((M, N), dtype=np.float32)
    output = np.floor(R * 0.299 + G * 0.587 + B * 0.144)

    return normalize_minmax(output, GRAY_LEVELS - 1)


def hog(image, nbins=9, bin_size=20, operator='sobel'):
    '''
    Calculate the Histogram of Oriented Gradients (HOG) descriptor of an image.

    Parameters
    ----------
    image : array-like
        Input image.
    nbins : int, optional
        Number of orientation bins.
    bin_size : int, optional
        Size of each orientation bin, in degrees.
    operator : str, optional
        Type of operator kernel to use. Valid values are 'sobel', 'scharr', and
        'prewitt'.

    Returns
    -------
    numpy.ndarray
        HOG descriptor of the input image.

    Raises
    ------
    ValueError
        If the `operator` parameter is not valid.
    '''
    # Define the operator kernels (Sobel, Scharr or Prewitt)
    kernel_x = np.zeros((3, 3), dtype=np.int32)

    if operator == 'sobel':
        kernel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif operator == 'scharr':
        kernel_x = np.array([[-3, -10, 3], [-10, 0, 10], [-3, 0, 3]])
    elif operator == 'prewitt':
        kernel_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    else:
        raise ValueError('Operator not valid.')

    kernel_y = kernel_x.T

    # Obtain gradients by convolving the image with the Sobel kernels
    gradient_x = ndimage.convolve(image, kernel_x)
    gradient_y = ndimage.convolve(image, kernel_y)

    # Obtain magnitude and angle
    magnitude = np.hypot(gradient_x, gradient_y)
    magnitude /= magnitude.sum()
    angle = np.degrees(np.arctan(gradient_y / gradient_x) + np.pi / 2)

    descriptor = np.zeros(nbins, dtype=np.float32)

    for i in range(nbins):
        # Create a mask selecting the pixels with an angle in the current bin
        # and compute the accumulated magnitude.
        bin_mask = (angle >= (i * bin_size)) & (angle < bin_size * (i + 1))
        descriptor[i] = magnitude[bin_mask].sum()

        # Mask for 9 bins of 20 degrees:
        # 0 * 20 = 0 <= angle < 20 = 20 * (0 + 1)
        # 1 * 20 = 20 <= angle < 40 = 20 * (1 + 1)
        # 2 * 20 = 40 <= angle < 60 = 20 * (2 + 1)
        # 3 * 20 = 60 <= angle < 80 = 20 * (3 + 1)
        # ...
        # 8 * 20 = 160 <= angle < 180 = 20 * (8 + 1)

    if DEBUG is True:
        plt.subplot(141)
        plt.title('input')
        plt.axis('off')
        plt.imshow(image, cmap='gray')

        plt.subplot(142)
        plt.title('magnitude')
        plt.axis('off')
        plt.imshow(magnitude, cmap='gray')

        plt.subplot(143)
        plt.title('angle')
        plt.axis('off')
        plt.imshow(angle, cmap='gray')

        plt.subplot(144)
        plt.title('descriptor')
        plt.xticks(np.arange(nbins))
        plt.xlabel('bins')
        plt.ylabel('acc. magnitude')
        plt.bar(np.arange(nbins), descriptor)

        plt.show()

        print(descriptor)

    return descriptor


def euclidean_distance(a, b):
    '''
    Calculate the Euclidean distance between two points in n-dimensional space.

    Parameters:
    -----------
    a : array-like
        The first point in n-dimensional space.
    b : array-like
        The second point in n-dimensional space.

    Returns:
    --------
    float
        The Euclidean distance between points a and b.

    Examples:
    ---------
    >>> euclidean_distance([0,0], [3,4])
    5.0
    >>> euclidean_distance([0,0,0], [1,1,1])
    1.7320508075688772
    '''
    return np.linalg.norm(a - b)


def knn(input_value, class_0_set, class_1_set, dist=euclidean_distance, k=3):
    '''
    Classifies an input value into one of two classes using k-nearest neighbors
    algorithm.

    Parameters:
    -----------
    input_value : array_like
        Input value to be classified.
    class_0_set : array_like
        Set of input values in class 0.
    class_1_set : array_like
        Set of input values in class 1.
    dist : function, optional
        Function that computes distance between two points. Default is
        Euclidean distance.
    k : int, optional
        Number of neighbors to consider. Default is 3.

    Returns:
    --------
    int
        The predicted class label (0 or 1).
    '''
    # Create an array for each class. The second column contains the class.
    class_0 = np.zeros((len(class_0_set), 2))
    class_1 = np.ones((len(class_1_set), 2))

    # Compute the distance between the input value and all values in each
    # training set, storing the results in the first column.
    class_0[:, 0] = [dist(input_value, i) for i in class_0_set]
    class_1[:, 0] = [dist(input_value, i) for i in class_1_set]

    # Concatenate the two arrays and sort by distance values
    classes = np.concatenate((class_0, class_1), axis=0)
    classes = classes[classes[:, 0].argsort()]

    # Count the number of votes for class 1 from the k-nearest neighbors
    votes_for_class_1 = int(classes[:k, 1].sum())

    if DEBUG is True:
        print(classes)
        print(f'\nVotes for class 0: {k - votes_for_class_1}')
        print(f'Votes for class 1: {votes_for_class_1}\n')

    # Return the predicted class label based on the majority vote
    return int(votes_for_class_1 > k // 2)


def accuracy(true_classes, predicted_classes):
    '''
    Calculate the accuracy score of predicted classes.

    Parameters
    ----------
    true_classes : array-like
        True target classes.

    predicted_classes : array-like
        Predicted target classes.

    Returns
    -------
    float
        The accuracy score.

    Examples
    --------
    >>> true_classes = [0, 1, 1, 0, 1]
    >>> predicted_classes = [0, 1, 0, 0, 1]
    >>> accuracy(true_classes, predicted_classes)
    0.6
    '''
    return np.sum(true_classes == predicted_classes) / len(true_classes)


if __name__ == '__main__':
    main()
