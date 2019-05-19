import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm


def create_masks(labels):

    """
    Creates a dictionary with masks for positions
    of each digit in the matrix containing all images.

    Parameters: numpy array of labelss

    To return a mask for a desired digit, use the digit
    as a key to the dictionary.

    Example: masks[4] returns mask for digit 4.
    """

    masks = {}

    for i in range(10):
        masks[i] = labels == i

    return masks


def invert_colors(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 0:
                image[i][j] = 1
            else:
                image[i][j] = 0
    return image


def calculate_residual(z, U):
    return norm((1 - U @ U.T)@z)


def find_optimum_k(images1, images2, labels, u0, u4, k_list):

    n_correct1 = []
    n_correct2 = []

    for k in k_list:

        results = np.zeros(len(images1))

        for i, z in enumerate(images1):
            results[i] = classify_image(z, u0, u4, "zero", "four", k)

        correct = results == labels[0]
        n_correct1.append(np.sum(correct))

    for k in k_list:

        results = np.zeros(len(images2))

        for i, z in enumerate(images2):
            s = classify_image(z, u0, u4, "zero", "four", k)
            print(s)
            results[i] = s
        correct = results == labels[1]
        n_correct2.append(np.sum(correct))
    n_correct = n_correct1 + n_correct2
    print(n_correct1)
    print(n_correct2)
    print(n_correct)
    k_index = np.argmax(n_correct)

    return k_list[k_index]


def visualize_eigenpatterns(u, k, pattern):

    for i in range(k):
        plt.imshow(u[:, i].reshape((28, 28)), cmap='gray')
        plt.title(f"Eigenpattern {i+1} for digit {pattern}")
        plt.show()


def classify_image(z, u1, u2, pattern1, pattern2, k):

    r1 = calculate_residual(z, u1[:, 0:k])
    r2 = calculate_residual(z, u2[:, 0:k])
    if r1 < r2:
        #print(f"Image is digit {pattern1}!")
        return 0
    elif r2 < r1:
        #print(f"Image is digit {pattern2}!")
        return 1

    print("Can't classify image")
    return -1
