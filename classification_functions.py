from numpy.linalg import norm
from data_functions import read_images, flatten_images, get_class_mappings
import numpy as np


def calculate_residual(z, U):
    return norm((1 - U @ U.T)@z)


def classify_image(z, u1, u2, k1, k2, classes):

    r1 = calculate_residual(z, u1[:, 0:k1])
    r2 = calculate_residual(z, u2[:, 0:k2])

    if r1 < r2:
        return classes[0]
    if r2 < r1:
        return classes[1]

    return -1


def classify_images(filenames, u1, u2, k1, k2, training_classes, image_classes):

    z = read_images(filenames)
    z_flattened = flatten_images(z)

    classification_results = []
    for i, image in enumerate(z_flattened):
        classification_results.append(classify_image(image, u1, u2, k1, k2, training_classes))

    return classification_results


def assess_classification_results(classifications, image_classes, filenames):
    digit_mappings = get_class_mappings()
    for i, classification in enumerate(classifications):
        correct_class = image_classes[i]
        if classification == -1:
            print(f"{filenames[i]} could not be classified")
        elif classification == correct_class:
            print(f"{filenames[i]} correctly classified as a {digit_mappings[correct_class]}!")
        else:
            print(f"{filenames[i]} misclassified as a {digit_mappings[classification]}!")


def test_performance(digit_1_test, digit_2_test, classes, tags, n_testimages, u0, u4, k1, k2, print_results=False):

    results = np.zeros(n_testimages)
    for i, image in enumerate(digit_1_test):
        results[i] = classify_image(image, u0, u4, k1, k2, classes)

    correct1 = results == classes[0]
    if print_results:
        print(f"{np.sum(correct1)/n_testimages * 100}% correct for {tags[0]}s")

    # Test if fours are correctly classified
    results = np.zeros(n_testimages)
    for i, image in enumerate(digit_2_test):
        results[i] = classify_image(image, u0, u4, k1, k2, classes)

    correct2 = results == classes[1]
    if print_results:
        print(f"{np.sum(correct2)/n_testimages * 100}% correct for {tags[1]}s")

    return (np.sum(correct1)+np.sum(correct2))/(2*n_testimages)
