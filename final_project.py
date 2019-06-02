import numpy as np
import mnist
from numpy.linalg import svd
from functions import create_train_test_split, classify_images, assess_classification_results, visualize_eigenpatterns, classify_image, create_masks, test_performance, load_input_data
from imageio import imread
from matplotlib import pyplot as plt
import sys


def main():

    # Read the input file (given as a command line argument)
    # for the digits to use, their names, 
    # filenames and their classes
    input_file = sys.argv[1]
    digits, tags, filenames, image_classes = load_input_data(input_file)

    n_testimages = 500
    first_digit, first_digit_test, second_digit, second_digit_test = create_train_test_split(digits, n_testimages)

    # calculate singular value decompositions for digits zero and four
    u1, _s1, _v1 = svd(first_digit)
    u2, _s2, _v2 = svd(second_digit)

    # visualize eigenpatterns for digit zero
    visualize_eigenpatterns(u1, 3, tags[0])

    # visualize eigenpatterns for digit four
    visualize_eigenpatterns(u2, 3, tags[1])

    k = 20

    # use the zeros and fours from the testset to test the performance
    test_performance(first_digit_test, second_digit_test, digits, tags, n_testimages, u1, u2, k)

    # Classify the provded images
    classifications = classify_images(filenames, u1, u2, k, digits, image_classes)

    # Print out the classification results
    assess_classification_results(classifications, image_classes, filenames)

if __name__ == "__main__":
    main()
