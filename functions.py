import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import mnist


def load_input_data(input_filename):

    """
    Reads the content of the input file to variables.

    The firt line of the input file should contain the two integers
    which the images are wanted to classify as, separated by spaces.

    The second line should contain the  names of the digits,
    for example "zero", separated by spaces.

    The third line should contain the filenames of the images that will
    be classified, separated by spaces.

    The fourth and final line should contain, as integers, the whether
    the images belong to the first class, given in the first line, or
    the second class. Denote first class with 0, second class with 1,
    and separate withspaces.

    Returns: lists of the digits, names of digits, filenames and clases
    """

    with open(input_filename, "r") as input_file:
        for i, line in enumerate(input_file):
            if i == 0:
                digits = map(int, line.split())
            elif i == 1:
                tags = line.split()
            elif i == 2:
                filenames = line.split()
            elif i == 3:
                image_classes = map(int, line.split())

    return list(digits), tags, filenames, list(image_classes)


def pick_digit_from_data(x, y, digit):

    mask = create_masks(y)

    return x[mask[digit]]

def create_train_test_split(digits, n_testimages, transpose_trainig=True, transpose_test=False):

    x_train, y_train, x_test, y_test = mnist.load()

    d1_train = pick_digit_from_data(x_train, y_train, digits[0])
    d1_test = pick_digit_from_data(x_test, y_test, digits[0])[0:n_testimages]


    d2_train = pick_digit_from_data(x_train, y_train, digits[1])
    d2_test = pick_digit_from_data(x_test, y_test, digits[1])[0:n_testimages]

    if transpose_trainig:
        d1_train = d1_train.T
        d2_train = d2_train.T

    if transpose_test:
        d1_test = d1_test.T
        d2_test = d2_test.T

    return d1_train, d1_test, d2_train, d2_test


def get_class_mappings():
    digit_names = "zero one two three four five six seven eight nine ten".split()
    mappings = {}

    for i in range(11):
        mappings[i] = digit_names[i]

    return mappings


def read_images(filenames):
    z = []
    for file in filenames:
        image = plt.imread(file)
        z.append(image)
    return z


def flatten_images(z):
    for i, image in enumerate(z):
        z[i] = image.flatten()

    return z


def classify_images(filenames, u1, u2, k, training_classes, image_classes):

    z = read_images(filenames)
    z_flattened = flatten_images(z)

    classification_results = []
    for i, image in enumerate(z_flattened):
        classification_results.append(classify_image(image, u1, u2, k, training_classes))

    return classification_results


def assess_classification_results(classifications, image_classes, filenames):
    digit_mappings = get_class_mappings()
    for i, classification in enumerate(classifications):
        correct_class = image_classes[i]
        if classification == -1:
            print(f"{filename[i]} could not be classified")
        elif classification == correct_class:
            print(f"{filenames[i]} correctly classified as a {digit_mappings[correct_class]}!")
        else:
            print(f"{filenames[i]} misclassified as a {digit_mappings[classification]}!")

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
        plt.savefig(f"{pattern}{i}.jpg")
        plt.show()


def classify_image(z, u1, u2, k, classes):

    r1 = calculate_residual(z, u1[:, 0:k])
    r2 = calculate_residual(z, u2[:, 0:k])

    if r1 < r2:
        return classes[0]
    if r2 < r1:
        return classes[1]

    return -1


def test_performance(digit_1_test, digit_2_test, classes, tags, n_testimages, u0, u4, k):

    results = np.zeros(n_testimages)
    for i, image in enumerate(digit_1_test):
        results[i] = classify_image(image, u0, u4, k, classes)

    correct = results == classes[0]
    print(f"{np.sum(correct)/n_testimages * 100}% correct for {tags[0]}s")

    # Test if fours are correctly classified
    results = np.zeros(n_testimages)
    for i, image in enumerate(digit_2_test):
        results[i] = classify_image(image, u0, u4, k, classes)

    correct = results == classes[1]
    print(f"{np.sum(correct)/n_testimages * 100}% correct for {tags[1]}s")
