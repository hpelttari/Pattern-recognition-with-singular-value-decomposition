import mnist
from matplotlib import pyplot as plt


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
    # load mnist data
    x_train, y_train, x_test, y_test = mnist.load()

    # pick the images of the first digit from training and test data
    d1_train = pick_digit_from_data(x_train, y_train, digits[0])
    d1_test = pick_digit_from_data(x_test, y_test, digits[0])[0:n_testimages]

    # pick the images of the second digit from training and test data
    d2_train = pick_digit_from_data(x_train, y_train, digits[1])
    d2_test = pick_digit_from_data(x_test, y_test, digits[1])[0:n_testimages]

    # Transpose the data if parameter are set to True
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
