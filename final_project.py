import sys
from numpy.linalg import svd
from data_functions import load_input_data, create_train_test_split
from misc import visualize_eigenpatterns, optimum_k, find_k
from classification_functions import test_performance, classify_images, assess_classification_results


def main():

    # Check if the correct number of command line arguments are given
    if len(sys.argv) < 2:
        raise Exception("You must give an input file as a command line argument!")
    elif len(sys.argv) > 2:
        raise Exception("Too many command line arguments given!")

    # Read the input file (given as a command line argument)
    # for the digits to use, their names,
    # filenames and their classes
    print("Reading input file...")
    input_file = sys.argv[1]

    digits, tags, filenames, image_classes = load_input_data(input_file)

    print("Creating train-test split...")
    n_testimages = 500
    first_digit, first_digit_test, second_digit, second_digit_test = create_train_test_split(digits, n_testimages)

    # Calculate singular value decompositions for digits zero and four
    print("Calculating singular value decomposition...")
    u1, _s1, _v1 = svd(first_digit)
    u2, _s2, _v2 = svd(second_digit)

    # Visualize eigenpatterns for digit zero
    print("Visualizing eigenpattterns...")
    visualize_eigenpatterns(u1, 3, tags[0])

    # Visualize eigenpatterns for digit four
    visualize_eigenpatterns(u2, 3, tags[1])

    # Find the optimal cut-off parameter
    print("Finding optimal cut-off parameter...")
    k = find_k(first_digit_test, second_digit_test, digits, tags, n_testimages, u1, u2)

    print(f"Optimal cut-off parameter k is {k}")

    # Use the zeros and fours from the testset to test the performance
    print("Testing performance...")
    test_performance(first_digit_test, second_digit_test, digits, tags, n_testimages, u1, u2, k, k, print_results=True)

    # Classify the provded images
    print("Classifying images...")
    classifications = classify_images(filenames, u1, u2, k, k, digits, image_classes)

    # Print out the classification results
    assess_classification_results(classifications, image_classes, filenames)


if __name__ == "__main__":
    main()
