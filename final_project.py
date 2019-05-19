import numpy as np
import mnist
from matplotlib import pyplot as plt
from numpy.linalg import svd
from functions import visualize_eigenpatterns, classify_image, create_masks, invert_colors
from PIL import Image

def main():
    x_train, y_train, x_test, y_test = mnist.load()

    # create masks for picking the images with right digits
    masks_train = create_masks(y_train)

    # create arrays of images
    zeros = x_train[masks_train[0]].T
    fours = x_train[masks_train[4]].T

    # masks for a few test images
    masks_test = create_masks(y_test)
    z1 = plt.imread("zero1.png")
    z2 = plt.imread("zero2.png")
    z3 = plt.imread("four.png")
    plt.imshow(z3, cmap='gray')
    plt.show()

    n_testimages = 500
    # create arrays of test  images
    zeros_test = x_test[masks_test[0]][0:n_testimages]
    fours_test = x_test[masks_test[4]][0:n_testimages]

    # calculate singular value decompositions for digits zero and four
    u0, _s0, _v0 = svd(zeros)
    u4, _s4, _v4 = svd(fours)

    # visualize eigenpatterns for digit zero
    visualize_eigenpatterns(u0, 3, "zero")

    # visualize eigenpatterns for digit four
    visualize_eigenpatterns(u4, 3, "four")

    k = 20

    # Test if zeros are correctly classified
    results = np.zeros(n_testimages)
    for i, zero in enumerate(zeros_test):
        results[i] = classify_image(zero, u0, u4, "zero", "four", k)

    correct = results == 0
    print(f"{np.sum(correct)/n_testimages * 100}% correct for zeros")

    # Test if fours are correctly classified
    results = np.zeros(n_testimages)
    for i, four in enumerate(fours_test):
        results[i] = classify_image(four, u0, u4, "zero", "four", k)

    correct = results == 1
    print(f"{np.sum(correct)/n_testimages * 100}% correct for fours")

    print(classify_image(z1, u0, u4, "zero", "four", k))
    print(classify_image(z2, u0, u4, "zero", "four", k))
    print(classify_image(z3, u0, u4, "zero", "four", k))


if __name__ == "__main__":
    main()
