import numpy as np
import mnist
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from numpy.linalg import svd, norm
from functions import calculate_residual, visualize_eigenpatterns, classify_image



def main():
    x_train, y_train, x_test, y_test = mnist.load()

    # create masks for picking the images with right digits
    zeros_mask = y_train == 0
    fours_mask = y_train == 4

    # create arrays of images
    zeros = x_train[zeros_mask].T
    fours = x_train[fours_mask].T

    # masks for a few test images
    zeros_mask = y_test == 0
    fours_mask = y_test == 4

    n_testimages = 500
    # create arrays of test  images
    zeros_test = x_test[zeros_mask][0:n_testimages]
    fours_test = x_test[fours_mask][0:n_testimages]

    # calculate singular value decompositions for digits zero and four
    u0, _s0, _v0 = svd(zeros)
    u4, _s4, _v4 = svd(fours)
    
    # visualize eigenpatterns for digit zero
    visualize_eigenpatterns(u0, 2, "zero")
    
    # visualize eigenpatterns for digit four
    visualize_eigenpatterns(u4, 2, "four")
    
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
    

if __name__ == "__main__":
    main()
