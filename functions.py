import numpy as np
import mnist
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from numpy.linalg import svd, norm


def calculate_residual(z, U):
    return norm((1 - U @ U.T)@z)


def find_optimum_k(z, u0, u4):
    r = []
    for k in range(2, u0.shape[1]):
        r0 = calculate_residual(z, u0[:, 0:k])
        r4 = calculate_residual(z, u4[:, 0:k])
        r.append(np.abs(r0-r4))
    return np.argmax(r)+2


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
