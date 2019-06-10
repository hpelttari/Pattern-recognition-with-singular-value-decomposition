import numpy as np
from classification_functions import classify_image, calculate_residual, test_performance
from matplotlib import pyplot as plt


def invert_colors(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 0:
                image[i][j] = 1
            else:
                image[i][j] = 0
    return image


def find_optimal_k(images1, images2, digits, tags, n_images, u1, u2, step_size=1, max_k=50, start_k=1):
    k_list = list(range(start_k, max_k, step_size))
    accuracy = np.zeros(len(k_list))

    for i, k in enumerate(k_list):
        print(f"k = {k}")
        accuracy[i] = test_performance(images1, images2, digits, tags, n_images, u1, u2, k, k)

    plt.figure(1)
    plt.plot(k_list, accuracy)
    plt.title("Classification accuracy as a function fo cut-off parameter k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.savefig("cut_off.jpg")

    k = k_list[np.argmax(accuracy)]

    return k


def visualize_eigenpatterns(u, k, pattern):

    for i in range(k):
        plt.imshow(u[:, i].reshape((28, 28)), cmap='gray')
        plt.title(f"Eigenpattern {i+1} for digit {pattern}")
        plt.savefig(f"{pattern}{i}.jpg")
        plt.show()
