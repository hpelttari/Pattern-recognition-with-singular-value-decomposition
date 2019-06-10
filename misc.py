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


def find_optimum_k(images1, images2, labels, classes, u0, u4, k_list):

    n_correct1 = []
    n_correct2 = []

    for k in k_list:

        results = np.zeros(len(images1))

        for i, z in enumerate(images1):
            results[i] = classify_image(z, u0, u4, k, classes)

        correct = results == labels[0]
        n_correct1.append(np.sum(correct))

    for k in k_list:

        results = np.zeros(len(images2))

        for i, z in enumerate(images2):
            s = classify_image(z, u0, u4, k, classes)
            results[i] = s
        correct = results == labels[1]
        n_correct2.append(np.sum(correct))
    n_correct = n_correct1 + n_correct2
    print(n_correct1)
    print(n_correct2)
    print(n_correct)
    k_index = np.argmax(n_correct)

    return k_list[k_index]


def optimum_k(images1, images2, u1, u2):
    k_list = [1, 3, 5, 10, 15, 20, 25, 30, 50, 60, 70]
    residual_sums1 = np.zeros(len(k_list))
    residual_sums2 = np.zeros(len(k_list))
    for i, k in enumerate(k_list):
        print(f"K = {k}")
        residual_sum1 = 0

        for image in images1:
              residual_sum1 += calculate_residual(image, u1[:, :k])
        residual_sums1[i] = residual_sum1

        residual_sum2 = 0
        for image in images2:
              residual_sum2 += calculate_residual(image, u2[:, :k])
        residual_sums2[i] = residual_sum2
    plt.figure(1)
    plt.plot(k_list, residual_sums1)
    plt.title("Sum of residuals as a function of cut-off parameter k")
    plt.xlabel("k")
    plt.ylabel("Sum of residuals")
    plt.show()
    print(f"last residual = {residual_sums1[-1]}")

    plt.figure(2)
    plt.plot(k_list, residual_sums2)
    plt.title("Sum of residuals as a function of cut-off parameter k")
    plt.xlabel("k")
    plt.ylabel("Sum of residuals")
    plt.show()


    plt.figure(3)
    plt.plot(k_list, residual_sums2+residual_sums1)
    plt.title("Sum of residuals as a function of cut-off parameter k")
    plt.xlabel("k")
    plt.ylabel("Sum of residuals")
    plt.show()

    k1 = k_list[np.argmin(residual_sums1)]
    k2 = k_list[np.argmin(residual_sums2)]
    k3 = k_list[np.argmin(residual_sums1+residual_sums2)]
    return k1, k2, k3

def find_k(images1, images2, digits, tags, n_images, u1, u2, step_size=1, max_k=50, start_k=1):
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
