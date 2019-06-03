import numpy as np
from classification_functions import classify_image
from matplotlib import pyplot as plt

def invert_colors(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 0:
                image[i][j] = 1
            else:
                image[i][j] = 0
    return image


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
