import numpy as np
import math
import matplotlib.pyplot as plt


def plot(X, y):
    # Plot the dataset X and the corresponding labels y
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.show()


def euclidean_distance(x1,x2):
    #TODO
    distance = 0
    x1 = list(x1)
    x2 = list(x2)
    #calculates l2 distance between two vectors
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2
    distance = math.sqrt(distance)
    return distance
