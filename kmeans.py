import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def loadData():
    df = pd.read_csv("./data/data.csv")
    return df.values


def loadBirch():
    df = pd.read_table("./data/birch.txt", header=None, delim_whitespace=True)
    print(df.values)
    return df.values


def euclideanDistance(vector1, vector2):
    return math.sqrt(sum(np.power(vector1 - vector2, 2)))


def initRandomCentroids(data, k):
    count, dim = data.shape
    centroids = np.zeros((k, dim))
    colMax = np.max(data, axis=0)
    colMin = np.min(data, axis=0)
    colRange = colMax - colMin
    for i in range(k):
        centroid = colMin + np.random.rand(dim) * colRange
        centroids[i, :] = centroid
    print(centroids)
    return centroids


def initBadCentroids(data, k):
    count, dim = data.shape
    if dim == 2 and k == 3:
        centroids = np.zeros((3, 2))
        centroids[0, :] = [0, 5]
        centroids[1, :] = [20, 7]
        centroids[2, :] = [27, 8]
        return centroids
    else:
        print("unable to construct special centroids")
        return initRandomCentroids(data, k)


def initGoodCentroids(data, k):
    count, dim = data.shape
    if dim == 2 and k == 3:
        centroids = np.zeros((3, 2))
        centroids[0, :] = [-3, 9]
        centroids[1, :] = [25, 8]
        centroids[2, :] = [4, 2]
        return centroids
    else:
        print("unable to construct special centroids")
        return initRandomCentroids(data, k)


def kmeans(k, case):
    data = loadData()
    # data = loadBirch()
    count = data.shape[0]
    centroids = []
    if case == 0:
        centroids = initRandomCentroids(data, k)
    elif case == 1:
        centroids = initBadCentroids(data, k)
    elif case == 2:
        centroids = initGoodCentroids(data, k)

    clusterBound = np.zeros((count, 2))
    index = np.zeros((count, 1))
    processing = True
    step = 1
    # while step < 20:
    while processing:
        step = step + 1
        processing = False
        for i in range(count):
            minIndex = 0
            minDist = float("inf")
            for j in range(k):
                distance = euclideanDistance(centroids[j, :], data[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            if clusterBound[i, 0] != minIndex:
                processing = True
                clusterBound[i, :] = minIndex, minDist ** 2
        index[:, 0] = clusterBound[:, 0]

        visualization(centroids, clusterBound, data)
        for j in range(k):
            newCentroid = data[np.all(index == j, axis=1), :]
            centroids[j, :] = np.mean(newCentroid, axis=0)
    print("k means finished!")
    visualization(centroids, clusterBound, data)


def visualization(centroids, clusterBound, data):
    plotMarkList = ['oy', 'og', 'oc', 'oc', '^m', '+y', 'sk', 'dw', '<b', 'pg']
    centroidMarkList = ['Dr', 'Dr', 'Dr', 'Dy', '^k', '+w', 'sb', 'dg', '<r', 'pc']
    k = centroids.shape[0]
    count = data.shape[0]
    if data.shape[1] != 2:
        print("too many dimensions to draw :(")
        return
    if k > len(plotMarkList):
        print("too many centroids to draw :(")
        return
    plt.figure()
    for i in range(count):
        mark = plotMarkList[int(clusterBound[i, 0])]
        plt.plot(data[i, 0], data[i, 1], mark)
    for i in range(k):
        mark = centroidMarkList[i]
        plt.plot(centroids[i, 0], centroids[i, 1], mark)
    plt.show()


if __name__ == "__main__":
    kmeans(3, 0)
