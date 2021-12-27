import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def loadData():
    df = pd.read_csv("./data/data.csv")
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


def kmeans(k):
    data = loadData()
    count = data.shape[0]
    centroids = initRandomCentroids(data, k)
    clusterBound = np.zeros((count, 2))
    index = np.zeros((count, 1))
    processing = True
    while processing:
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
        for j in range(k):
            newCentroid = data[np.all(index == j, axis=1), :]
            centroids[j, :] = np.mean(newCentroid, axis=0)
    print("k means finished!")
    visualization(centroids, clusterBound, data)


def visualization(centroids, clusterBound, data):
    plotMarkList = ['oy', 'og', 'or', 'oc', '^m', '+y', 'sk', 'dw', '<b', 'pg']
    centroidMarkList = ['Dr', 'Dc', 'Dm', 'Dy', '^k', '+w', 'sb', 'dg', '<r', 'pc']
    k = centroids.shape[0]
    count = data.shape[0]
    if data.shape[1] != 2:
        print("too many dimensions to draw :(")
        return
    if k > len(plotMarkList):
        print("too many centroids to draw :(")
        return
    for i in range(count):
        mark = plotMarkList[int(clusterBound[i, 0])]
        plt.plot(data[i, 0], data[i, 1], mark)
    for i in range(k):
        mark = centroidMarkList[i]
        plt.plot(centroids[i, 0], centroids[i, 1], mark)
    plt.show()


if __name__ == "__main__":
    kmeans(3)
