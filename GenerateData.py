import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# if __name__ == "__main__":
#     num, dim = 120, 2
#     num2 = 300
#     np.random.seed(0)
#     x1 = np.random.randn(num, dim)
#     x2 = np.random.randn(num2, dim)
#     x3 = np.random.randn(num, dim)
#     # C1 = [[0.6, -0.88], [-0.458, 0.31]]
#     C1 = [[1.5, 2.5], [2.5, -1.5]]
#     C2 = [[4, -2], [2, 3.5]]
#     # C3 = [[0.7, -0.9], [-0.8, -0.6]]
#     C3 = [[2, -2], [2, 2]]
#     W1 = [-3, 9]
#     W2 = [25, 8]
#     W3 = [3, 0]
#     Z1 = np.dot(x1, C1) + W1
#     Z2 = np.dot(x2, C2) + W2
#     Z3 = np.dot(x3, C3) + W3
#     Z = np.vstack((Z1, Z2, Z3))
#     plt.scatter(Z[:, 0], Z[:, 1])
#     df = pd.DataFrame(Z)
#     df.to_csv("./data/data.csv", index=False)
#     plt.show()

if __name__ == "__main__":
    num, dim = 200, 2
    num2 = 200
    np.random.seed(0)
    x1 = np.random.randn(num, dim)
    x2 = np.random.randn(num2, dim)
    x3 = np.random.randn(num, dim)
    C1 = [[10, 5], [5, -10]]
    C2 = [[10, -6], [5.5, 9]]
    C3 = [[7, -10], [8, 9]]
    W1 = [-25, 30]
    W2 = [40, 10]
    W3 = [0, 0]
    Z1 = np.dot(x1, C1) + W1
    Z2 = np.dot(x2, C2) + W2
    Z3 = np.dot(x3, C3) + W3
    Z = np.vstack((Z1, Z2, Z3))
    plt.scatter(Z[:, 0], Z[:, 1])
    df = pd.DataFrame(Z)
    df.to_csv("./data/data.csv", index=False)
    plt.show()

