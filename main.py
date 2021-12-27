# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    a = np.zeros((10, 2))
    b = np.zeros((10, 1))
    a[2, 0] = 1
    a[3, 1] = 2
    print(a)
    b[:, 0] = a[:, 0]
    print(b)
    print(a[np.all(b == 0, axis=1), :])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
