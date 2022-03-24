import numpy as np


def cubic(x):
    if isinstance(x, int):
        return x ** 3
    elif isinstance(x, list):
        y = []
        for pt in x:
            y.append(pt ** 3)
        return y


cubic = np.frompyfunc(cubic, 1, 1)
# print(cubic([1, 2, 3]))

# x1 = np.arange(9.0).reshape((3, 3))
# x2 = np.arange(3.0)
# print(x1)
# print(x2)
# print(x1 + x2)
print(np.zeros((2, 4)))
