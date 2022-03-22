import matplotlib.pyplot as plt
import argparse
import numpy as np
import math

# TODO placeholder; should be replaced with a flexible h
STEP = 10 ** -3


def cubic(x, q):
    if isinstance(x, int):
        return q[0] * x ** 3 + q[1] * x ** 2 + q[2] * x + q[3]
    elif isinstance(x, list):
        y = []
        for pt in x:
            y.append(q[0] * pt ** 3 + q[1] * pt ** 2 + q[2] * pt + q[3])
        return y


# TODO redefine to meet int list type specification
def gaussian(x, q):
    return q[0] ** 2 * math.e ** ((x - q[1]) ** 2 / q[2] ** 2) + q[3] ** 2


# TODO redefine to meet int list type specification
def sine(x, q):
    return q[0] * np.sin(q[1] * x + q[2]) + q[3]


def error(data, f, q):
    e = 0
    for i in range(len(data[0])):
        e += data[1] - f(data[0], q)
    return e / 2


# TODO define this properly
def partial(f, q, i):
    return lambda t: (-f(t + 2 * STEP) + 8 * f(t + STEP) - 8 * f(
        t - STEP) + f(t - 2 * STEP)) / (12 * STEP)


def plot_function(f, color):
    """
    Plots the function in the given color
    @param f: The function to be plotted
    @param color: The color the function should be plotted in
    """
    vals = []
    times = np.linspace(-10, 10, 100)
    for time in times:
        vals.append(f(time))

    plt.scatter(times, vals, color=color)
    plt.plot(times, vals, color=color)
    plt.draw()


def main():
    print("DO THIS")


if __name__ == '__main__':
    main()
