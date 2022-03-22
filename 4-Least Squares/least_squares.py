import matplotlib.pyplot as plt
import argparse
import numpy as np
import math

# TODO placeholder; should be replaced with a flexible h
STEP = 10 ** -3


def cubic(x, q):
    if isinstance(x, (int, np.float64)):
        return q[0] * x ** 3 + q[1] * x ** 2 + q[2] * x + q[3]
    elif isinstance(x, (list, np.ndarray, np.generic)):
        y = []
        for pt in x:
            y.append(q[0] * pt ** 3 + q[1] * pt ** 2 + q[2] * pt + q[3])
        return y


def gaussian(x, q):
    if isinstance(x, (int, np.float64)):
        return q[0] ** 2 * math.e ** ((x - q[1]) ** 2 / q[2] ** 2) + q[3] ** 2
    elif isinstance(x, (list, np.ndarray, np.generic)):
        y = []
        for pt in x:
            y.append(q[0] ** 2 * math.e ** ((pt - q[1]) ** 2 / q[2] ** 2) + q[
                3] ** 2)
        return y


def sine(x, q):
    if isinstance(x, (int, np.float64)):
        return q[0] * np.sin(q[1] * x + q[2]) + q[3]
    elif isinstance(x, (list, np.ndarray, np.generic)):
        y = []
        for pt in x:
            y.append(q[0] * np.sin(q[1] * pt + q[2]) + q[3])
        return y


def error(data, f, q):
    e = 0
    for i in range(len(data[0])):
        e += (data[1] - f(data[0], q)) ** 2
    return e / 2


def partial(f, x, q, i):  # limit definition: f(x,y,z) = f(x,y+h,z) - f(x,y,z)/h
    q[i] += 2 * STEP  # 2h
    v1 = f(x, q)
    q[i] -= STEP  # h
    v2 = f(x, q)
    q[i] -= 2 * STEP  # -h
    v3 = f(x, q)
    q[i] -= STEP
    v4 = f(x, q)
    return (-v1 + 8 * v2 - 8 * v3 + v4) / (12 * STEP)


def delta(data, f, x, q):
    l = 1  # lambda
    newq = [0] * len(q)
    newq[0] = q[0]

    for j in range(1, len(q)):  # skip q0 (constant)
        eqj = 0  # partial E/qj
        for i in range(1, len(q)):
            eqj += (data[1] - f(x, q)) * partial(f, x, q, i)

    print("DO THIS")


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


def plot_axes(x, y):
    """
    Plots x and y axes
    @param x: list of min and max x values
    @param y: list of min and max y values
    """
    # x = []
    # for time in times:
    #     x.append(0)
    # plt.plot(times, x, 'k-')
    # plt.plot(x, times, 'k-')
    t = [0, 0]
    plt.plot(x, t, 'k-')
    plt.plot(t, y, 'k-')


def main():
    plt.figure("Least Squares plot", figsize=(8, 8))
    x_bound = [-7, 7]
    y_bound = [-5, 5]
    plt.xlim(x_bound[0], x_bound[1])
    plt.ylim(y_bound[0], y_bound[1])
    plt.xlabel("time")
    plt.ylabel("values")
    plot_axes(x_bound, y_bound)

    times = np.linspace(-10, 10, 100)
    q = [1, 1, 0, 0]
    # print(times)
    # print(type(times))
    # print(sine(times, [1, 1, 1, 1]))
    plt.plot(times, sine(times, q))
    plt.scatter(times, sine(times, q))
    partial_list = []
    for t in times:
        partial_list.append(partial(sine, t, q, 2))
    plt.plot(times, partial_list)
    plt.scatter(times, partial_list)

    plt.show()


if __name__ == '__main__':
    main()
