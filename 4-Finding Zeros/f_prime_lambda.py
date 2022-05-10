import matplotlib.pyplot as plt
import numpy as np
import math

times = np.linspace(-10, 10, 100)
step = times[1] - times[0]


def f(t):
    return 5 * t ** 3


def f_prime(f):
    return lambda t: (-f(t + 2 * step) + 8 * f(t + step) - 8 * f(
        t - step) + f(t - 2 * step)) / (12 * step)


def plot_axes():
    x = []
    for time in times:
        x.append(0)
    plt.plot(times, x, 'k-')
    plt.plot(x, times, 'k-')


def plot_function(f, color):
    vals = []
    for time in times:
        vals.append(f(time))

    plt.scatter(times, vals, color=color)
    plt.plot(times, vals, color=color)
    plt.draw()


def main():
    # Set the plot boundaries
    plt.figure("Functions", figsize=(8, 8))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("time")
    plt.ylabel("values")
    plot_axes()
    plot_function(f, 'b')
    plot_function(f_prime(f), 'r')
    plt.show()


if __name__ == '__main__':
    main()
