import matplotlib.pyplot as plt
import numpy as np
import math

times = np.linspace(-10, 10, 100)


def f0(t):
    return t ** 3 - t - 2


def f1(t):
    return t ** 2


def f2(t):
    return 1 + t ** 2


def f3(t):
    return math.exp(-t ** 2)


def f4(t):
    return t * math.exp(-t ** 2)


def f5(t):
    return t * math.exp(t ** 2)


def f6(t):
    return t * math.exp(-t)


def f7(t):
    return t * math.exp(t)


def f8(t):
    return math.sqrt(abs(t))


def f9(t):
    return t ** 3


def bisection(left, right, tol, max_iter, f):
    assert left < right, \
        "Leftmost endpoint must be less than rightmost endpoint."
    assert (f(left) < 0 and f(right) > 0) or (f(left) > 0 and f(right) < 0), \
        "One endpoint must be greater than 0; The other less than 0."

    n = 1


    while n < max_iter:
        mid = (left + right) / 2
        mid_val = f(mid)
        if mid_val == 0 or (right - left) / 2 < tol:  # solution found
            return [mid, mid_val]
        n += 1
        if np.sign(mid_val) == np.sign(f(left)):
            left = mid
        else:
            right = mid
    return "Bisection failed."


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
    function_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    for function in function_list:
        try:
            print(bisection(-3, 3, 0.000001, 100000, function))
        except AssertionError as msg:
            print(msg)

    color_list = ['r', 'b', 'g', 'y', 'm', 'c', '#3c00ff', '#ff007f', '#00ffae']
    # Set the plot boundaries
    plt.figure("Functions", figsize=(8, 8))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("time")
    plt.ylabel("values")
    plot_axes()
    for i in range(len(function_list)):
        plot_function(function_list[i], color_list[i])
    plt.show()


if __name__ == '__main__':
    main()
