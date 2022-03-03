import matplotlib.pyplot as plt
import numpy as np
import math

times = np.linspace(-10, 10, 100)
step = times[1] - times[0]


def f0(t):
    return t ** 3 - t


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


def f_prime(f):
    return lambda t: (-f(t + 2 * step) + 8 * f(t + step) - 8 * f(
        t - step) + f(t - 2 * step)) / (12 * step)


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
    raise Exception("Bisection failed.")


def newton_raphson(initial, tol, max_iter, f):
    x = initial
    for i in range(max_iter):
        deriv = f_prime(f)(x)
        val = f(x)
        if abs(deriv) < tol:
            raise Exception("Stationary point reached.")
        x = x - val / deriv
        if abs(val) < tol:
            return [x, val]
    raise Exception("Newton-Raphson failed.")


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
    i = 1
    for function in function_list:
        print("f" + str(i))
        i += 1
        try:
            print(bisection(-10, 10, 10 ** -7, 10 ** 5, function))
        except Exception as e:
            print(e)

        try:
            print(newton_raphson(.2, 10 ** -7, 10 ** 5, function))
        except Exception as e:
            print(e)
        print("")

    # function = f4
    # bi = bisection(-10, 10, 0.000001, 100000, function)
    # nr = newton_raphson(1, 0.000001, 100000, function)

    color_list = ['r', 'b', 'g', 'y', 'm', 'c', '#03cf00', '#ff007f', '#00ffae']
    # Set the plot boundaries
    # plt.figure("Functions", figsize=(8, 8))
    # plt.xlim(-3, 3)
    # plt.ylim(-3, 3)
    # plt.xlabel("time")
    # plt.ylabel("values")
    # plot_axes()
    # plot_function(function, 'b')
    #
    # try:
    #     print(bi, nr)
    #     plt.plot(bi[0], bi[1], 'rD')
    #     plt.plot(nr[0], nr[1], 'gs')
    # except Exception as e:
    #     print(e)

    # for i in range(len(function_list)):
    #     plot_function(function_list[i], color_list[i])
    #     plot_function(f_prime(function_list[i]), color_list[i])
    plt.show()


if __name__ == '__main__':
    main()
