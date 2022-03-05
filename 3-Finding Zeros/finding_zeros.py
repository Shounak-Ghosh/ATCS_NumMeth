import matplotlib.pyplot as plt
import argparse
import numpy as np
import math

"""
Tests the bisection and Newton-Raphson methods for numerically converging 
to the zeros of a function.

The methods above for finding zeros are also applied to the function's 
derivative to find local min/maxes.

Test functions f0 through f9 are included.

@author Shounak Ghosh
@version 3.04.2022
"""

TOLERANCE = 10 ** -7
MAX_ITERATIONS = 10 ** 5
times = np.linspace(-10, 10, 100)
step = times[1] - times[0]


def f0(t):
    """
    Computes f(t) = sin(t)
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return np.sin(t)


def f1(t):
    """
    Computes f(t) = t^2
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return t ** 2


def f2(t):
    """
    Computes f(t) = 1 + t^2
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return 1 + t ** 2


def f3(t):
    """
    Computes f(t) = e^(-t^2)
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return math.exp(-t ** 2)


def f4(t):
    """
    Computes f(t) = te^(-t^2)
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return t * math.exp(-t ** 2)


def f5(t):
    """
    Computes f(t) = te^(t^2)
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return t * math.exp(t ** 2)


def f6(t):
    """
    Computes f(t) = te^(-t)
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return t * math.exp(-t)


def f7(t):
    """
    Computes f(t) = te^(t)
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return t * math.exp(t)


def f8(t):
    """
    Computes f(t) = |t|^(1/2)
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return math.sqrt(abs(t))


def f9(t):
    """
    Computes f(t) = .7t^3 -7t - 2
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return 0.7 * t ** 3 - 7 * t - 2


def f_prime(f):
    """
    Approximates the derivative of a given function at some time
    @param f: The function whose derivative we are approximating
    @return A function which takes in a time t and outputs an approximation
            of the inputted functions derivative (uses the five-point stencil)
    """
    return lambda t: (-f(t + 2 * step) + 8 * f(t + step) - 8 * f(
        t - step) + f(t - 2 * step)) / (12 * step)


def bisection(left, right, tol, max_iter, f):
    """
    Uses the bisection method to find a root (if any) for a given function
    @param left: A x-value to the left of the root
    @param right: A x-value to the right of the root
           (y-value should evaluate to the opposite side of the axis
            compared to the left y-value)
    @param tol: The interval of tolerance that is "close enough" to zero
    @param max_iter: The upper bound on number of search iterations
    @param f: The function we are performing the bisection method on
    @return: A root, if one exists
    """
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
    """
    Uses the Newton-Raphson method to find a root (if any) for a given function
    @param initial: The initial x-value
           (should be as close as possible to the actual root)
    @param tol: The interval of tolerance that is "close enough" to zero
    @param max_iter: The upper bound on number of search iterations
    @param f: The function we are performing the bisection method on
    @return: A root, if one exists
    """
    x = initial
    for i in range(max_iter):
        deriv = f_prime(f)(x)
        val = f(x)
        if deriv == 0:
            raise Exception("Local Min/Max encountered. Unable to continue.")
        x = x - val / deriv
        if abs(val) < tol:
            if np.sign(f(x + step)) != np.sign(f(x - step)):
                return [x, val]
            else:
                break
    if np.sign(f(x + step)) > 0 and np.sign(f(x - step)) > 0:
        raise Exception(
            "Approaching positive asymptote. Choose a smaller initial value.")
    elif np.sign(f(x + step)) < 0 and np.sign(f(x - step)) < 0:
        raise Exception(
            "Approaching negative asymptote. Choose a larger initial value.")
    else:
        raise Exception("Newton-Raphson failed.")


def find_min_max(f):
    """
    Finds a local min/max of the given function
    @param f: The function in question
    @return A local min/max of the function
    """
    try:
        pt = newton_raphson(.6, TOLERANCE, MAX_ITERATIONS, f_prime(f))
        if f_prime(f_prime(f))(pt[0]) < 0:  # local max found
            return [pt[0], f(pt[0]), True]
        else:
            return [pt[0], f(pt[0]), False]
    except Exception as e:
        print(e)


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


def plot_function(f, color):
    """
    Plots the function in the given color
    @param f: The function to be plotted
    @param color: The color the function should be plotted in
    """
    vals = []
    for time in times:
        vals.append(f(time))

    plt.scatter(times, vals, color=color)
    plt.plot(times, vals, color=color)
    plt.draw()


def main(index, v):
    """
    Driver method
    @param index: The index of the function to be plotted
    @param v: Displays a verbose output when '-v' flag is used
    """
    function_list = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]
    color_list = ['r', 'b', 'g', 'y', 'm', 'c', '#03cf00', '#ff007f', '#00ffae']

    if v:
        i = 0
        for f in function_list:
            print("f" + str(i))
            i += 1
            try:
                print("b_zero",
                      bisection(-10, 10, TOLERANCE, MAX_ITERATIONS, f))
            except Exception as e:
                print(e)

            try:
                print("n_zero",
                      newton_raphson(.1, TOLERANCE, MAX_ITERATIONS, f))
                val = find_min_max(f)
                if val[2]:
                    print("Local max found.")
                else:
                    print("Local min found.")
                print(val[:2])
            except Exception as e:
                print(e)
            print("")

    if index is not None:
        function = function_list[index]
        # Set the plot boundaries
        plt.figure("Functions", figsize=(8, 8))
        x_bound = [-5, 5]
        y_bound = [-10, 10]
        plt.xlim(x_bound[0], x_bound[1])
        plt.ylim(y_bound[0], y_bound[1])
        plt.xlabel("time")
        plt.ylabel("values")
        plot_axes(x_bound, y_bound)
        plot_function(function, 'b')
        plot_function(f_prime(function), 'c')

        try:
            bi = bisection(-10, 10, 0.000001, 100000, function)
            nr = newton_raphson(.2, 0.000001, 100000, function)
            mm = find_min_max(function)
            plt.plot(bi[0], bi[1], 'rD')
            plt.plot(nr[0], nr[1], 'gs')
            if mm is not None:
                plt.plot(mm[0], mm[1], color='#ff007f', marker='p')
        except Exception as e:
            print(e)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",
                        help="output zeros and min/max for all functions",
                        action='store_true')
    parser.add_argument("-f", "--plot", help="index of function to be plotted",
                        type=int,
                        default=None)

    args = parser.parse_args()
    main(args.plot, args.verbose)
