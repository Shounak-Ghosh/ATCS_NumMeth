import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    """
    Function 1 to integrate
    @param x: x value
    @return: f(x)
    """
    return 1 + x**2


def f2(x):
    """
    Function 2 to integrate
    @param x: x value
    @return: f(x)
    """
    return x * np.exp(-x ** 2)


def f3(x):
    """
    Function 3 to integrate
    @param x: x value
    @return: f(x)
    """
    return x * np.exp(-x)


def f4(x):
    """
    Function 4 to integrate
    @param x: x value
    @return: f(x)
    """
    return np.sin(x)

# TODO Check if this is actually correct
# trapezoidal rule
def trapezoidal(f, a, b, n):
    """
    Uses the trapezoidal rule to approximate the integral of f from a to b
    @param f: function to integrate
    @param a: lower bound
    @param b: upper bound
    @param n: number of subintervals
    @return: integral of f from a to b
    """
    h = (b - a) / n # delta x
    x = np.linspace(a, b, n + 1) # n + 1 to include the endpoints
    sum = np.sum(f(x[1:-1]))
    return h / 2 * (f(a) + f(b) + sum)

# simpsons 1/3 rule
def simpsons(f, a, b, n):
    """
    Uses the simpsons 1/3 rule to approximate the integral of f from a to b
    @param f: function to integrate
    @param a: lower bound
    @param b: upper bound
    @param n: number of subintervals
    @return: integral of f from a to b
    """
    h = (b - a) / n # delta x
    x = np.linspace(a, b, n + 1) # n + 1 to include the endpoints
    sum = np.sum(f(x[::2]) + 4 * f(x[1::2]) + f(x[2::2]))
    return h / 3 * sum


def main():
    """
    Main function
    """
    x = np.linspace(-2, 2, 1000)
    plt.plot(x, f1(x), label='f1')
    plt.plot(x, f2(x), label='f2')
    plt.plot(x, f3(x), label='f3')
    plt.plot(x, f4(x), label='f4')
    plt.legend()
    plt.show()


# run main
if __name__ == '__main__':
    main()
