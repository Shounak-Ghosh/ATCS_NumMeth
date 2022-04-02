import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    """
    Function 1 to integrate
    @param x: x value/array
    @return: f(x)
    """
    return 1 + x**2


def f1_integral(x):
    """
    Closed form integral of f1
    @param x: x value/array
    @return: integral of f1
    """
    return x + 1/3 * x ** 3


def f2(x):
    """
    Function 2 to integrate
    @param x: x value
    @return: f(x)
    """
    return x * np.exp(-x ** 2)


def f2_integral(x):
    """
    Closed form integral of f2
    @param x: x value/array
    @return: integral of f2
    """
    return np.exp(-x ** 2) / 2


def f3(x):
    """
    Function 3 to integrate
    @param x: x value
    @return: f(x)
    """
    return x * np.exp(-x)

def f3_integral(x):
    """
    Closed form integral of f3
    @param x: x value/array
    @return: integral of f3
    """
    return -np.exp(-x) * x - np.exp(-x)


def f4(x):
    """
    Function 4 to integrate
    @param x: x value
    @return: f(x)
    """
    return np.sin(x)

def f4_integral(x):
    """
    Closed form integral of f4
    @param x: x value/array
    @return: integral of f4
    """
    return np.cos(x)

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
    h = (b - a) / (n + 1)  # delta x
    x = np.linspace(a, b, n + 1)  # n + 1 to include the endpoints
    sum = np.sum(f(x) + np.pad(f(x[1:]), (1,0)))
    return h / 2 * sum


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

    if n % 2 == 1:
        n += 1

    h = (b - a) / n  # delta x
    x = np.linspace(a, b, n)  
    sum = np.sum(f(x[::2]) + 4 * f(x[1::2]) + np.pad(f(x[2::2]), (1,0)))
    return h / 3 * sum


def main():
    """
    Main function
    """
    x = np.linspace(-2, 2, 10)
    # Original function plots
    plt.plot(x, f1(x), label='f1')
    # plt.plot(x, f2(x), label='f2')
    # plt.plot(x, f3(x), label='f3')
    # plt.plot(x, f4(x), label='f4')

    # Closed form integral plots
    plt.plot(x, f1_integral(x), label='f1 integral')
    plt.scatter(x, f1_integral(x))
    # plt.plot(x, f2_integral(x), label='f2 integral')
    # plt.scatter(x, f2_integral(x))
    # plt.plot(x, f3_integral(x), label='f3 integral')
    # plt.scatter(x, f3_integral(x))
    # plt.plot(x, f4_integral(x), label='f4 integral')
    # plt.scatter(x, f4_integral(x))

    # # Trapezoidal rule plots
    # plt.scatter(x[0], trapezoidal(f1, x[0], x[1], 1000), label='f1 trapezoidal')


    print(f1_integral(2) - f1_integral(-2))
    print(trapezoidal(f1, -2, 2, 1000))
    print(simpsons(f1, -2, 2, 10))

    plt.legend()
    plt.show()


# run main
if __name__ == '__main__':
    main()
