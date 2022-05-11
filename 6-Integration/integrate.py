"""
Integration Lab
@author Shounak Ghosh
@version 5.09.2022
"""
from turtle import color
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
    return -np.exp(-x ** 2) / 2


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
    return -np.cos(x)


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
    sum = np.sum(f(x) + np.pad(f(x[1:]), (1, 0)))
    return h / 2 * sum

def simpsons13(f, a, b, n):
    """
    Uses the simpsons 1/3 rule to approximate the integral of f from a to b
    @param f: function to integrate
    @param a: lower bound
    @param b: upper bound
    @param n: number of subintervals
    @return: integral of f from a to b
    """
    return (b-a)/6 * (f(a) + 4*f((a+b)/2) + f(b))

def simpsons_composite(f, a, b, n):
    """
    Uses the simpsons 1/3 rule to approximate the integral of f from a to b
    @param f: function to integrate
    @param a: lower bound
    @param b: upper bound
    @param n: number of subintervals
    @return: integral of f from a to b
    """
    # make n even
    if n % 2 == 1:
        n += 1

    h = (b - a) / n  # delta x
    x = np.linspace(a, b, n)
    sum = np.sum(f(x[::2]) + 4 * f(x[1::2]) + np.pad(f(x[2::2]), (1, 0)))
    return h / 3 * sum

def simpsons38(f, a, b, n):
    """
    Uses the simpsons 3/8 rule to approximate the integral of f from a to b
    @param f: function to integrate
    @param a: lower bound
    @param b: upper bound
    @param n: number of subintervals
    @return: integral of f from a to b
    """
    return (b-a)/8 * (f(a) + 3*f((2*a+b)/3) + 3*f((a+2*b)/3) + f(b))


def plot_integral(intf, f, a, b, n):
    """
    Plots the integral of f from a to b
    @param intf: The integrating function being used (trapezoidal or simpsons)
    @param f: function to integrate
    @param a: lower bound
    @param b: upper bound
    @param n: number of subintervals
    """
    x = np.linspace(a, b, n)
    y = np.zeros(n)

    for i in range(n):
        y[i] = intf(f, a, x[i], n)

    # find constant and adjust the integral plot accordingly
    # min_index = np.argmin(f(x))
    # max_index = np.argmax(f(x))
    # c = y[min_index]
    # if min_index == 0 or min_index == n - 1:
    #     c = y[max_index]
    # elif max_index == 0 or max_index == n - 1:
    #     c = y[min_index]
    # elif min_index == n-1 or min_index == n - 1 and max_index == 0 or max_index == n - 1:
    #     c = 0
    # print(min_index, np.amin(f(x)), y[min_index])
    # print(max_index, np.amax(f(x)), y[max_index])
    # print(c)
    c=0

    y += -c
    return y


def plot_axes(x, y):
    """
    Plots x and y axes
    @param x: list of min and max x values
    @param y: list of min and max y values
    """

    t = [0, 0]
    plt.plot(x, t, 'k-')
    plt.plot(t, y, 'k-')

#TODO closed form plot does not align with the numerically calculated integral
# because the "sweeping" method (subtracting evenly spaced points from A) 
# is not being used
def main():
    """
    Main function
    """
    f_arr = [f1, f2, f3, f4]
    f_closed_arr = [f1_integral, f2_integral, f3_integral, f4_integral]
    n = 100
    for i in range(len(f_arr)):
        print("f"+str(i))
        print("closed form integral:", str(
            f_closed_arr[i](2) - f_closed_arr[i](-2)))
        print("trapezoidal:", trapezoidal(f_arr[i], -2, 2, n))
        print("simpsons:", simpsons38(f_arr[i], -2, 2,100), '\n')

    # print(trapezoidal(f4, -2, 0, n))
    # print(simpsons(f4, -2, 0, n))
    # print(f4_integral(0)- f4_integral(-2))

    # Plotting
    for index in range(4):
        plt.figure(str(index+1)+" Integral Plot", figsize=(6, 6))
        x = np.linspace(-2, 2, 100)
        # plot_axes([-2, 2], [-4, 4])
        # plt.plot(x, f_arr[index](x), 'b-', label='f'+str(index + 1))
        plt.plot(x, f_closed_arr[index](x), 'r-',
             label='closed form f' + str(index + 1) + 'integral')
        plt.scatter(x, plot_integral(trapezoidal,
                f_arr[index], -2, 2, 100), marker='o', color='xkcd:warm purple', alpha=0.4, label='f' + str(index + 1) + '_trapezoidal')
        plt.scatter(x, plot_integral(simpsons38,
                f_arr[index], -2, 2, 100), marker='^', color='c', alpha=0.4, label='f' + str(index + 1) + '_simpsons')
         # plt.scatter(np.linspace(-3.14/2,2,50), plot_integral(trapezoidal,f4,-3.14/2,2,50), marker='o', label='f4_trapezoidal')
        plt.draw()
        
    plt.show()


# run main
if __name__ == '__main__':
    main()
