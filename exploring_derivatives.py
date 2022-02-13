import matplotlib.pyplot as plt
import numpy as np
import math
"""
Explores various methods of computing derivatives for a given function.

Simple Slope Derivatives: utilizes the slope between two adjacent points
Computed at left, right and mid time values.

Root Mean Square (RMS): Metric for evaluating approximation's closeness to the 
true value. Lower RMS values indicate a closer fit.

3-Point Derivative: Averages the slopes on either side of a given point

Functional Fit Derivative: Uses a parabolic fit on 3 distinct points, computes
the derivative based on the respective a, b, and c values found

Five-Point Stencil: Uses Taylor series for a direct approximation of the 
derivative.

@author Shounak Ghosh
@version 2.11.2022 

"""


def compute_f1(t):
    """
    Computes the value of the function whose derivative we are interested in
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return math.exp(-t * t)


def compute_f2(t):
    """
    Computes the value of the function's derivative (found via calculus)
    @param t: The time t the function is being evaluated at
    @return: The value of the function at time t
    """
    return -2 * t * math.exp(-t * t)


def compute_times(lowerbound, higherbound, num):
    """
    Determine the time points at which we are interested in function/derivative
    computations
    @param lowerbound: The min time value
    @param higherbound: The max time value
    @param num: The number of time points
    @return: An evenly spaced list of num times between lower and higher bound
    """
    return np.linspace(lowerbound, higherbound, num)


def compute_f1_values(times):
    """
        Computes the values of the function whose derivative we are interested in
        at the given times
        @param times: List of times at which f1 is to be evaluated at
        @return: A list of [t, f1(t)] pairs for each of the specified times
    """
    values = []
    for t in times:
        values.append([t, compute_f1(t)])
    return values


def compute_f2_values(times):
    """
        Computes the values of the function's derivative we are interested in
        at the given times
        @param times: List of times at which f2 is to be evaluated at
        @return: A list of [t, f2(t)] pairs for each of the specified times
    """
    values = []
    for t in times:
        values.append([t, compute_f2(t)])
    return values


def compute_simple_derivatives(values):
    """
    Computes derivatives using the simple slope method
    @param values: A list of [time, f1(time)] pairs
    @return: Right, left, and mid derivative lists in a [t, derivative] format
    """
    first = []
    second = []
    mid = []
    for i in range(len(values) - 1):
        slope = (values[i + 1][1] - values[i][1]) / (
                values[i + 1][0] - values[i][0])
        first.append([values[i][0], slope])  # time, value pair
        second.append([values[i + 1][0], slope])
        mid.append([(values[i][0] + values[i + 1][0]) / 2, slope])
    return [first, second, mid]


def compute_simple_rms_values(derivs):
    """
    Computes the RMS for right, left, and middle derivatives
    @param derivs: List of right, left and middle derivatives ([t, derivative])
    @return: A list with the three RMS values
    """
    rms_first = 0
    rms_second = 0
    rms_mid = 0
    for deriv in derivs[0]:
        rms_first += abs(deriv[1] ** 2 - compute_f2(deriv[0]) ** 2)
    for deriv in derivs[1]:
        rms_second += abs(deriv[1] ** 2 - compute_f2(deriv[0]) ** 2)
    for deriv in derivs[2]:
        rms_mid += abs(deriv[1] ** 2 - compute_f2(deriv[0]) ** 2)
    n = len(derivs[0]) + 1
    rms_first /= n
    rms_second /= n
    rms_mid /= n
    return [math.sqrt(rms_first), math.sqrt(rms_second), math.sqrt(rms_mid)]


def output_results_to_file(values, derivs):
    """
    Outputs the values, simple derivatives, and RMS values to a file
    @param values: The function values as a list in a [t, f1(t)] format
    @param derivs: The right, left, and middle derivatives ([t, derivative])
    """
    rms = compute_simple_rms_values(derivs)
    # inf represents NULL, included for even array spacing
    derivs[0].append([values[len(values) - 1][0], float('-inf')])
    derivs[1].insert(0, [values[0][0], float('inf')])
    derivs[2].insert(0, [values[0][0], float('inf')])
    out = []

    for i in range(len(values)):
        out.append(
            [values[i][0], values[i][1], derivs[0][i][1], derivs[1][i][1],
             derivs[2][i][1]])
    out.append([rms[0], rms[1], rms[2], float('inf'), float('inf')])
    np.savetxt("simple_slopes.txt", out, delimiter=', ')


def compute_3_point_derivative(values):
    """
    Computes derivatives using the 3-point method
    @param values: A list of [time, f1(time)] pairs
    @return: A list of [time, derivative] pairs
    """
    deriv = []

    for i in range(1, len(values) - 1):
        left_slope = (values[i][1] - values[i - 1][1]) / (
                values[i][0] - values[i - 1][0])
        right_slope = (values[i + 1][1] - values[i][1]) / (
                values[i + 1][0] - values[i][0])
        deriv.append([values[i][0], (left_slope + right_slope) / 2])
    return deriv


def compute_functional_fit_derivative(values):
    """
    Computes derivatives using the functional fit method
    @param values: A list of [time, f1(time)] pairs
    @return: A list of [time, derivative] pairs
    """
    derivs = []
    for i in range(1, len(values) - 1):
        t1 = values[i - 1][0]
        t2 = values[i][0]
        t3 = values[i + 1][0]
        y1 = values[i - 1][1]
        y2 = values[i][1]
        y3 = values[i + 1][1]
        # m1 and m2 are matrices, ans equals [a, b, c] via basic linear algebra
        m1 = np.array([[t1 ** 2, t1, 1], [t2 ** 2, t2, 1], [t3 ** 2, t3, 1]])
        m2 = np.array([y1, y2, y3])
        ans = np.matmul(np.linalg.inv(m1), m2)
        derivs.append([t2, 2 * ans[0] * t2 + ans[1]])
    # add derivative values for first and last time points
    derivs.insert(0, derivs[0])
    derivs.append(derivs[len(derivs) - 1])
    return derivs


def compute_functional_fit_rms(derivs):
    """
    Computes the RMS for functional fit derivatives
    @param derivs: List of functional fit derivatives ([t, derivative])
    @return: The RMS value
    """
    rms = 0
    for deriv in derivs:
        rms += abs(deriv[1] ** 2 - compute_f2(deriv[0]) ** 2)
    rms /= len(derivs)
    return math.sqrt(rms)


def compute_five_point_stencil(values):
    """
    Computes derivatives using the functional fit method
    @param values: A list of [time, f1(time)] pairs
    @return: A list of [time, derivative] pairs
    """
    derivs = []
    for i in range(2, len(values) - 2):
        deriv = (-values[i + 2][1] + 8 * values[i + 1][1] - 8 * values[i - 1][
            1] + values[i - 2][1]) / (12 * (values[i + 1][0] - values[i][0]))
        derivs.append([values[i][0], deriv])
    return derivs


def generate_rms_plot_data(initial, numpoints, scale):
    """
    Computes RMS values for different derivative computation methods
    for differing values of N (the number of points)
    @param initial: Starting value of N
    @param numpoints: the number of different N's we are finding RMS values for
    @param scale: Multiplicative factor by which N is increased
    @return: List of RMS values for each N,
             format [N, simple RMS's, functional fit RMS]
    """
    rms = []
    for i in range(numpoints):
        n = int(initial * scale ** i)
        times = compute_times(-10.0, 10.0, n)
        vals = compute_f1_values(times)
        simple_derivs = compute_simple_derivatives(vals)
        s_rms = compute_simple_rms_values(simple_derivs)
        ff_derivs = compute_functional_fit_derivative(vals)
        ff_rms = compute_functional_fit_rms(ff_derivs)
        s_rms.insert(0, n)
        s_rms.append(ff_rms)
        rms.append(s_rms)
    return rms


def plot_rms_n(rms):
    """
    Creates an RMS vs N plot
    @param rms: RMS data in the [N, simple RMS's, functional fit RMS] format
    """
    reformat = []
    for i in range(1, len(rms[0])):
        reformat.append([])
    for vals in rms:
        for i in range(1, len(vals)):
            reformat[i - 1].append([vals[0], vals[i]])

    # Use plot_points if there is overlap
    plot_values(reformat[0], 'b')
    plot_points(reformat[1], 'g')
    plot_values(reformat[2], 'y')
    plot_values(reformat[3], 'r')


def plot_points(values, color):
    """
    Plots the points given by values
    @param values: List of points in a [x, y] format
    @param color: The color of the points
    """
    for val in values:
        plt.plot(val[0], val[1], color + '^')
        plt.draw()


def plot_values(values, color):
    """
    Plots and connects the points given by values
    @param values: List of points in a [x, y] format
    @param color: The color of the points
    """
    t = []
    v = []
    for val in values:
        t.append(val[0])
        v.append(val[1])
    plt.scatter(t, v, color=color)
    plt.plot(t, v, color=color)


def main():
    """
    Driver method
    """
    times = compute_times(-10.0, 10.0, 100)
    vals = compute_f1_values(times)

    # Set the plot boundaries
    plt.figure("Exploring Derivatives", figsize=(8, 8))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("time")
    plt.ylabel("values")

    plot_values(vals, 'k')
    plot_values(compute_f2_values(times), 'c')

    simple_derivs = compute_simple_derivatives(vals)
    plot_values(simple_derivs[0], 'r')
    plot_values(simple_derivs[1], 'y')
    plot_values(simple_derivs[2], 'm')

    output_results_to_file(vals, simple_derivs)

    ff_derivs = compute_functional_fit_derivative(vals)
    plot_values(ff_derivs, '#ff007f')

    plot_values(compute_3_point_derivative(vals), 'g')

    plot_points(compute_five_point_stencil(vals), 'b')

    plt.draw()

    plt.figure("RMS Vs. Step Plot", figsize=(8, 8))
    plt.xlabel("N")
    plt.ylabel("RMS Values")
    rms_plt_data = generate_rms_plot_data(10, 10, 2)
    plot_rms_n(rms_plt_data)
    plt.draw()

    plt.show()


if __name__ == '__main__':
    main()
