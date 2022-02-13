import matplotlib.pyplot as plt
import numpy as np
import math

#TODO document the code

'''

@author Shounak Ghosh
@version 2.11.2022 

'''


def compute_f1(t):
    return math.exp(-t * t)


def compute_f2(t):
    return -2 * t * math.exp(-t * t)


def compute_times(lowerbound, higherbound, num):
    return np.linspace(lowerbound, higherbound, num)


def compute_f1_values(times):
    values = []
    for t in times:
        values.append([t, compute_f1(t)])
    return values


def compute_f2_values(times):
    values = []
    for t in times:
        values.append([t, compute_f2(t)])
    return values


def compute_simple_derivatives(values):
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


def output_results_to_file(filename, values, derivs):
    fo = open(filename, "wb")
    # TODO I/O STUFF
    fo.close()


def compute_3_point_derivative(values):
    deriv = []

    for i in range(1, len(values) - 1):
        left_slope = (values[i][1] - values[i - 1][1]) / (
                values[i][0] - values[i - 1][0])
        right_slope = (values[i + 1][1] - values[i][1]) / (
                values[i + 1][0] - values[i][0])
        deriv.append([values[i][0], (left_slope + right_slope) / 2])
    return deriv


def compute_functional_fit_derivative(values):
    derivs = []
    for i in range(1, len(values) - 1):
        t1 = values[i - 1][0]
        t2 = values[i][0]
        t3 = values[i + 1][0]
        y1 = values[i - 1][1]
        y2 = values[i][1]
        y3 = values[i + 1][1]
        a = np.array([[t1 ** 2, t1, 1], [t2 ** 2, t2, 1], [t3 ** 2, t3, 1]])
        b = np.array([y1, y2, y3])
        ans = np.matmul(np.linalg.inv(a), b)
        derivs.append([t2, 2 * ans[0] * t2 + ans[1]])
    derivs.insert(0, derivs[0])
    derivs.append(derivs[len(derivs) - 1])
    return derivs


def compute_functional_fit_rms(derivs):
    rms = 0
    for deriv in derivs:
        rms += abs(deriv[1] ** 2 - compute_f2(deriv[0]) ** 2)
    rms /= len(derivs)
    return math.sqrt(rms)


def compute_five_point_stencil(values):
    derivs = []
    for i in range(2, len(values) - 2):
        deriv = (-values[i + 2][1] + 8 * values[i + 1][1] - 8 * values[i - 1][
            1] + values[i - 2][1]) / (12 * (values[i + 1][0] - values[i][0]))
        derivs.append([values[i][0], deriv])
    return derivs


def generate_rms_plot_data(initial, numpoints, scale):
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
    reformat = []
    for i in range(1, len(rms[0])):
        reformat.append([])
    for vals in rms:
        for i in range(1, len(vals)):
            reformat[i - 1].append([vals[0], vals[i]])

    plot_values(reformat[0], 'b')
    plot_points(reformat[0], 'g')
    plot_points(reformat[0], 'y')
    plot_points(reformat[0], 'r')


def plot_points(values, color):
    for val in values:
        plt.plot(val[0], val[1], color + '^')
        plt.draw()


def plot_values(values, color):
    t = []
    v = []
    for val in values:
        t.append(val[0])
        v.append(val[1])
    plt.scatter(t, v, color=color)
    plt.plot(t, v, color=color)


def main():
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

    # rms = compute_simple_rms_values(simple_derivs)

    ff_derivs = compute_functional_fit_derivative(vals)
    plot_values(ff_derivs, '#ff007f')
    # rms = compute_functional_fit_rms(ff_derivs)

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
