"""
Least squares lab
@author Shounak Ghosh
@version 5.03.2022
"""
import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt


def cubic(x, q):
    """
    Cubic function
    @param x: x value
    @param q: list of coefficients
    @return: y value
    """
    return q[0] * x ** 3 + q[1] * x ** 2 + q[2] * x + q[3]


def gaussian(x, q):
    """
    Gaussian function
    @param x: x value
    @param q: list of coefficients
    @return: y value
    """
    return q[0] ** 2 * math.e ** -((x - q[1]) ** 2 / q[2] ** 2) + q[3] ** 2


def sine(x, q):
    """
    Sine function
    @param x: x value
    @param q: list of coefficients
    @return: y value
    """
    return q[0] * np.sin(q[1] * x + q[2]) + q[3]


def read_data(filename):
    """
    Reads data from file
    @param filename: name of file
    @return: list of data
    """
    x = np.loadtxt(filename, usecols=0, dtype=float)
    y = np.loadtxt(filename, usecols=1, dtype=float)
    return np.array([x, y])


def generate_least_squares(data, f, q, step, learning_rate, momentum, threshold, max_iterations):
    """
    Generates least squares coefficients for the given function
    @param data: the raw data points
    @param f: the function being fitted to the data
    @param q: list of intiial coefficients
    @param step: step size for partial derivative computation
    @param l: the learning rate for gradient descent
    @return: list of final coefficients

    """
    epoch = 1e5
    prev_e = error(data, f, q)
    num_args = np.shape(q)[0]
    prev_dq = np.zeros(num_args)
    print("Inital Error:", "{:e}".format(prev_e))

    init_start = time.time()
    for k in range(int(max_iterations / epoch)):  # segment the iteration
        cycle_start = time.time()
        for m in range(int(epoch)):
            dq = learning_rate * compute_deltas(data, f, q, step)
            dq += momentum * prev_dq
            q += dq
            prev_dq = dq

            e = error(data, f, q)
            if e > prev_e:  # rollback
                q -= dq
                prev_dq = 0
                learning_rate /= 2
            else:
                learning_rate *= 1.05

            # print(prev_e - e)
            if abs(e - prev_e) < threshold:
                final_end = time.time()
                print("\nFinal # of Iterations:", int((k + 1) * epoch + m))
                print("Final Error:", "{:e}".format(e))
                print("Q:",
                      np.array2string(q, formatter={
                          'float_kind': '{0:.3f}'.format}))
                print("Total Runtime:", final_end - init_start, "\n")
                return q
            prev_e = e
        cycle_end = time.time()
        print("\n# of Iterations:", int((k + 1) * epoch))
        print("Current Error:", "{:e}".format(e))
        print("Q:",
              np.array2string(q, formatter={
                  'float_kind': '{0:.3f}'.format}))
        print("Epoch Runtime:", cycle_end - cycle_start)

    print("\nMaximum iterations exceeded.")
    print("Final Error:", "{:e}".format(e))
    print("Q:",
          np.array2string(q, formatter={
              'float_kind': '{0:.3f}'.format}))
    final_end = time.time()
    print("Total Runtime:", final_end - init_start, "\n")
    return q


def error(data, f, q):
    err_form = data[1] - f(data[0], q)
    # np.subtract(data[1], f(data[0], q))
    e = np.sum(err_form ** 2)
    return e / 2


# limit definition: f(x,y,z) = f(x,y+h,z) - f(x,y,z)/h
def compute_partials(data, f, q, j, step):
    """
    Computes the partial derivatives of the function using the five-point stencil method
    @param data: the raw data points
    @param f: the function being fitted to the data
    @param q: array of coefficients
    @param j: index of the partial derivative
    @return: array of partial derivatives with respect to j
    """

    q[j] += 2 * step  # 2h
    x = data[0]
    v1 = f(x, q)
    q[j] -= step  # h
    v2 = f(x, q)
    q[j] -= 2 * step  # -h
    v3 = f(x, q)
    q[j] -= step
    v4 = f(x, q)
    return (-v1 + 8 * v2 - 8 * v3 + v4) / (12 * step)


def compute_deltas(data, f, q, step):
    """
    Computes the deltas for the coefficients
    @param data: the raw data points
    @param f: the function being fitted to the data
    @param q: array of coefficients
    @return: array of deltas
    """
    num_args = np.shape(q)[0]
    dq = np.zeros(num_args)
    for j in range(num_args):
        partials = compute_partials(data, f, q, j, step)
        dq[j] = np.sum((data[1] - f(data[0], q)) * partials)
    return dq


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


def main(c, g, s, d_filename):
    """
    Main function
    """

    if d_filename is not None:
        data = read_data(d_filename)
    else:
        return

    # Define constants across all functions
    MAX_ITERATIONS = 1e6  # .6e6
    STEP = 1e-6
    MOMENTUM = 0.9
    if c is not None:
        # Specific constants for the gaussian function
        LAMBDA = 1e-10
        THRESHOLD = 1e-1
        cubic_q = np.asarray(c).astype(float)
        print("Fitting CUBIC to", d_filename)
        plt.figure("Cubic Least Squares Plot", figsize=(6, 6))
        plt.xlabel("x")
        plt.ylabel("y")
        cubic_q = generate_least_squares(
            data, cubic, cubic_q, STEP, LAMBDA, MOMENTUM, THRESHOLD, MAX_ITERATIONS)
        plt.scatter(data[0], data[1], label="Raw Data", color='b')
        plt.scatter(data[0], cubic(data[0], cubic_q), color='r',
                    label="Least Squares Estimation")
        plt.plot(data[0], cubic(data[0], cubic_q), color='r')
        plt.legend()
    elif g is not None:  # q = 1 10 3 2
        # Specific constants for the gaussian function
        LAMBDA = 1e-11
        THRESHOLD = 1e-1
        gaussian_q = np.asarray(g).astype(float)
        print("Fitting GAUSSIAN to", d_filename)
        plt.figure("Gaussian Least Squares Plot", figsize=(6, 6))
        plt.xlabel("x")
        plt.ylabel("y")
        gaussian_q = generate_least_squares(
            data, gaussian, gaussian_q, STEP, LAMBDA, MOMENTUM, THRESHOLD, MAX_ITERATIONS)
        plt.scatter(data[0], data[1], label="Raw Data", color='b')
        plt.scatter(data[0], gaussian(data[0], gaussian_q), color='r',
                    label="Least Squares Estimation")
        plt.plot(data[0], gaussian(data[0], gaussian_q), color='r')
        plt.legend()
    elif s is not None:
        # Define constants
        LAMBDA = 1e-5
        THRESHOLD = 1e-5

        sine_q = np.asarray(s).astype(float)
        print("Fitting SINE to", d_filename)
        plt.figure("Sine Least Squares Plot", figsize=(6, 6))
        plt.xlabel("x")
        plt.ylabel("y")
        sine_q = generate_least_squares(
            data, sine, sine_q, STEP, LAMBDA, MOMENTUM, THRESHOLD, MAX_ITERATIONS)
        plt.scatter(data[0], data[1], label="Raw Data", color='b', s=50)
        plt.scatter(data[0], sine(data[0], sine_q), color='r',
                    label="Least Squares Estimation")
        plt.plot(data[0], sine(data[0], sine_q), color='r')
        plt.legend()

    if c is None and g is None and s is None:
        print("No function selected.")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cubic", nargs="+",
                        help="initial parameters for cubic function")
    parser.add_argument("-g", "--gaussian", nargs='+',
                        help="intial parameters for gaussian function")
    parser.add_argument("-s", "--sine", nargs="+",
                        help="initial parameters for sine function")
    parser.add_argument("-d", "--dataset",
                        help="pwd of txt file containing the raw data",
                        type=str)
    args = parser.parse_args()
    main(args.cubic, args.gaussian, args.sine, args.dataset)
