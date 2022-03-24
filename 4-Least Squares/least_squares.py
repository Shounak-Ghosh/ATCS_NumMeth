import matplotlib.pyplot as plt
import argparse
import numpy as np
import math
import time

MAX_ITERATIONS = 1e8
THRESHOLD = 1e-5
# TODO placeholder; should be replaced with a flexible h
STEP = 1e-6
MOMENTUM = 0.9
LAMBDA = 1e-11


def cubic(x, q):
    return q[0] * x ** 3 + q[1] * x ** 2 + q[2] * x + q[3]


def gaussian(x, q):
    return q[0] ** 2 * math.e ** -((x - q[1]) ** 2 / q[2] ** 2) + q[3] ** 2


def sine(x, q):
    return q[0] * np.sin(q[1] * x + q[2]) + q[3]


def read_data(filename):
    raw_data = [[], []]
    with open(filename, 'r') as f:
        for line in f:
            l = line.split()
            # print(l)
            raw_data[0].append(int(l[0]))
            raw_data[1].append(int(l[1]))
    data = np.zeros((2, len(raw_data[0])))
    for i in range(len((data[0]))):
        data[0][i] = raw_data[0][i]
        data[1][i] = raw_data[1][i]
    return data


def generate_least_squares(data, f, q):
    check = 1e5
    prev_e = error(data, f, q)
    prev_dq = np.zeros(len(q))
    print("Inital Error:", "{:e}".format(error(data, f, q)))

    for i in range(int(MAX_ITERATIONS / check)):  # segment the iteration
        start = time.time()
        for j in range(int(check)):
            dq = compute_deltas(data, f, q)
            dq = np.add(dq, np.multiply(prev_dq, MOMENTUM))
            q = np.add(q, dq)

            e = error(data, f, q)
            # print(prev_e - e)
            if abs(prev_e - e) < THRESHOLD:
                print("\nFinal # of Iterations:", int((i + 1) * check + j))
                print("Final Error:", "{:e}".format(e))
                print("Q:",
                      np.array2string(q, formatter={
                          'float_kind': '{0:.3f}'.format}))
                return q
            prev_dq = dq
            prev_e = e
        end = time.time()
        print("\nCheck # of Iterations:", int((i + 1) * check))
        print("Current Error:", "{:e}".format(e))
        print("Q:",
              np.array2string(q, formatter={
                  'float_kind': '{0:.3f}'.format}))
        print("Cycle Runtime:", end - start)

    raise Exception("Maximum iterations exceeded.")


def error(data, f, q):
    e = 0
    err_form = np.subtract(data[1], f(data[0], q))
    e = np.sum(np.multiply(err_form,err_form))
    # for i in range(len(data[0])):
    #     e += (data[1][i] - f(data[0][i], q)) ** 2
    return e / 2


def compute_partials(data, f, q,
                     j):  # limit definition: f(x,y,z) = f(x,y+h,z) - f(x,y,z)/h
    q[j] += 2 * STEP  # 2h
    x = data[0]
    v1 = f(x, q)
    q[j] -= STEP  # h
    v2 = f(x, q)
    q[j] -= 2 * STEP  # -h
    v3 = f(x, q)
    q[j] -= STEP
    v4 = f(x, q)
    return (-v1 + 8 * v2 - 8 * v3 + v4) / (12 * STEP)


def compute_deltas(data, f, q):
    dq = np.zeros(len(q))
    for j in range(len(q)):
        partials = compute_partials(data, f, q, j)
        eqj = np.sum(np.multiply(np.subtract(data[1], f(data[0], q)), partials))
        dq[j] = LAMBDA * eqj
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


def main():
    # plt.figure("Function and Partials plot", figsize=(8, 8))
    # x_bound = [-7, 7]
    # y_bound = [-5, 5]
    # plt.xlim(x_bound[0], x_bound[1])
    # plt.ylim(y_bound[0], y_bound[1])
    # plt.xlabel("time")
    # plt.ylabel("values")
    # plot_axes(x_bound, y_bound)
    #
    # times = np.linspace(-10, 10, 100)
    # q = [1, 1, 0, 0]
    # plt.plot(times, sine(times, q))
    # plt.scatter(times, sine(times, q))
    # partial_list = []
    # for t in times:
    #     partial_list.append(partial(sine, t, q, 1))
    # plt.plot(times, partial_list)
    # plt.scatter(times, partial_list)

    plt.figure("Data plot", figsize=(8, 8))
    data = read_data("input.txt")
    plt.scatter(data[0], data[1])
    q = [1, 10, 2, 3]
    q = generate_least_squares(data, gaussian, q)
    plt.scatter(data[0], gaussian(data[0], q), color='r')

    plt.show()

    # print(cubic(np.array([1, 2, 3]), [1, 0, 0, 0]))


if __name__ == '__main__':
    main()
