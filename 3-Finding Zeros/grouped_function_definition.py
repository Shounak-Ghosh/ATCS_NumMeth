import matplotlib.pyplot as plt
import numpy as np
import math

color_list = ['r', 'b', 'g', 'y', 'm', 'k', 'c', '#ff007f']


def function_list(x):
    return [x ** 2, 1 + x ** 2, math.exp(-x ** 2), x * math.exp(-x ** 2),
            x * math.exp(x ** 2), x * math.exp(-x), x * math.exp(x),
            math.sqrt(abs(x))]


def generate_function_point_list():
    time_points = np.linspace(-10, 10, 200)
    point_list = []
    for t in time_points:
        vals = function_list(t)
        vals.insert(0, t)
        point_list.append(vals)
    return point_list


def plot_functions(point_list):
    num_functions = len(point_list[0]) - 1
    time_list = []
    val_list = []
    for i in range(num_functions):
        val_list.append(list())

    for point in point_list:
        time_list.append(point[0])
        for i in range(1, num_functions + 1):
            val_list[i - 1].append(point[i])

    for i in range(num_functions):
        plt.scatter(time_list, val_list[i], color=color_list[i])
        plt.plot(time_list, val_list[i], color=color_list[i])
        plt.draw()


def main():
    # Set the plot boundaries
    plt.figure("Functions", figsize=(8, 8))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("time")
    plt.ylabel("values")
    plot_functions(generate_function_point_list())
    plt.show()


if __name__ == '__main__':
    main()
