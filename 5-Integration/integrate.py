import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    return 1 + x**2

def f2(x):
    return x * np.exp(-x ** 2)

def f3(x):
    return x * np.exp(-x)

def f4(x):
    return np.sin(x)

#trapezoidal rule
#TODO Check if this is actually correct
def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    sum = np.sum(f(x))
    return h * (f(a) + f(b) + 2 * sum) / 2

#simpson's rule
#TODO Check if this is actually correct
def simpson(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    sum = 0
    for i in range(1, n):
        if i % 2 == 0:
            sum += 2 * f(x[i])
        else:
            sum += 4 * f(x[i])
    return h / 3 * (f(a) + f(b) + sum)


def main():
    x = np.linspace(-2, 2, 1000)
    plt.plot(x, f1(x), label='f1')
    plt.plot(x, f2(x), label='f2')
    plt.plot(x, f3(x), label='f3')
    plt.plot(x, f4(x), label='f4')
    plt.legend()
    plt.show()

#run main
if __name__ == '__main__':
    main()
