"""
Monte Carlo Lab
@author Shounak Ghosh
@version 5.09.2022
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse


def monte_carlo(dimensions, num_iterations):
    """
    Monte Carlo approximation
    @param dimensions: number of dimensions
    @param num_iterations: number of iterations
    """
    N = np.random.uniform(-1, 1, (dimensions, num_iterations))
    num_in_circle = 0

    for i in range(num_iterations):
        if np.linalg.norm(N[:, i]) <= 1:
            num_in_circle += 1

    print("Ratio:", num_in_circle / num_iterations)

    # random number histogram
    plt.figure("Random Number Histogram", figsize=(5, 5))
    plt.hist(N[0], bins=10)
    plt.show()


def plot_monte_carlo():
    """
    Plots Monte Carlo approximation and 3D histogram for two dimensions
    """
    plt.figure("Monte Carlo Plot", figsize=(5, 5))

    # plot circle
    t = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    plt.plot(x, y, 'k-')

    # generate random points
    num_iter = int(1e6)
    N = np.random.uniform(-1, 1, (2, num_iter))
    num_in_circle = 0

    for i in range(num_iter):
        curr = N[:, i]
        if  np.linalg.norm(curr) <= 1:
            plt.scatter(curr[0], curr[1], color='red')
            num_in_circle += 1
        else:
            plt.scatter(curr[0], curr[1], color='blue')
        plt.draw()
        plt.pause(1e-7)

    print("Ratio:", num_in_circle / num_iter)
    print("Pi (Approximation):", 4 * num_in_circle / num_iter)

    # random number histogram
    fig = plt.figure("Random Number Histogram", figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    hist, xedges, yedges = np.histogram2d(N[0], N[1], bins=6, range=[[-1, 1], [-1, 1]])
    # Construct arrays for the anchor positions 
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the bars
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    
    plt.show()

def main():
    # monte_carlo(3, int(1e6))
    plot_monte_carlo()


# main
if __name__ == "__main__":
    main()
