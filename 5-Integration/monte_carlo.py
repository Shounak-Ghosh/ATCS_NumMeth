import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    plt.figure("Monte Carlo Plot", figsize=(5, 5))

    # plot circle
    t = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    plt.plot(x, y, 'k-')
    

    #generate random points
    x = np.random.uniform(-1, 1, 1000)
    y = np.random.uniform(-1, 1, 1000)
    num_in_circle = 0

    for i in range(len(x)):
        if x[i]**2 + y[i]**2 <= 1:
            plt.scatter(x[i], y[i], color='red')
            num_in_circle += 1
        else:
            plt.scatter(x[i], y[i], color='blue')
        plt.draw()
        plt.pause(1e-7)

    print("Pi (approximation): ", 4 * num_in_circle / len(x))
    
    plt.show()

    

# main
if __name__ == "__main__":
    main()