import matplotlib.pyplot as plt
import numpy as np



def main():
    """
    Main function: Generates random data for a cubic function and plots it
    """
    q = np.random.randint(-10, high=10, size=4)
    x = np.arange(0, 40, 1)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = q[0] * x[i] ** 3 + q[1] * x[i] ** 2 + q[2] * x[i] + \
            q[3] + (np.random.randint(1) * 2 - 1) * np.random.rand(1) * 1e4
    
    print("Q: ", q)
    plt.figure("Raw Data Plot", figsize=(8, 8))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, label="Raw Data", color='b')
    plt.show()

    #save data to txt file
    np.savetxt("cubic_data.txt", np.column_stack((x, y)), delimiter=" ")


if __name__ == "__main__":
    main()
