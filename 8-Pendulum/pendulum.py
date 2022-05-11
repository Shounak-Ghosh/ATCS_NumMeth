"""
Pendulum Lab
@author Shounak Ghosh
@version 5.11.2022
"""
import numpy as np
import matplotlib.pyplot as plt

# modeling a simple pendulum
g = 9.8
L = 1
A = 0.7
PHI = 0
OMEGA = 3.16


def theta(A, omega, phi, t):
    """
    Theta function for a simple pendulum
    @param A: amplitude
    @param omega: angular frequency
    @param phi: phase shift
    @param t: time
    @return: theta value at time t
    """
    return A*np.sin(omega*t+phi)


def ang_vel(A, omega, phi, t):
    """
    Angular velocity of a simple pendulum
    @param A: amplitude
    @param omega: angular frequency
    @param phi: phase shift
    @param t: time
    @return: angular velocity at time t
    """
    return omega*A*np.cos(omega*t+phi)


def euler_ang_vel(A, omega, phi, t, dt):
    """
    Euler method for angular velocity
    @param A: amplitude
    @param omega: angular frequency
    @param phi: phase shift
    @param t: time
    @param dt: time step
    @return: angular velocity at time t + dt
    """
    return ang_vel(A, omega, phi, t) - g/L*np.sin(theta(A, omega, phi, t)) * dt


def euler_theta(A, omega, phi, t, dt):
    """
    Euler method for theta
    @param A: amplitude
    @param omega: angular frequency
    @param phi: phase shift
    @param t: time
    @param dt: time step
    @return: theta at time t + dt
    """
    return theta(A, omega, phi, t) - ang_vel(A, omega, phi, t) * dt

def euler_cromer_theta(A, omega, phi, t, dt):
    """
    Euler-Cromer method for theta
    @param A: amplitude
    @param omega: angular frequency
    @param phi: phase shift
    @param t: time
    @param dt: time step
    @return: theta at time t + dt
    """
    return theta(A, omega, phi, t) + ang_vel(A, omega, phi, t + dt) * dt


def main():
    """
    Main function
    """
    n = 1000
    # initializing theta and angular velocity
    theta_0 = 0
    ang_vel_0 = 0
    # initializing time
    t = 0
    # initializing arrays
    time = np.linspace(0, 10, n)
    theta_arr = np.zeros(n)
    closed_theta = np.zeros(n)
    ang_vel = np.zeros(n)
    # initializing time step
    dt = time[1] - time[0]

    # initializing theta and angular velocity
    theta_arr[0] = theta_0
    ang_vel[0] = ang_vel_0
    # iterating over time
    for i in range(n):
        # appending theta and angular velocity to the arrays
        closed_theta[i] = theta(A, OMEGA, PHI, time[i])
        theta_arr[i] = euler_cromer_theta(A, OMEGA, PHI, time[i], dt)
        ang_vel[i] = euler_ang_vel(A, OMEGA, PHI, time[i], dt)

    
    # plotting theta and angular velocity
    plt.figure("Pendulum Plot")
    plt.xlabel("Time")
    plt.ylabel("Theta")
    # for i in range(n):
    #     plt.scatter(time[i], theta_arr[i], color='b')
    #     plt.draw()
    #     plt.pause(.001)

    # plt.plot(time, closed_theta, color='r')
    plt.plot(time, theta_arr, color='b')
    plt.plot(time, ang_vel, color='g')
    plt.draw()


    plt.show()


# run main
if __name__ == "__main__":
    main()
