import numpy as np

x = y = z = np.linspace(-10.0, 10.0, 100)
np.savetxt('test.out', x, delimiter=',')  # X is an array
# np.savetxt('test.out', (x, y, z))  # x,y,z equal sized 1D arrays
# np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation
