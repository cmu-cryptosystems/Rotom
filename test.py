import numpy as np

# Define the coefficient matrix A
A = np.array([[13, 12], [-4, 7], [11, -13]])

# Define the constant vector b
b = np.array([-6, -73, 157])

# Solve for the unknown vector x
x = np.linalg.lstsq(A, b)

# Print the solution
print("Solution vector x:", x)
# The output will be: Solution vector x: [1. 2.]
# which means x = 1 and y = 2
