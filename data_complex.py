
import numpy as np

# Data for f(x) = sqrt(x^2 - 16)
# Includes real and complex values
x = np.array([-5, -4, 0, 4, 5])
# f(-5) = sqrt(25-16) = 3
# f(-4) = 0
# f(0) = sqrt(-16) = 4i
# f(4) = 0
# f(5) = 3
y = np.array([3, 0, 4j, 0, 3])
