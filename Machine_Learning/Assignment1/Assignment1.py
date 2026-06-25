import numpy as np
import matplotlib.pyplot as plt
import math

num_points = 1000
radius = 2.0
noise_stddev = 0.1

theta = np.linspace(0, 2 * np.pi, num_points)
x = radius * np.cos(theta) + np.random.normal(0, noise_stddev, num_points)
y = radius * np.sin(theta) + np.random.normal(0, noise_stddev, num_points)

theta = np.linspace(0, 2 * np.pi, num_points-100)
x2 = (radius-1) * np.cos(theta) + np.random.normal(0, noise_stddev, num_points-100)
y2 = (radius-1) * np.sin(theta) + np.random.normal(0, noise_stddev, num_points-100)

theta = np.linspace(0, 2 * np.pi, num_points-700)
x3 = (radius-1.9) * np.cos(theta) + np.random.normal(0, noise_stddev, num_points-700)
y3 = (radius-1.9) * np.sin(theta) + np.random.normal(0, noise_stddev, num_points-700)

plt.figure(figsize=(10, 10))
plt.scatter(x, y, marker='o', color='b', label='Class C1')
plt.scatter(x2, y2, marker='o', color='y', label='Class C2')
plt.scatter(x3, y3, marker='o', color='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Circularly Distributed Data with Gaussian Noise')
plt.legend()
plt.grid(True)
plt.show()

# new features:

def phi1(x,y):
    return x**2 + y**2

def phi2(x,y):
    r = np.sqrt(x**2 + y**2)
    result = np.sin(r)
    result[r != 0] *= np.sin(r[r != 0]) / r[r != 0]
    return result


plt.figure(figsize=(10, 10))
plt.scatter(phi1(x,y), phi2(x,y), marker='o', color='b', label='Class C1')
plt.scatter(phi1(x2,y2), phi2(x2,y2), marker='o', color='y', label='Class C2')
plt.scatter(phi1(x3,y3), phi2(x3,y3), marker='o', color='b')
plt.plot([x for x in range(4)], [0.09*x+0.4 for x in range(4)] , '--', color='r', label='Hyperplane')
plt.xlabel('$\phi_1$')
plt.ylabel('$\phi_2$')
plt.title('Linearly separable classes in feature space')
plt.legend(loc=(0.8, 0.7))
plt.grid(True)
plt.show()
