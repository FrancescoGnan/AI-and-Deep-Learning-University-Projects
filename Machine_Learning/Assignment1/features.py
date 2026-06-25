import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def phi1(x, y):
    return x**2 + y**2

def phi2(x, y):
    r = np.sqrt(x**2 + y**2)
    result = np.sin(r)
    result[r != 0] *= np.sin(r[r != 0]) / r[r != 0]
    return result

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

Z1 = phi1(X, Y)
Z2 = phi2(X, Y)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap='rainbow')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('$\phi_1(x, y)$')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap='rainbow')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('$\phi_2(x, y)$')
plt.show()
