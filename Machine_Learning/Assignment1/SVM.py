import matplotlib.pyplot as plt
import numpy as np
import math

def hyperplane(m,q,x):
    return x*m + q

def dist(m,q,x0,y0):
    return abs( -m*x0 + y0 -q ) / math.sqrt( m**2 + 1 )

def parallel(m,y1,x1,x):
    return y1 + m * (x - x1)
    
x = np.linspace(-1,2,100)

x1 = [1,2,2]
x2 = [1,2,0]
y1 = [1,-1,0]
y2 = [-1,0,1]

pt1 = [ 1, 1 ]
pt2 = [ 1, -1]
m = -4
q = 4


plt.figure()
plt.scatter(x1,x2,marker='o',color='r', label='Class +')
plt.scatter(y1,y2,marker='s',color='b', label='Class -')
plt.plot(x,[hyperplane(m,q,i) for i in x], '--', color='k',label='Hyperplane')
plt.plot(x,[parallel(m,pt1[1],pt1[0],i) for i in x], '-.', color='grey',label='Margin')
plt.plot(x,[parallel(m,pt2[1],pt2[0],i) for i in x], '-.', color='grey')
plt.fill_between(x,[hyperplane(m,q,i) for i in x],[parallel(m,pt1[1],pt1[0],i) for i in x], interpolate=True, color='yellow', alpha=0.5)
plt.fill_between(x,[hyperplane(m,q,i) for i in x],[parallel(m,pt2[1],pt2[0],i) for i in x], interpolate=True, color='yellow', alpha=0.5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()

d1 = dist(m,q,pt1[0],pt1[1])
d2 = dist(m,q,pt2[0],pt2[1])

print(d1)
print(d2)