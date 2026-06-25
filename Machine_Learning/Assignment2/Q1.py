import numpy as np

X = np.array([[0.6, -1.0], [0.8, -1.0], [-0.4, 0.9], [0.2, 0.0]])
t = np.array([-0.8, -0.1, 0.9, 0.7])
W1 = np.array([[-0.8, -0.7, 0.6], [-1.0, 0.5, -1.0]])
b1 = np.array([-0.2, -1.0, -0.7])
w2 = np.array([0.1, -1.0, 0.5])
b2 = -0.7


def f(u):
    return np.maximum(0, u)

def df(u):
    return np.where(u > 0, 1, 0)


u = np.dot( X, W1 ) + b1
y = np.dot(f(u), w2) + b2
L = 0.5 * np.sum( ( y - t )**2 )
print('Loss: \n',L)
print('----------------------')
dL_dy = y - t
dy_dw2 = f(u)
dL_dw2 = np.dot(dL_dy, dy_dw2)

print('dL_dw2: \n',dL_dw2)
print('----------------------')

dL_db2 = np.sum( dL_dy )
print('dL_db2: \n',dL_db2)
print('----------------------')

df_du = df(u)

delta1 = dL_dy[:,None] * w2
delta2 = delta1 * df_du

dL_dW1 = np.dot(X.T,delta2)
print('dL_dW1: \n',dL_dW1)
print('----------------------')

#print(np.sum(delta2[:,2]))
dL_db1 = np.sum(delta2, axis=0)
print('dL_db1: \n', dL_db1)

