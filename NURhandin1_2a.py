import numpy as np
import sys
import os
import matplotlib.pyplot as plt

data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),comments='#',dtype=np.float64)
x=data[:,0]
y=data[:,1]
xx=np.linspace(x[0],x[-1],1000) #x values to interpolate at

#2A

V = np.zeros((20,20))

for i in range(len(V[0])):
    for j in range(len(V[0])):
        V[i][j] = x[i]**(j)
        
def crout(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return "Error: not a square matrix. No solution"
    L = np.zeros(shape=matrix.shape)
    U = np.copy(L)
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if i == j:
                L[i][j] = 1
            if i <= j:
                U[i][j] = matrix[i][j]
                for k in range(len(matrix[0]-1)):
                    if k < i:
                        U[i][j] = U[i][j] - L[i][k]*U[k][j]
            if i > j:
                L[i][j] = 1/U[j][j] * matrix[i][j]
                for k in range(len(matrix[0]-1)):
                    if k < j and U[j][j] != 0:
                        L[i][j] = L[i][j] - 1/U[j][j] * L[i][k]*U[k][j]
    return L,U

def forward_sub(lower,b):
    y = np.zeros(b.shape)
    y[0] = b[0] / lower[0][0]
    for i in range(len(y)):
        y[i] = 1/lower[i][i] * b[i]
        for j in range(len(y)-1):
            if j < i: 
                y[i] = y[i] - 1/lower[i][i] * lower[i][j]*y[j]
    
    return y

def backward_sub(upper,y):
    x = np.zeros(y.shape)
    N_1 = len(x)-1 #define the index N-1, shorter than len(x)-1
    x[N_1] = y[N_1] / upper[N_1][N_1]
    for i in range(len(x)-1,-1,-1):
        x[i] = 1/upper[i][i] * y[i]
        for j in range(len(x)-1,-1,-1):
            if j > i: #ensures you start at j=i+1
                x[i] = x[i] - 1/upper[i][i] * upper[i][j]*x[j]
    return x 

def LU_decomp(matrix,b):
    L = crout(matrix)[0]
    U = crout(matrix)[1]
    y = forward_sub(L,b)
    return backward_sub(U,y)

c = LU_decomp(V,y)
print(c)

y_polynomial = np.zeros(len(xx))
y_diff = np.zeros(len(y))

for i in range(len(y_polynomial)):
    for j in range(len(c)):
        y_polynomial[i] = y_polynomial[i] + c[j]*(xx[i]**j)
    if i % 50 == 0:
        y_diff[int(i/50)] = np.abs(y_polynomial[i] - y[int(i/50)])

fig,ax=plt.subplots()
ax.plot(x,y,marker='o',linewidth=0)
ax.plot(xx,y_polynomial)
plt.xlim(-1,101)
plt.ylim(-400,400)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()

fig,ax=plt.subplots()
ax.plot(x,y_diff)
plt.ylim(0,150)
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.show()

#2B

