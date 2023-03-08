import numpy as np
import sys
import os
import matplotlib.pyplot as plt

#Problem 2A

data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),\
                   comments='#',dtype=np.float64)
x=data[:,0]
y=data[:,1]
xx=np.linspace(x[0],x[-1],1000) #x values to interpolate at

V = np.zeros((20,20)) #20x20 matrix 

for i in range(len(V[0])):
    for j in range(len(V[0])):
        V[i][j] = x[i]**(j) #create Vandermonde matrix using x values from data
                
def improved_crout(matrix): #inspired by slide 15 lecture 3, Impr. Crout algo 
    if matrix.shape[0] != matrix.shape[1]: #if # of rows != # of columns
        return "Error: not a square matrix. No solution"
    LU = np.copy(matrix) #start with matrix, adjust elements accordingly
    index_max = np.zeros(len(matrix[0]), dtype=int) #max index per column k
    for k in range(len(matrix[0])): 
        index_max[k] = int(k)
        for i in range(len(matrix[0])):
            if i >= k: #loop over rows i >= k 
                if np.abs(matrix[i][k]) > np.abs(matrix[int(index_max[k])][k]): 
                    index_max[k] = i #set max pivot candidate i_max here
        if index_max[k] != k: #if i_max != k, then loop over columns j 
            for j in range(len(matrix[0])): 
                LU[index_max][j] = LU[k][j] #replace LU elements accordingly
        for i in range(len(matrix[0])):
            if i > k: #loop over rows i
                LU[i][k] = LU[i][k] / LU[k][k] #replace LU elements
                for j in range(len(matrix[0])):
                    if j > k: #loop over columns j
                        LU[i][j] = LU[i][j] - LU[i][k]*LU[k][j] #replace LU el.
    return LU #returns the decomposed LU matrix

def forward_substitution(LU,b): #applies forward sub, slide 11 lecture 3  
    y = np.zeros(b.shape)
    for i in range(len(y)):
        y[i] = b[i]
        for j in range(len(y)-1):
            if j < i: 
                y[i] = y[i] - LU[i][j]*y[j]
    return y 

def backward_substitution(LU,y): #applies backward sub
    x = np.zeros(y.shape)
    N_1 = len(x)-1
    x[N_1] = y[N_1] / LU[N_1][N_1]
    for i in range(len(x)-1,-1,-1):
        x[i] = 1/LU[i][i] * y[i]
        for j in range(len(x)-1,-1,-1):
            if j > i:
                x[i] = x[i] - 1/LU[i][i] * LU[i][j]*x[j]
    return x 

def LU(matrix,sol): #combines multiple functions to solve for input A and b. 
    LU = improved_crout(matrix) #gets LU matrix for matrix A
    y = forward_substitution(LU,sol) #returns y from (LU)b = y
    return backward_substitution(LU,y) #returns the solution x from (LU)x = y

c = LU(V,y) #finds solutions c for matrix V and vector y 
print('The 20 c_j values are given by the following matrix:\n\n',c)

y_polynomial = np.zeros(len(xx)) 
y_LU = np.zeros(len(y))

for i in range(len(y_polynomial)): #interpolate 1000 y values from xx array 
    for j in range(len(c)): #for 1000 values, sum 20 j-values 
        y_polynomial[i] = y_polynomial[i] + c[j]*(xx[i]**j) #eq. 2 of hand in 1
        
for i in range(len(y)): #only obtain interp y values for the data x array 
    for j in range(len(c)):
        y_LU[i] = y_LU[i] + c[j]*(x[i]**j)
       
y_diff_LU = np.abs(y_LU - y) #compute the abs difference between LU and data

fig,ax=plt.subplots() #plotting the data vs the Lagrange polynomial from LU
ax.plot(x,y,marker='o',linewidth=0,label='Data')
ax.plot(xx,y_polynomial,label='Polynomial (LU)')
plt.xlim(-1,101)
plt.ylim(-400,400)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.title('2a: comparing data with y(x) from LU decomp')
plt.legend()
plt.show()

fig,ax=plt.subplots() #plot absolute error as a function of the data 
ax.scatter(x,y_diff_LU)
plt.ylim(-0.02,0.4)
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.title('2a: absolute error as a function of data x_i')
plt.show()

#Problem 2B

def neville(x,xdata,ydata): #function to run neville's algorithm
    n = len(xdata) #order we use is M-1, e.g. 20 sample points this is order 19 
    P = np.copy(ydata)
    for k in range(1,n):
        for i in range(0,n-k):
            P[i] = ((xdata[i+k] - x) * P[i] + (x - xdata[i]) * P[i+1])\
                / (xdata[i+k] - xdata[i])
    return P[0] #solution is stored in P[0]: equiv. to interpolated value of y 

y_interp = np.zeros(len(xx))
for i in range(len(xx)): #again interpolate 1000 y values from xx array 
    y_interp[i] = neville(xx[i],x,y) #fill array directly with interp y values
    
y_neville = np.zeros(len(y))
for i in range(len(y)): #only for the 20 data points to compare LU and Neville
    for j in range(len(c)):
        y_neville[i] = neville(x[i],x,y)
                                     
y_diff_neville = np.abs(y_neville - y) #difference Neville interp and data 
    
fig,ax=plt.subplots() #plot the data vs Lagrange polynomial from Neville's algo
ax.plot(x,y,marker='o',linewidth=0, label='Data')
ax.plot(xx,y_interp, label='Polynomial (Neville interp)')
plt.xlim(-1,101)
plt.ylim(-400,400)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.title('2b: comparing data with y(x) from Neville interpolation')
plt.legend()
plt.show()

fig,ax=plt.subplots() #compare the error between LU decomp and Neville's algo
ax.scatter(x,y_diff_LU, label='LU decomp')
ax.scatter(x,y_diff_neville, label='Neville')
plt.ylim(-0.02,0.4)
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.title('2b: comparing the error of the two methods')
plt.legend(loc='upper left')
plt.show()

#Problem 2C

def matrix_multiplication(A,x):
    output = np.zeros(len(A))
    for i in range(len(A)):
        for j in range(len(A)):
            output[j] = output[j] + A[j][i] * x[i]
    return output

def LU_iterations(A,b,number):
    if number == 0:
        return "No iterations performed"
    improved_sol = LU(A,b)
    for i in range(number-1):
        A_times_x = matrix_multiplication(A,improved_sol)
        delta_b = A_times_x - b
        error = LU(A,delta_b)
        improved_sol = LU(A,b) - error
    return improved_sol


A_times_x = matrix_multiplication(V,LU(V,y))
delta_y = A_times_x - y
error = LU(V,delta_y)
#print(c)
#print(error)
new_c = LU(V,y) - error
#print(new_c)
#print(LU_iterations(V,y,10))

#print(c)
c_10it = LU_iterations(V,y,10)
#print(c_10it)
y_polynomial_10it = np.zeros(len(xx))

for i in range(len(y_polynomial_10it)):
    for j in range(len(c_10it)):
        y_polynomial_10it[i] = y_polynomial_10it[i] + c_10it[j]*(xx[i]**j)
        
y_LU10it = np.zeros(len(y))

for i in range(len(y)):
    for j in range(len(c)):
        y_LU10it[i] = y_LU10it[i] + c_10it[j]*(x[i]**j)
       
y_diff_LU10it = np.abs(y_LU10it - y)

"""
fig,ax=plt.subplots()
ax.plot(x,y,marker='o',linewidth=0,label='data')
ax.plot(xx,y_polynomial,label='1 iter')
ax.plot(xx,y_polynomial_10it,label='10 iter')
plt.xlim(-1,101)
plt.ylim(-400,400)
plt.legend()
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()
"""

fig,ax=plt.subplots()
ax.scatter(x,y_diff_LU, label='LU (1 iteration)')
ax.scatter(x,y_diff_LU10it, label='LU (10 iterations)')
plt.ylim(-0.02,0.4)
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.legend(loc='upper left')
plt.show()


#Problem 2D: I remove earlier comments from the problems for readability

import timeit #import the timeit module to time how fast the code runs 

begin_2a = timeit.default_timer() #start timing runtime of problem 2a

for k in range(50): #100 iterations 
    c = LU(V,y)
    y_polynomial = np.zeros(len(xx))
    for i in range(len(y_polynomial)):
        for j in range(len(c)):
            y_polynomial[i] = y_polynomial[i] + c[j]*(xx[i]**j)
            
    y_LU = np.zeros(len(y))        
    for i in range(len(y)): 
        for j in range(len(c)):
            y_LU[i] = y_LU[i] + c[j]*(x[i]**j)
           
    y_diff_LU = np.abs(y_LU - y)
    
averagetime_2a = (timeit.default_timer() - begin_2a)/50
print('time taken for 2a', averagetime_2a, 's')

begin_2b = timeit.default_timer() #start timing runtime of problem 2b

for k in range(20): #10 iterations 
    y_interp = np.zeros(len(xx))
    for i in range(len(xx)):
        y_interp[i] = neville(xx[i],x,y)
        
    y_neville = np.zeros(len(y))
    for i in range(len(y)): #only for the 20 data points to compare LU and Neville
        for j in range(len(c)):
            y_neville[i] = neville(x[i],x,y)
                                         
    y_diff_neville = np.abs(y_neville - y) #difference Neville interp and data 
        
averagetime_2b = (timeit.default_timer() - begin_2b)/20
print('time taken for 2b', averagetime_2b, 's')

begin_2c = timeit.default_timer()

for k in range(50): #10 iterations
    c_10it = LU_iterations(V,y,10)
    y_polynomial_10it = np.zeros(len(xx))
    for i in range(len(y_polynomial_10it)):
        for j in range(len(c_10it)):
            y_polynomial_10it[i] = y_polynomial_10it[i] + c_10it[j]*(xx[i]**j)
            
    y_LU10it = np.zeros(len(y))
    for i in range(len(y)):
        for j in range(len(c)):
            y_LU10it[i] = y_LU10it[i] + c_10it[j]*(x[i]**j)
           
    y_diff_LU10it = np.abs(y_LU10it - y)
    
averagetime_2c = (timeit.default_timer() - begin_2c)/50
print('time taken for 2c', averagetime_2c, 's')


