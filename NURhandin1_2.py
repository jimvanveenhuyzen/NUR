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
                y[i] = y[i] - LU[i][j]*y[j] #scale solution using row reduction
    return y 

def backward_substitution(LU,y): #applies backward sub
    x = np.zeros(y.shape)
    N_1 = len(x)-1 #define the largest index 
    x[N_1] = y[N_1] / LU[N_1][N_1] #take the maximum index of LU 
    for i in range(len(x)-1,-1,-1):
        x[i] = 1/LU[i][i] * y[i] #scale the values 
        for j in range(len(x)-1,-1,-1):
            if j > i:
                x[i] = x[i] - 1/LU[i][i] * LU[i][j]*x[j] #row reduction
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
ax.plot(x,y,marker='o',linewidth=0,label='Data',zorder=10)
ax.plot(xx,y_polynomial,label='Polynomial (LU decomp)',zorder=1)
plt.xlim(-1,101)
plt.ylim(-400,400)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.title('2a: comparing data with y(x) from LU decomp')
plt.legend()
plt.show()

fig,ax=plt.subplots() #plot absolute error as a function of the data 
ax.scatter(x,y_diff_LU)
plt.yscale('log') #important: use log plot to see difference 
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.title('2a: absolute error as a function of data x_i')
plt.show()

#Problem 2B

def bisection(x,sample,order): #bisection to find nearest points for x 
    if x < sample[0]: #extrapolation: if x < smallest sample value, j_low=0
        return 0 
    if x > sample[-1]: #extrapolation: if x > largest sample value, j_low=N-2
        return (len(sample)-1)-order
    start = 0 #start at the first index 
    end = len(sample)-1 #end at the last index 
    size = (end-start)*0.5 #calculate the 'average', or rather middle index
    while np.abs(start-end) > 1: #while the diff of indices > given error 
        if x <= sample[int(start+size)]: #if in left half, decrease end index 
            end = end - size
        else: #if in right half, increase start index 
            start = start + size
        size = (end-start)*0.5 #update for a new middle index 
    if int(start) - order < 0: #to prevent falling out of range for our order
        return 0 #if floor(j_low) is smaller than the order, return 0 
    elif int(start) + order > len(sample)-1: #fall out of range for given order
        return (len(sample)-1)-order #return (max index-order) if idx too large 
    return int(start)#return the j_low index

def neville(x,xdata,ydata,order): #neville: enter x to find interp y
    j_low = bisection(x,xdata,order) #append bisection alg. to find j_low 
    M_1 = order #we use M-1 orders for the algorithm
    P = np.copy(ydata[j_low:j_low+M_1]) #range is [j_low,j_low + M-1]
    xdata = np.copy(xdata[j_low:j_low+M_1]) #get x and y copies for given range
    for k in range(1,M_1): 
        for i in range(0,M_1-k): #nested loop through P[i]
            P[i] = ((xdata[i+k] - x) * P[i] + (x - xdata[i]) * P[i+1]) \
                / (xdata[i+k] - xdata[i]) #apply H(x) polynomial 
    return P[0] #first value should be the interpolated solution

y_interp = np.zeros(len(xx))
for i in range(len(xx)): #again interpolate 1000 y values from xx array 
    y_interp[i] = neville(xx[i],x,y,20) #fill array with interp y values
    
y_neville = np.zeros(len(y))
for i in range(len(y)): #only for the 20 data points to compare LU and Neville
    for j in range(len(c)):
        y_neville[i] = neville(x[i],x,y,20)
                                     
y_diff_neville = np.abs(y_neville - y) #difference Neville interp and data 

fig,ax=plt.subplots() #plot the data vs Lagrange polynomial from Neville's algo
ax.plot(x,y,marker='o',linewidth=0, label='Data',zorder=10)
ax.plot(xx,y_interp, label='Polynomial (Neville interpolation)',zorder=1)
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
plt.yscale('log') #log plot 
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.title('2b: comparing abs error of the two methods')
plt.legend(loc='upper left')
plt.show()

#Problem 2C

def matrix_multiplication(A,x): #function to multiply square matrix with vector
    output = np.zeros(len(A))
    for i in range(len(A)):
        for j in range(len(A)):
            output[j] = output[j] + A[j][i] * x[i]
    return output #matrix A size: NxN, vector x size: Nx1, returns Nx1 solution

def LU_iterations(A,b,number): #performs multiple iterations of LU decomp
    if number == 0:
        return "No iterations performed"
    improved_sol = LU(A,b) #first iterations of LU 
    for i in range(number-1): #for each iteration, find the error and subtract
        A_times_x = matrix_multiplication(A,improved_sol) #find Ax' = b
        delta_b = A_times_x - b #delta b = Ax' - b
        error = LU(A,delta_b) #A delta x = delta b
        improved_sol = LU(A,b) - error #x" = x' - delta x 
    return improved_sol #returns x^(number of iterations)

c_10it = LU_iterations(V,y,10) #find the c values from 10 iterations 
y_polynomial_10it = np.zeros(len(xx)) 
for i in range(len(y_polynomial_10it)): #interpolate 1000 values as accordingly
    for j in range(len(c_10it)):
        y_polynomial_10it[i] = y_polynomial_10it[i] + c_10it[j]*(xx[i]**j)
        
y_LU10it = np.zeros(len(y))
for i in range(len(y)): #interpolate 20 data values to compare with 1 LU decomp
    for j in range(len(c)):
        y_LU10it[i] = y_LU10it[i] + c_10it[j]*(x[i]**j)
       
y_diff_LU10it = np.abs(y_LU10it - y) #find difference 10 LU iter. vs the data 
 
fig,ax=plt.subplots() #plot the data with polynomials from 1 LU and 10 LU iter.
ax.plot(x,y,marker='o',linewidth=0,label='data',zorder=0)
ax.plot(xx,y_polynomial,label='1 LU iterations')
ax.plot(xx,y_polynomial_10it,label='10 LU iterations')
plt.xlim(-1,101)
plt.ylim(-400,400)
plt.legend()
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.title('2c: comparing y(x) with data for 1 and 10 LU iterations')
plt.show()

fig,ax=plt.subplots() #plot absolute error compared to data for 1 LU and 10 LU
ax.scatter(x,y_diff_LU, label='LU (1 iteration)')
ax.scatter(x,y_diff_LU10it, label='LU (10 iterations)')
plt.yscale('log')
ax.set_xlabel('$x$')
ax.set_ylabel('$|y(x)-y_i|$')
plt.legend(loc='upper left')
plt.title('2c: comparing abs error of 1 and 10 LU iterations')
plt.show()

#Problem 2D: I remove some earlier comments from the problems for readability

import timeit #import the timeit module to time how fast the code runs 

begin_2a = timeit.default_timer() #start timing runtime of problem 2a

for k in range(300): #use 300 iterations 
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
    
averagetime_2a = (timeit.default_timer() - begin_2a)/300 #divide by # of iter
print('The time taken for 2a is:', np.around(averagetime_2a,3), 's')

begin_2b = timeit.default_timer() #start timing runtime of problem 2b

for k in range(50): #use 50 iterations 
    y_interp = np.zeros(len(xx))
    for i in range(len(xx)):
        y_interp[i] = neville(xx[i],x,y,20)
        
    y_neville = np.zeros(len(y))
    for i in range(len(y)): #only for 20 data points to compare LU and Neville
        for j in range(len(c)):
            y_neville[i] = neville(x[i],x,y,20)
                                         
    y_diff_neville = np.abs(y_neville - y) #difference Neville interp and data 
        
averagetime_2b = (timeit.default_timer() - begin_2b)/50 
print('The time taken for 2b is:', np.around(averagetime_2b,3), 's')

begin_2c = timeit.default_timer()

for k in range(200): #use 200 iterations 
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
    
averagetime_2c = (timeit.default_timer() - begin_2c)/200
print('The time taken for 2c is:', np.around(averagetime_2c,3), 's')


