import numpy as np

def ln_factorial(k): #compute the natural log of k! for an integer k 
    value = 0.0 #if k=0, we return 0.0 as 0! = 1 (and ln(1) = 0)
    for i in range(1,k+1): #k+1 is not included
        value += np.float32(np.log(i)) #sum values ln(1)+ln(2)+..+ln(k)
    return value #returns sum of all values of ln(i) within [1,k]

def ln_poisson(lambdas,k):
    return np.int32(k)*np.float32(np.log(lambdas)) - np.float32(lambdas) - \
        np.float32(ln_factorial(k)) #transforming to log space
        
def poisson(lambdas,k): #in this function we take exponents of the ln() values
    return np.float32(np.exp(ln_poisson(lambdas,k)))

print("The value of poisson(lambda=1,k=0) is",poisson(1,0))
print("The value of poisson(lambda=5,k=10) is",poisson(5,10))
print("The value of poisson(lambda=3,k=21) is",poisson(3,21))
print("The value of poisson(lambda=2.6,k=40) is",poisson(2.6,40))
print("The value of poisson(lambda=101,k=200) is",poisson(101,200))
