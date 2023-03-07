import numpy as np

def ln_factorial(n):
    value = 0.0
    for i in range(1,n+1):
        value += np.float32(np.log(i))
    return value

def ln_poisson(lambdas,k):
    return np.int32(k)*np.float32(np.log(lambdas)) - np.float32(lambdas) - \
        np.float32(ln_factorial(k)) #transforming to log space
        
def poisson(lambdas,k):
    return np.float32(np.exp(ln_poisson(lambdas,k)))

print("The value of poisson(lambda=1,k=0) is",poisson(1,0))
print("The value of poisson(lambda=5,k=10) is",poisson(5,10))
print("The value of poisson(lambda=3,k=21) is",poisson(3,21))
print("The value of poisson(lambda=2.6,k=40) is",poisson(2.6,40))
print("The value of poisson(lambda=101,k=200) is",poisson(101,200))
