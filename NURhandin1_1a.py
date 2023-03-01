import numpy as np

def factorial(n):
    value = 1
    for i in range(1,n+1):
        value *= i
    return value

def gamma(n):
    return factorial(n-1)

def poisson(lambdas,k):
    return ((lambdas**k) * (np.exp(-lambdas))) / (factorial(k))

def ln_poisson(lambdas,k):
    if k < 40:
        return np.float32(k)*np.log(np.float32(lambdas)) - np.float32(lambdas) - np.log(np.float32(factorial(k))) #transforming to log space
    else:
        return k*np.log(lambdas/k) - (lambdas+k) #using Stirling's approximation for large k
    #else:
     #   return k*np.log(lambdas) - lambdas - np.sqrt(2*np.pi*k)*(np.float32(k)/np.exp(1))**np.float32(k)
    

print(ln_poisson(1,0))
print(ln_poisson(5,10))
print(ln_poisson(3,21))
print(ln_poisson(2.6,40))
print(ln_poisson(101,200))

print(poisson(1,0))
print(poisson(5,10))
print(poisson(3,21))
print(poisson(2.6,40))
#print(poisson(101,200))