import numpy as np

def ln_factorial(n):
    value = 0.0
    for i in range(1,n+1):
        value += np.float32(np.log(i))
    return value

def ln_poisson(lambdas,k):
    return np.int32(k)*np.float32(np.log(lambdas)) - np.float32(lambdas) - np.float32(ln_factorial(k)) #transforming to log space


print(ln_poisson(1,0))
print(ln_poisson(5,10))
print(ln_poisson(3,21))
print(ln_poisson(2.6,40))
print(ln_poisson(101,200))

print("The value is",np.float32(np.exp(ln_poisson(1,0))))
print("The value is",np.float32(np.exp(ln_poisson(5,10))))
print("The value is",np.float32(np.exp(ln_poisson(3,21))))
print("The value is",np.float32(np.exp(ln_poisson(2.6,40))))
print("The value is",np.float32(np.exp(ln_poisson(101,200))))

