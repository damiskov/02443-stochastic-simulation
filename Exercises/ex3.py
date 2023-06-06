import numpy as np
import matplotlib.pyplot as plt
from math import log, floor



# Uniform distribution using inversion method:

def uniform(a, b, U):
    return [a+(b-a)*i for i in U]

# Exponential distribution

def exponential(l, U):
    return [-log(i)/l for i in U]

# Pareto 

def pareto(k, beta, U):
    return [beta*(i**(-1/k)-1) for i in U]

def box_muller(U1, U2):
    return np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2), np.sqrt(-2*np.log(U1))*np.sin(2*np.pi*U2)

def hyper_exp(probabilities, lambdas):
    
    # Using "crude" method to choose I
    
    cdf = np.cumsum(probabilities)
    intervals = [(0, cdf[0])]+[(cdf[i-1], cdf[i]) for i in range(1, len(cdf))]
    i_continuous = np.random.random(low=0, high=1)
    i_discrete = 0

    for i, interval in enumerate(intervals):
        if i_continuous > interval[0] and i_continuous <= interval[1]:

            i_discrete = i

    Z = exponential(lambdas[i_discrete], np.random.random(size=10000))
    return Z


def erlang(l, n):
    U = np.random.random(low=1, high=10, size=n)
    prod = 1
    for i in range(n):
        prod *= U[i]
    return (-1/l)*log(prod)





def main():

    # 1 - Generating distributions and testing/comparing

    # a) Exponential

    U = np.random.random(size=10000)
    l = 0.2
    exp_dist = exponential(l, U)

    # b) Normal

    # c) Pareto
    # i) k = 2.05
    # ii) k = 2.5
    # iii) k = 3
    # iv) k = 4

    return

if __name__=="__main()__":
    main()


