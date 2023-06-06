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



def main():
    
    return

if __name__=="__main()__":
    main()


