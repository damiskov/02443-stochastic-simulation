import numpy as np
import matplotlib.pyplot as plt
from task1 import MCMC



def crude_monte_carlo(n=200):

    lifetimes_below_350 = 0

    for _ in range(n):

        lifetime, _ = MCMC()

        if lifetime < 350:

            lifetimes_below_350 += 1
        
    return lifetimes_below_350/n


def get_prop(n=100):
    
    mean_prop = 0

    for _ in range(n):
        mean_prop += crude_monte_carlo()
    
    return mean_prop/n


if __name__=="__main__":
    print(f"Proportion of women with lifetime below 350: {get_prop()}")