import numpy as np
import matplotlib.pyplot as plt

# Markov Chain Monte Carlo

def MCMC_task4():

    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    
    N = len(P[0])  
    
    # initial state = 0

    current_state = 0
    lifetime = 0
    
    cancer_reappearance = False # Boolean recording if cancer reappears locally
    survived_12_months = False

    while current_state != N-1: # Death occurs in final column

        # performing transition

        transition_probabilities = P[current_state]
        new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column

        lifetime += 1

        # Updating

        if new_state in [1,2,3] and lifetime <= 12:

            cancer_reappearance = True
        
        if lifetime > 12:
            survived_12_months = True

        current_state = new_state
    

    return lifetime, cancer_reappearance, survived_12_months

def get_lifetimes():

    lifetimes = []

    while len(lifetimes) < 1000:

        lifetime, cancer_reappearance, survived_12_months = MCMC_task4()

        if cancer_reappearance and survived_12_months:

            lifetimes.append(lifetime)

    
    return lifetimes


def examine_dist():

    lifetimes = get_lifetimes()
    print(np.mean(lifetimes))

if __name__=="__main__":
    examine_dist()