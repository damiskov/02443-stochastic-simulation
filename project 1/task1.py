import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Markov Chain Monte Carlo

def MCMC():

    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    
    N = len(P[0])  
    
    # initial state = 0

    states = [0]
    
    local_reappearance = False # Boolean recording if cancer reappears locally

    while states[-1] != N-1: # Death occurs in final column

        currrent_state = states[-1] # Current i state (as defined by previous transition)

        # performing transition

        transition_probabilities = P[currrent_state]
        new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column

        # Updating

        if new_state == 1: # If transitioning into state 2, local reappearance has occured

            local_reappearance = True

        states.append(new_state)

    lifetime = (len(states)-1)/12

    return lifetime, local_reappearance


def plot_distribution_lifetimes(lifetimes):
    mean = round(np.mean(lifetimes), 2)
    plt.hist(lifetimes, 20, color="lightsteelblue", alpha=0.8, edgecolor='gray')
    plt.vlines(mean,ymin=0, ymax=200, color="lightcoral", alpha=0.8, label=r"$\bar{x}$"+f"= {mean}")
    plt.grid()
    plt.legend()
    plt.ylabel("frequency")
    plt.xlabel("lifetime after tumor removal (years)")
    plt.title("Distribution of lifetimes")
    plt.show()



def gen_analyse_samples(n):
    
    lifetimes, local_reappearances = [], 0


    # Random sampling for n times

    for i in range(n):
        
        if i%100==0:
            print(f"simulated {i} samples")

        # Perform MCMC
        
        lifetime, local_reappearance = MCMC()
        lifetimes.append(lifetime)
        if local_reappearance: local_reappearances += 1


    print("Finished generating samples")
    plot_distribution_lifetimes(lifetimes)
    print(f"Mean lifetime after tumor removal: {round(np.mean(lifetimes), 2)}")
    print(f"Proportion of women experiencing local reappearance of cancer: {local_reappearances/1000}")

    
if __name__=="__main__":
    gen_analyse_samples(1000)



