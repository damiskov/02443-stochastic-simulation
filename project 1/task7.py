
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def CTMC_monte_carlo():

    Q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
                  [0, -0.014, 0.005, 0.004, 0.005],
                  [0, 0, -0.008, 0.003, 0.005],
                  [0, 0, 0, -0.009, 0.009],
                  [0, 0, 0, 0, 0]])
    
    N = len(Q[0])  
    
    # initial state = 0

    current_state = 0
    lifetime = 0
    distant_reappearance = False
    
    while current_state != N-1: # Death occurs in final state

        # performing transition

        time_in_current_state = np.random.exponential(scale=-1/Q[current_state][current_state])

        lifetime += time_in_current_state
        
        transition_probabilities = []

        for i in range(N):

            if i != current_state:
                transition_probabilities.append(-Q[current_state][i]/Q[current_state][current_state])
            else:
                transition_probabilities.append(0) # Cannot remain in same state any longer


        new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column


        if new_state in [2,3]:

            distant_reappearance = True

        # Updating

        current_state = new_state
    

    return lifetime, distant_reappearance
    

def get_lifetimes(n=1000):

    lifetimes = []
    prop = 0

    for i in range(n):

        if i%10==0:
            print(f"i: {i}")


        lifetime, distant_reappearance = CTMC_monte_carlo()
        lifetimes.append(lifetime)
        if distant_reappearance:
            prop+=1

    mean = round(np.mean(lifetimes), 2)
    plt.hist(lifetimes, 20, color="lightsteelblue", alpha=0.8, edgecolor='gray')
    plt.vlines(mean,ymin=0, ymax=200, color="lightcoral", alpha=0.8, label=r"$\bar{x}$"+f"= {mean}")
    plt.grid()
    plt.legend()
    plt.ylabel("Frequence")
    plt.xlabel("lifetime after tumor removal (months)")
    plt.title("Distribution of Lifetimes (Continuous Markov Chain)")
    plt.show()

    print(f"Proportion of women experiencing distant reappearance of cancer: {prop/n}")

if __name__=="__main__":
    get_lifetimes()
