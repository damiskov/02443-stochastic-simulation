import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

def make_dist(p):
    
    counts = [p.count(i)/len(p) for i in range(5)]
    
    return counts
    
def doChi2(observed, expected):

    _, p_val = stats.chisquare(f_obs=observed, f_exp=expected)

    return p_val


def get_state_vector(P, t):

    # Making use of Markov Property
    
    P_t = np.linalg.matrix_power(P, t)
    p0 = np.array([1,0,0,0,0])

    return np.dot(p0, P_t) 


    
# Markov Chain Monte Carlo

def MCMC():

    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    
    N = len(P[0])  
    
    # initial state
    
    states = [0]
    
    for _ in range(120): # t = 0 -> 120

        currrent_state = states[-1] # Current state (as defined by previous transition)

        # performing transition

        transition_probabilities = P[currrent_state]
        new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column

        # Updating

        states.append(new_state)


    return states

def get_samples(n):
    
    observed = np.zeros(5)
    t = 120
    
    
    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    expected = get_state_vector(P, t)

    for i in range(n):
        
        if i%100==0:
            print(f"simulated {i} samples")
        
        sequence_of_states = MCMC()
        state_vector = np.array(make_dist(sequence_of_states))
        observed += state_vector

    observed = observed/n # Mean state vector
    
    
    print(f"Observed: {observed}")
    print(f"Expected: {expected}")

    x = np.array(list(range(1, len(observed)+1)))

    plt.bar(x-0.2, observed, color="lightblue", label="Simulated",alpha=0.8, width=0.4, align='center')
    plt.bar(x+0.2, expected,  color="lightcoral", label="Analytical", alpha=0.8, width=0.4, align='center')
    plt.xticks(x)
    plt.xlabel(r"$i$")
    plt.ylabel(r"$P(X = i)$")
    plt.title("Probability Distribution "+r"$(t=120)$")
    plt.legend()
    plt.grid()
    plt.show()

    

    p = doChi2(observed, expected)
    print("chi-2 test:")
    print(f"P-value: {p}")


    


    
if __name__=="__main__":
    get_samples(10000)

    





