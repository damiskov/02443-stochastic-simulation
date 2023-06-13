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


def expected_dist(P):
    P_120 = np.linalg.matrix_power(P, 120)

    p0 = P[0]


    return np.dot(p0, P_120)

def theoretical_probability_mass():
    pi = np.array([0.9915, 0.005, 0.0025, 0, 0.001])
    





# Markov Chain Monte Carlo

def MCMC():

    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    
    N = len(P[0])  
    
    # initial state
    
    J = [0]
    
    for _ in range(120): # t = 0 -> 120

        currrent_i = J[-1] # Current i state (as defined by previous transition)

        # performing transition

        transition_probabilities = P[currrent_i]
        new_j = np.random.choice(list(range(N)), p = transition_probabilities) # New column

        # Updating

        J.append(new_j)


    return J

def get_samples(n):
    
    observed = np.zeros(5)

    
    
    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    expected = expected_dist(P)

    for i in range(n):
        
        if i%100==0:
            print(f"simulated {i} samples")
        
        sequence_of_states = MCMC()
        current_dist = np.array(make_dist(sequence_of_states))
        observed += current_dist

    observed = observed/sum(observed) # Normalising
    
    
    print(f"Observed: {observed}")
    print(f"Expected: {expected}")

    plt.bar(list(range(1, len(observed)+1)), observed, color="b", label="Sample", alpha=0.5, width=0.8)
    plt.bar(list(range(1, len(observed)+1)), expected,  color="r", label="Expected", alpha=0.5, width=0.8)
    plt.xticks(list(range(1, len(observed)+1)))
    plt.xlabel(r"i")
    plt.title("Distribution of States (at t=120)")
    plt.legend()
    plt.grid()
    plt.show()

    

    p = doChi2(observed, expected)
    print("chi-2 test:")
    print(f"P-value: {p}")


    


    
if __name__=="__main__":
    get_samples(1000)

    





