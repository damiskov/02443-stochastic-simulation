import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def get_analytical_state_vector(P, t):

    
    P_t = np.linalg.matrix_power(P, t)
    p0 = np.array([1,0,0,0,0])

    return np.dot(p0, P_t) 


    
# Markov Chain Monte Carlo

def get_state_at_120_MCMC():

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

    return states[-1] # State at t = 120

def task_2(n):
    """
    Comparing distribution over states at t = 120 for both
    analytical and simulated approach
    """

    # Generating analytical distribution

    t = 120
    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
       
    analytical = 1000*get_analytical_state_vector(P, t)

    simulated = np.zeros(5)

    for i in range(n): # Simulating n women
        
        if i%100==0:
            print(f"simulated {i} samples")
        
        state_at_120 = get_state_at_120_MCMC()
        simulated[state_at_120] += 1 # Counting number of times in each state

    simulated = simulated # Normalizing
    
    
    print(f"Simulated: {simulated}")
    print(f"analytical: {analytical}")

    x = np.array(list(range(1, len(simulated)+1)))

    plt.bar(x-0.2, simulated, color="lightsteelblue", label="Simulated",alpha=0.8, width=0.4, align='center')
    plt.bar(x+0.2, analytical,  color="lightcoral", label="Analytical", alpha=0.8, width=0.4, align='center')
    plt.xticks(x)
    plt.xlabel(r"$i$")
    plt.ylabel(r"$P(X = i)$")
    plt.title("Frequency Distribution "+r"$(t=120)$")
    plt.legend()
    plt.grid()
    plt.show()

    

    t, p = stats.chisquare(f_obs=simulated, f_exp=analytical)
    print("chi-2 test:")
    print(f"Test statistic: {t}")
    print(f"P-value: {p}")


    
if __name__=="__main__":
    task_2(1000)

    





