import numpy as np
import matplotlib.pyplot as plt
from task12 import get_jumps
from task12 import get_Y
from task12 import create_Q

""""
- Generate 1000 simulations (women), observed
- while error between subsequent Qs is > eps:
    - for each simulation
        - Using continuous-time model with Qi = Qk (initially Q0), produce similar time series as that in Yi (observed)
        - E.g., observed with an observed time series of [0, 2, 4]
                                [0, 48, 96]  
            it would be sufficient to come up with a solution which has 0<t<=24: 0 and 24<t<=48: 1 etc (i.e, more jumps permissible)
        - Use this to produce a set of jumps N and soujorn times for each state S (record this in dictionary and return dictionaries of times
        spent in each state.
        - Use S and N 
"""
def gen_partial_time_series(Q, time_frame, initial_state, target_state):
    """
    Generates a partial time series within specified timeframe.
    - Q: Transition Matrix
    - time_frame: timeframe in which to generate time series
    - intial_state: Starting state for tine series generation

    Returns:
    - new_time_series: list of states in new time series
    - sojourn_times: time spent in each state
    """
    
    new_time_series, sojourn_times = [initial_state], np.array([0 for _ in range(5)])


    time = 0
    current_state = initial_state

    N = len(Q[0])

    while time < time_frame or current_state!=N-1: 

        # performing transition

        scale=-1/Q[current_state][current_state]
        
        
        time_in_current_state = np.random.exponential(scale=scale)

        sojourn_times[current_state] = time_in_current_state
    
        time += round(time_in_current_state, 2)

        transition_probabilities = []

        for i in range(N): # Generating transition 

            if i != current_state:
                transition_probabilities.append(-Q[current_state][i]/Q[current_state][current_state])
            else:
                transition_probabilities.append(0) # Cannot remain in same state any longer


        new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column
        new_time_series.append(new_state)

        # Updating

        current_state = new_state


    return new_time_series, sojourn_times


def MC_expectation_maximisation():

    Y = get_Y() # set of 1000 samples of observed data (using target Q)

    Q_0 = np.array(
        [[-(0.0025+0.00125+0+0.001), 0.0025, 0.00125, 0, 0.001],
        [0, -(0.002+0.005), 0, 0.002, 0.005],
        [0, 0, -(0.003+0.005), 0.003, 0.005],
        [0, 0, 0, -0.009, 0.009],
        [0, 0, 0, 0, 0]], dtype=np.float64
    ) # Initial guess

    error = np.inf
    errors = [] 
    Q_k = Q_0
    iteration = 0 

    while error > 1e-3: # Main loop
        
        print(f"Iteration: {iteration}")

        Q_k_1 = np.zeros(shape=Q_k.shape) # initialisation of updated Q
        N_k = np.zeros(shape=(5,5))
        S_k = [0 for _ in range(5)]

        for Y_i in Y: # Iterating through observed samples
            

            satisficatory_time_series = False
            generated_time_series = []
            current_sojourn = np.array([0 for _ in range(5)])
            j = 0

            print(f"Target Series: {Y_i}")
            
            while not satisficatory_time_series: # Loop for generating satisfactory time series
                
                initial_state, target_state = Y_i[j], Y_i[j+1]

                print(f"Initial State:{initial_state}")
                print(f"Target State:{target_state}")
    

                new_partial_time_series, new_partial_sojourn = gen_partial_time_series(Q_k, initial_state, target_state)

                # print(f"Generated Partial time series: {new_partial_time_series}")
                
                if new_partial_time_series[-1] == target_state:

                    print(f"Valid partial time series found: {new_partial_time_series}")

                    print(f"Time spent in states: {new_partial_sojourn}")
                    
                    generated_time_series = generated_time_series + new_partial_time_series # updating generated time series
                    current_sojourn += new_partial_sojourn # updating sojourn times
                    j += 1 # Only progress when partial time series is satisfactory
                    
                    if target_state == 4 and generated_time_series[-1]==4: # Person is dead - According to both observed and generated time series
                        
                        satisficatory_time_series = True
                else:
                    print("Fail")
                
            print(f"Observed: {Y_i}")
            print(f"Generated: {generated_time_series}")
            
            # Exit while loop - Have generated acceptable time series
            # Need to update: N_k and S_k

            N_k += get_jumps(generated_time_series)
            S_k += current_sojourn
        
        """
        - Have finished updating N_k and S_k based on simulated time series adhering to observed time series.
        - Create Q_k_1 using these values, compare to Q_k and check error
        """
        iteration += 1
        Q_k_1 = create_Q(S_k, N_k)

        error = np.linalg.norm(Q_k, Q_k_1, ord=np.inf)
        print(f"Error: {error}")
        errors.append(error)
        Q_k = Q_k_1

    # error threshold satisfied

    plt.plot([i for i in range(iteration)], errors, color="cornflowerblue")
    plt.grid()
    plt.legend()
    plt.title("Error vs Iteration")
    plt.xlabel(r"$i$")
    plt.ylabel(r"$||Q_k - Q_{k+1}||_{\infty}$")
    plt.show()

    return Q_k
    
def analyse_result():
    Q = MC_expectation_maximisation()
    print(f"Final Q:\n{Q}")

if __name__=="__main__":
    analyse_result()