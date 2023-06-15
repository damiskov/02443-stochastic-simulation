import numpy as np
import matplotlib.pyplot as plt
from task12 import get_jumps
from task12 import get_sojourns
from task12 import get_Y


def MCMC_task13(Q, t, initial_state):
    
    N = len(Q[0])  
    
    # initial state = 0

    states = [initial_state]
    prev_time = 0
    intervals = []
    lifetime = 0
    current_state=0


    while lifetime <= t:  # Only simulate until time = t

        # performing transition

        time_in_current_state = np.random.exponential(scale=-1/Q[current_state][current_state])

        lifetime += round(time_in_current_state, 2)
        intervals.append((prev_time, lifetime))
        prev_time=lifetime

        transition_probabilities = []

        for i in range(N):

            if i != current_state:
                transition_probabilities.append(-Q[current_state][i]/Q[current_state][current_state])
            else:
                transition_probabilities.append(0) # Cannot remain in same state any longer


        new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column
        states.append(new_state)

        # Updating

        current_state = new_state


    return states, intervals



def MC_expectation_max():

    observed_women = get_Y()

    Q_0 = np.array(
        [-0.4, 0.1, 0.1, 0.1, 0.1],
        [0, -0.3, 0.1, 0.1, 0.1],
        [0, 0, -0.2, 0.1, 0.1],
        [0, 0, 0, -0,1, 0.1],
        [0,0,0,0,0]
    )


