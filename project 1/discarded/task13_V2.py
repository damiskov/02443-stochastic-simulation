import numpy as np
import matplotlib.pyplot as plt
from task12 import get_jumps
from task12 import get_Y
from task12 import create_Q


def discretise_ts(states, times):

    final_state = states[-1]

    total_time = sum(times)

    new_times = [int(48*i/total_time) for i in times]
    
    discretised_ts = []
    
    new_sojourn = [0 for _ in range(5)]

    for state, time in zip(states, new_times):
        discretised_ts = discretised_ts + [state for i in range(time)]
        new_sojourn[state] = new_times[state]
    
    discretised_ts.append(final_state)

    return discretised_ts, new_sojourn

def gen_partial_time_series(Q, time_frame, initial_state, target_state):
    """
    Generates a partial time series within specified timeframe.
    - Q: Transition Matrix
    - time_frame: timeframe in which to generate time series
    - intial_state: Starting state for tine series generation

    Returns:
    - new time series: list of states in new time series
    - sojourn_times: time spent in each state
    """
    
    sojourn_times = [0 for _ in range(5)]


    time = 0
    current_state = initial_state
    states = [initial_state]
    generated_time_series = []

    N = len(Q[0])

    while time < 2000: 

        # performing transition
        
        
        scale=-1/Q[current_state][current_state]
        if scale < 0: 
            print("Scale < 0")
            print(Q[current_state][current_state])
            exit()

        
        time_in_current_state = round(np.random.exponential(scale=scale))

        time += time_in_current_state

        sojourn_times[current_state] += time_in_current_state



        # in the case that the target state = initial state state:

        if target_state == initial_state:

            print("Target state = Initial state")
            print("Checking if we've remained in same state for t < 96:")
            print(f"Time: {time_in_current_state}")

            if time_in_current_state > time_frame and time_in_current_state < 2*time_frame:

                print("Satisfied! Generating time series based on time and state")

                print(f"States: {states}")
                print(f"Times: {sojourn_times}")
                print(discretise_ts(states, sojourn_times))
                exit()
            
            else:

                # print("Unsatisfied. Recursively calling until acceptable time series generated")

                gen_partial_time_series(Q, time_frame, initial_state, target_state) # Recursive call until correctly generated time series
                
        else: 
            # target state != intial state - transition

            transition_probabilities = []

            for i in range(N): # Generating transition 

                if i != current_state:
                    transition_probabilities.append(-Q[current_state][i]/Q[current_state][current_state])
                else:
                    transition_probabilities.append(0) # Cannot remain in same state any longer


            new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column
            states.append(new_state)

            if new_state > target_state:

                new_state = target_state
                states.append(new_state)
                
                return discretise_ts(states, sojourn_times)

            print(f"Transitioning to new state: {new_state}")
            exit()
            # Updating

            current_state = new_state

            if current_state == target_state:
            
                return discretise_ts(states, sojourn_times)
            
            elif current_state > target_state or time > 2*time_frame: # Have missed target state or exceeded time between two observations

                gen_partial_time_series(Q, time_frame, initial_state, target_state)


        
        # elif current_state > target_state or time >= 2*time_frame:

        #     gen_partial_time_series(Q, time_frame, initial_state, target_state)

    raise Exception(f"Time exceeded realistic limit")
def test():

    print("Generating Y...")

    Y = get_Y() # set of 1000 samples of observed data (using target Q)


    Q_0 = np.array(
        [[-(0.0025+0.00125+0+0.001), 0.0025, 0.00125, 0, 0.001],
        [0, -(0.002+0.005), 0, 0.002, 0.005],
        [0, 0, -(0.003+0.005), 0.003, 0.005],
        [0, 0, 0, -0.009, 0.009],
        [0, 0, 0, 0, 0]]
    ) # Initial guess

    print(f"Q:\n{Q_0}")
    
    #Y_i = Y[0]
    Y_i = [1,2,2,3,4]
    initial_state = Y_i[0]
    target_state = Y_i[1]

    print(f"Finished generating Y. Taking a look at first time series:\n{Y_i}")


    print(f"Initial state: {initial_state}")
    print(f"Target state: {target_state}")

    print("Attempting to generate suitable partial time series... ")
    
    ts, times = gen_partial_time_series(Q_0, 48, initial_state, target_state)

    print(f"Succeeded:\nPartial Time Series: {ts}")




if __name__=="__main__":

    test()