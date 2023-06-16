import numpy as np
import matplotlib.pyplot as plt
from task12 import get_jumps
from task12 import get_Y
from task12 import create_Q

def compress_time_series(ts, timeframe=48):
    
    states = sorted(list(set(ts)))
    times = [0 for _ in range(5)]

    for state in states:
        if state!=4:
            times[state] += timeframe*(ts.count(state))
        
    

    return states, times


def generate_48_months(Q, intial_state, target_state, time_frame):
    accepted_sequence = False
    N = len(Q[0])
    states = []
    
    if intial_state == 4:
        return [4]

    while not accepted_sequence:
    
        sojourn_times = [0 for _ in range(5)]
        time = 0
        states = [intial_state]

        

        while time < time_frame and states[-1]!=4:

            current_state = states[-1]
            scale=-1/Q[current_state][current_state]
            time_in_current_state = round(np.random.exponential(scale=scale))
            time += time_in_current_state
            sojourn_times[current_state]+=time_in_current_state

            transition_probabilities = []

            for i in range(N): # Generating transition 

                if i != current_state:
                    transition_probabilities.append(-Q[current_state][i]/Q[current_state][current_state])
                else:
                    transition_probabilities.append(0) # Cannot remain in same state any longer

            new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column
            states.append(new_state)
        
        if states[-1] == target_state:
            print("Generated acceptable sequence!")
            accepted_sequence = True
        
    return states[:-1] # Ignoring last value
                

def gen_entire_time_series(Q, observed, timeframes):

    total_seq = []

    for i, state in enumerate(observed):
        if state == 4:
            total_seq = total_seq +[4]
        
        else:
            initial_state = state
            target_state = observed[i+1]

            print(f"Initial state: {initial_state}")
            print(f"Target state: {target_state}")
            print(f"Time frame: {timeframes[state]}")
            
            partial_seq = generate_48_months(Q, initial_state, target_state, timeframes[state])
        
            total_seq = total_seq + partial_seq
    
    return total_seq 




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
    
    Y_i = Y[0]

    
    Y_i_compressed, timeframes = compress_time_series(Y_i)
    initial_state = Y_i_compressed[0]
    target_state = Y_i_compressed[1]

    print(f"Finished generating Y. Taking a look at first time series:\n{Y_i}")
    print(f"Compressed time series: {Y_i_compressed}")
    
    print(f"Initial state: {initial_state}")
    print(f"Target state: {target_state}")
    print(f"Time frames: {timeframes}")

    print("Attempting to generate suitable partial time series... ")
    
    partial_seq = generate_48_months(Q_0, initial_state, target_state, timeframes[0])
    print(f"Generate partial sequence between {initial_state} and {target_state}")
    print(partial_seq)

    print(f"Attempting to generate entire new series...")
    new_seq = gen_entire_time_series(Q_0, Y_i_compressed, timeframes)
    print(f"Original Sequence: {Y_i}")
    print(f"New Sequence: {new_seq}")




if __name__=="__main__":

    test()