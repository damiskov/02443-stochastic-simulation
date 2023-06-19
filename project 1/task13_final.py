import numpy as np
import matplotlib.pyplot as plt
import time
from task12 import get_Y
from task12 import create_Q


def calc_jumps(seq):

    N = np.zeros(shape=(5,5))
    for i in range(len(seq)-1):
        if seq[i]!=seq[i+1]:
            N[seq[i]][seq[i+1]] += 1

    return N



def generate_partial_seq(initial_state, target_state, Q, time_frame):
    
    seq = [initial_state] # Sequence to be accepted
    sojourn_times = [0 for _ in range(5)] # records time spent in states for partial sequence
    accepted = False # Boolean to record whether or not we have generated suitable sequence
    N = 5 # Number of states
    
    if target_state == initial_state: # No consideration of transition
        
        sojourn_times[initial_state] += time_frame
        return [initial_state], sojourn_times


    while not accepted:


        sojourn_times = [0 for _ in range(5)] # records time spent in states for partial sequence
        time = 0
        states = [initial_state]

        # In the case that target = initial:

        if target_state == initial_state: # No consideration of transition
        
            sojourn_times[initial_state] += time_frame
            return [initial_state], sojourn_times


        while time < time_frame and states[-1]!=4:

            current_state = states[-1]
            # scale=-1/Q[current_state][current_state]
            # time_in_current_state = round(np.random.exponential(scale=scale))
            # time += time_in_current_state
            # sojourn_times[current_state] += time_in_current_state
            transition_probabilities = []

            for i in range(N): # Generating transition 

                if i != current_state:
                    transition_probabilities.append(-Q[current_state][i]/Q[current_state][current_state])
                else:
                    transition_probabilities.append(0) # Cannot remain in same state any longer

            new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column
            # states.append(new_state)
            scale=-1/Q[current_state][current_state]
            time_in_current_state = round(np.random.exponential(scale=scale))
            # time += time_in_current_state
            # sojourn_times[current_state] += time_in_current_state

            if new_state == target_state: # Sequence to target state generated
                # Calculating time in state prior to target state
                if time+time_in_current_state >= time_frame:
                    
                    time_in_current_state = time_frame - time
                    sojourn_times[current_state] += time_in_current_state

                elif time+time_in_current_state < time_frame:
                    
                    sojourn_times[current_state] = time_in_current_state 
                    time_in_target_state = time_frame - time
                    sojourn_times[target_state] = time_in_target_state


                return states[:-1], sojourn_times
            
            else:
                
                time += time_in_current_state
                sojourn_times[current_state] += time_in_current_state
                states.append(new_state)

        # print(f"Potential Sequence: {states}")
        # Out of loop - time > time_frame
        
        if states[-1] == target_state: # time condition satisfied, accept generated series
            # print("Accept generated sequence")
            seq = states[:-1] # Ignore target state, will become part of sequence in next iteration
            accepted = True

    return seq, sojourn_times


def simulate_seq(observed, Q):
    
    time_frame = 48
    generated_seq = []
    sojourn_times = np.array([0 for _ in range(5)])

    for i in range(len(observed)-1):
        
        initial = observed[i]
        target = observed[i+1]
        # print(f"Intial: {initial}")
        # print(f"Target: {target}")
        seq, times = generate_partial_seq(initial, target, Q, time_frame)
        generated_seq += seq
        sojourn_times += np.array(times)

    generated_seq += [4]

    return generated_seq, sojourn_times



def main_loop(epsilon):

    errors = []
    k = 0
    Y = get_Y() # set of 1000 samples of observed data (using target Q)
    Q_k = np.array(
        [[-(0.0025+0.00125+0+0.001), 0.0025, 0.00125, 0, 0.001],
        [0, -(0.002+0.005+0.004), 0.004, 0.002, 0.005],
        [0, 0, -(0.003+0.005), 0.003, 0.005],
        [0, 0, 0, -0.009, 0.009],
        [0, 0, 0, 0, 0]]
    ) # Initial guess

    # Q_k = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
    #               [0, -0.014, 0.005, 0.004, 0.005],
    #               [0, 0, -0.008, 0.003, 0.005],
    #               [0, 0, 0, -0.009, 0.009],
    #               [0, 0, 0, 0, 0]])
    err = np.inf
    
    while err > epsilon and k < 100:
        
        # initialise S_k and N_k
        N_k = np.zeros(shape=(5,5))
        S_k = np.zeros(5)

        for i in range(len(Y)):
            
            simulated_seq, sojourns = simulate_seq(Y[i], Q_k)
            # print(simulated_seq)
            # print(sojourns)
            N_k += calc_jumps(simulated_seq)
            S_k += sojourns

        # print(N_k)
        # print(S_k)
        time.sleep(1)
    
        Q_k_1 = create_Q(S_k, N_k)
        print(f"New Q:\n{Q_k_1}")

        err = np.linalg.norm(Q_k_1-Q_k, ord=np.inf)
        k += 1
        print(k)
        print(err)
        errors.append(err)

    plt.plot([i for i in range(k)], errors, color="lightcoral")
    plt.plot([i for i in range(k)], [epsilon for _ in range(k)], color="cornflowerblue", label=r"$\epsilon=10^{-3}$")
    plt.ylabel(r"$||Q_{k+1}-Q_{k}||_{\infty}$")
    plt.xlabel(r"$k$")
    plt.title("Error value vs Iteration")
    plt.show()
            






def test():
    
    example_ts = [0,0,1,2,4]
    Q_k = np.array(
        [[-(0.0025+0.00125+0+0.001), 0.0025, 0.00125, 0, 0.001],
        [0, -(0.002+0.005+0.004), 0.004, 0.002, 0.005],
        [0, 0, -(0.003+0.005), 0.003, 0.005],
        [0, 0, 0, -0.009, 0.009],
        [0, 0, 0, 0, 0]]
    ) # Initial guess

    initial, target = example_ts[0], example_ts[1]

    print(f"Initial State: {initial}")
    print(f"Target: {target}")

    print(f"Generated partial sequence and sojourn times:\n{generate_partial_seq(initial, target, Q_k, 48)}")


    print(f"Simulating full sequence:\n{simulate_seq(example_ts,Q_k)}")



if __name__=="__main__":
    main_loop(1e-3)






