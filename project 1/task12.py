import numpy as np
import matplotlib.pyplot as plt


def MCMC_task12():

    Q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
                  [0, -0.014, 0.005, 0.004, 0.005],
                  [0, 0, -0.008, 0.003, 0.005],
                  [0, 0, 0, -0.009, 0.009],
                  [0, 0, 0, 0, 0]])
    
    N = len(Q[0])  
    
    # initial state = 0

    states = [0]
    prev_time = 0
    intervals = []
    lifetime = 0
    current_state=0


    while current_state != N-1: # Death occurs in final state

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



def get_Y(n=1000):

    Y = []

    for _ in range(n):
        
        states, intervals = MCMC_task12()
    
        observed_states = []
        time = 0
        lifetime = intervals[-1][1]

        while time <= lifetime:
            
            for i, state_time in enumerate(intervals): # Linear search through intervals
                
                if time >= state_time[0] and time < state_time[1]: # Interval contains current time
                    
                    observed_states.append(states[i])
            
            time += 48

        observed_states.append(4)
            

        Y.append(observed_states)

    
    return Y

def get_sojourns(Y):

    S = {i: 0 for i in range(5)}

    for state in range(5):
        for row in Y:
            
            state_freq = row.count(state)
            if state_freq > 0:
                time_in_state = (state_freq-1)*48
                S[state] += time_in_state
    return S

def get_jumps(Y):

    N = np.zeros(shape=(5,5))

    for woman in Y:
        
        states = list(set(woman))

        for i in range(len(states)-1):

            N[states[i]][states[i+1]] += 1

    return N

def create_Q(S, N):

    Q = np.zeros(shape=(5,5))

    for i in range(len(Q)): # rows
        
        for j in range(len(Q[0])): # columns
            
            if i!=j:
                Q[i][j] = N[i][j]/S[i]
        
        Q[i][i] = -sum(Q[i])
        
    Q[-1] = np.zeros(5)

    return Q




def task12():
    Y = get_Y()
    S = get_sojourns(Y)
    N = get_jumps(Y)
    Q = create_Q(S, N)
    print(f"Created Q:\n{Q}")

if __name__=="__main__":
    task12()