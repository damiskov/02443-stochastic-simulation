import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations






def MCMC_task12(Q):

    # Q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
    #               [0, -0.014, 0.005, 0.004, 0.005],
    #               [0, 0, -0.008, 0.003, 0.005],
    #               [0, 0, 0, -0.009, 0.009],
    #               [0, 0, 0, 0, 0]])
    
    N = len(Q)  
    #print(Q)
    
    # initial state = 0

    states = [0]
    prev_time = 0
    intervals = []
    lifetime = 0
    current_state=0

    #record time spent in each state
    times_states = []
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


def MCMC_13(Q, initial_state, target_state):

    # Q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
    #               [0, -0.014, 0.005, 0.004, 0.005],
    #               [0, 0, -0.008, 0.003, 0.005],
    #               [0, 0, 0, -0.009, 0.009],
    #               [0, 0, 0, 0, 0]])
    
    N = len(Q)  

    N1 = np.zeros((5, 5))
    S = np.zeros(5)
    #print(Q)
    
    # initial state = 0

    current_state= initial_state
    states = [current_state]
    prev_time = 0
    intervals = []
    lifetime = 0

    sojourn = 0
    jumps = 0
   

    #record time spent in each state
    times_states = []
    while lifetime <= 48 and current_state != N-1: # end of 48 month timeframe

        if initial_state==target_state:
            time_in_current_state = np.random.exponential(scale=-1/Q[current_state][current_state])

            #update sojourn time
            S[current_state] += time_in_current_state

            lifetime += round(time_in_current_state, 2)
            states.append(current_state)

        else:
            # performing transition

            # print(Q)
            # print(f"current_state -> {current_state}" )
            # print(f"Q[cur][cur] -> {Q[current_state][current_state]}")

            time_in_current_state = np.random.exponential(scale=-1/Q[current_state][current_state])

            #update sojourn time
            S[current_state] += time_in_current_state

            #print(f"TIME: {time_in_current_state}")

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

            #update sojourn time
            S[current_state] += time_in_current_state
            states.append(new_state)

            # Updating
            current_state = new_state
    

    # for i in range(len(states)-1):
    #     if states[i]!=states[i+1]:
    #         N[states[i]] [states[i+1]] += 1
                
    


    for i in range (len(states)-1):
        if states[i]!=states[i+1]:
            N1[states[i]] [states[i+1]] += 1

    return states, N1, S


def get_Y(n, Q):

    Y = []

    for i in range(n):
        
        states, intervals = MCMC_task12(Q)
        #if i==0:
            #print(f"states, intervals = {states}, {intervals}")
    
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


def update_diagonal_elems(Q):
    sum = 0
    for i in range (len(Q)):
        for j in range (len(Q)):

            # set last row all zeros
            if i == len(Q) - 1:
                Q[i][j] = 0
            
            # set Q as lower triangular matrix
            if i > j:
                Q[i][j] = 0

    for i in range(len(Q)):
        for j in range (len(Q)):
            sum = 0
            if i == j:
                    for k in range(i + 1, len(Q)):
                        sum += Q[i][k]
                    Q[i][i] = -sum
            
            

    
    return Q          


#sum soujourn for different i's
def sum_soujourn(trajectories, states_intervals):
    S = np.zeros(5)
    for p in range(len(trajectories)):
        state = trajectories[p]
        intervals = states_intervals[p]
        print (f"state = {state}")
        print (f"time = {intervals}")
        for i in range (len(state)):
            S[state[i]] += 48*i
    print(f"S = {S}")
    return S

            




# start 

Q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
                  [0, -0.014, 0.005, 0.004, 0.005],
                  [0, 0, -0.008, 0.003, 0.005],
                  [0, 0, 0, -0.009, 0.009],
                  [0, 0, 0, 0, 0]])

Q0 = np.array([[0, 0.0025, 0.00125, 0, 0.001],
                    [0, 0, 0, 0.002, 0.005],
                    [0, 0, 0, 0.003, 0.005],
                    [0, 0, 0, 0, 0.009],
                    [0, 0, 0, 0, 0]])
Q0 = update_diagonal_elems(Q0)

print(Q0)

observations = get_Y(1000, Q)

path = []
N = np.zeros((5, 5))
S = np.zeros(5)



Q1 = np.array([[-0.4, 0.1, 0.1, 0.1, 0.1],
                    [0, -0.3, 0.1, 0.1, 0.1],
                    [0, 0, -0.2, 0.1, 0.1],
                    [0, 0, 0, -0.1, 0.1],
                    [0, 0, 0, 0, 0]])
Q1 = Q / 10 + 30 * np.identity(5)
Q1 = update_diagonal_elems(Q1)

# PROGRAM LOOP --------------------------------------------------------------------------------------------------------------

for _ in range (10):
    for obs in observations:
        Q_before = Q1
        for i in range (len(obs)-1):
            initial_state = obs[i]
            target_state = obs[i+1]
            accepted = False
            while accepted == False:
                new_sim, jumps, sojourn = MCMC_13(Q_before, initial_state, target_state)
                # print(obs)
                # print(f"New sim = {new_sim}")
                # print(f"Initial = {initial_state}")
                # print(f"Target = {target_state}")

                if new_sim[-1] == target_state:
                    #update S and N
                    
                    N += jumps
                    S += sojourn
                    accepted = True
        
    # update Q after going through all observations
    Q1 = Q_before
    for i in range(len(Q1)):
        for j in range(len(Q1)):
            if i != j:
                Q1[i][j] = N[i][j] / S[i]
    
    #set last row to zeros and update diagonal elements
    Q1 = update_diagonal_elems(Q1)


    print(f"Q = {Q1}")
    print(f"ERROR: {np.linalg.norm(Q_before - Q1, ord=np.inf)}")





# TESTING --------------------------------------------------------------------------------------------------------------

# obs = observations[0]
# Q_before = Q
# for i in range (len(obs)-1):
#     initial_state = obs[i]
#     target_state = obs[i+1]
#     accepted = False
#     while accepted == False:
#         new_sim, jumps, sojourn = MCMC_13(Q_before, initial_state, target_state)

#         print(obs)
#         print(f"New sim = {new_sim}")
#         print(f"Initial = {initial_state}")
#         print(f"Target = {target_state}")

#         if new_sim[-1] == target_state:
#             # update S and N
            
#             N += jumps
#             S += sojourn
#             accepted = True
# TESTING --------------------------------------------------------------------------------------------------------------

# print (obs)
# #print(path)
# print(N)
# print(S)

#loop_algo(observations, Q0)



