import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Markov Chain Monte Carlo

P1 = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])

def MCMC():

    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    
    N = len(P[0])  
    
    # initial state = 0

    states = [0]
    
    local_reappearance = False # Boolean recording if cancer reappears locally
    twelve_month_reappearance = False # Boolean recording if cancer reappears within 12 months

    while states[-1] != N-1: # Death occurs in final column

        currrent_state = states[-1] # Current i state (as defined by previous transition)

        # performing transition

        transition_probabilities = P[currrent_state]
        new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column

        # Updating

        if new_state == 1: # If transitioning into state 2, local reappearance has occured

            local_reappearance = True

        states.append(new_state)

        # Checking if 12 month reappearance has occured
        if (len(states)-1) == 12 and local_reappearance == True: # If transitioning into state 2 within 12 months, 12 month reappearance has occured
            twelve_month_reappearance = True

    lifetime = len(states)-1

    return lifetime, local_reappearance, twelve_month_reappearance


def plot_distribution_lifetimes(lifetimes):
    plt.hist(lifetimes, 20, color="b", alpha=0.5)
    plt.grid()
    plt.ylabel("count")
    plt.xlabel("years after tumor removal")
    plt.title("Distribution of lifetimes")
    plt.show()


def gen_analyse_samples(task, n):
    
    lifetimes, local_reappearances = [], 0


    if task == "task4":
        while len(lifetimes) < n:
            

            lifetime, local_reappearance, twelve_m_reappearence = MCMC()

            if lifetime > 12 and twelve_m_reappearence == True:

                lifetimes.append(lifetime)

            if local_reappearance: local_reappearances += 1

        
        print("Finished generating samples")
        plot_distribution_lifetimes(lifetimes)
        print(f"Mean lifetime after tumor removal: {round(np.mean(lifetimes), 2)}")
        print(f"Proportion of women experiencing local reappearance of cancer: {local_reappearances/1000}")
    
    elif task == "task5":

        fraction_collection, mean_lifetime_collection = [], []
        n_sims = 10
        for j in range (n_sims):
            print(j)
            #print("loop started")
            less350_count = 0
            lifetimes = []

            for i in range(200):

                lifetime, local_reappearance, twelve_m_reappearence = MCMC()
                lifetimes.append(lifetime)
                if lifetime <= 350: less350_count += 1

            fraction_collection.append(less350_count/200)
            mean_lifetime_collection.append(np.mean(lifetimes))
            #print(lifetimes)

        #print(fraction_collection)
        print(f"Estimate of fraction within 350: {np.mean(fraction_collection)}")
        print(f"Variance of fraction within 350: {np.var(fraction_collection)}")

        ## CONTROL VARIABLE METHOD

        Y = mean_lifetime_collection
        X = fraction_collection

        multXY = []
        for a in range(n_sims):
            multXY.append(Y[a]*X[a])

        cov1 = np.mean(multXY) - np.mean(Y) * np.mean(X)
        var1 = np.var(Y)

        c = -cov1 / var1


        Z = X + c*(Y - np.mean(Y))
        
        estimate = np.mean(Z)
        variance =np.var(Z)

        print(estimate)
        print(variance)



    
    
if __name__=="__main__":
    gen_analyse_samples("task4", 1000)