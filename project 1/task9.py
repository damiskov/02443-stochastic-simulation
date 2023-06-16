import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from task1 import MCMC
from task3 import gen_ecdf


def CTMC_monte_carlo_treatment():

    Q = np.array(
        [[-(0.0025+0.00125+0+0.001), 0.0025, 0.00125, 0, 0.001],
        [0, -(0.002+0.005), 0, 0.002, 0.005],
        [0, 0, -(0.003+0.005), 0.003, 0.005],
        [0, 0, 0, -0.009, 0.009],
        [0, 0, 0, 0, 0]]
    )

    N = len(Q[0])  
    
    # initial state = 0

    current_state = 0
    lifetime = 0
    
    while current_state != N-1: # Death occurs in final state

        # performing transition

        time_in_current_state = np.random.exponential(scale=-1/Q[current_state][current_state])

        lifetime += time_in_current_state
        
        transition_probabilities = []

        for i in range(N):

            if i != current_state:
                transition_probabilities.append(-Q[current_state][i]/Q[current_state][current_state])
            else:
                transition_probabilities.append(0) # Cannot remain in same state any longer


        new_state = np.random.choice(list(range(N)), p = transition_probabilities) # New column

        # Updating

        current_state = new_state
    

    return lifetime

def calc_kaplan_meier(lifetimes):

    t = sorted(lifetimes)

    S_t = [float((len(t)-i))/float(len(t)) for i in range(1,len(t)+1)]

    return S_t


def eval_treatment(n=1000):

    lifetimes_treatment = []
    lifetimes_no_treatment = []

    for _ in range(n):

        lifetimes_treatment.append(CTMC_monte_carlo_treatment())
        lifetime, _ = MCMC()
        lifetimes_no_treatment.append(lifetime)

    # Comparing Distributions

    plt.hist(lifetimes_treatment,bins=30, color="cornflowerblue", alpha=0.5, label="Treatment",edgecolor="gray")
    plt.hist(lifetimes_no_treatment,bins=30, color="lightcoral", alpha=0.5, label="No treatment",edgecolor="gray")
    plt.vlines(np.mean(lifetimes_no_treatment), ymin=0, ymax=140, color="darkkhaki", label=r"$\bar{x}=$"+f"{round(np.mean(lifetimes_no_treatment), 2)}")
    plt.vlines(np.mean(lifetimes_treatment), ymin=0, ymax=140, color="olivedrab", label=r"$\bar{x}_t=$"+f"{round(np.mean(lifetimes_treatment), 2)}")
    plt.title("Distribution of lifetimes before and after treatment")
    plt.xlabel(r"Lifetime")
    plt.ylabel("Frequency")
    plt.grid()
    plt.legend()
    plt.show()

    KM_treatment = calc_kaplan_meier(lifetimes_treatment)
    KM_no_treatment = calc_kaplan_meier(lifetimes_no_treatment)

    plt.plot(np.sort(lifetimes_treatment), KM_treatment, color="cornflowerblue", alpha=0.8, label="Treatment")
    plt.plot(np.sort(lifetimes_no_treatment), KM_no_treatment, color="lightcoral", alpha=0.8, label="No treatment")
    plt.title("Survival Function (Kaplan-Meier)")
    plt.xlabel(r"$t$")
    plt.grid()
    plt.legend()
    plt.ylabel(r"$\hat{S}(t)$")
    plt.show()
    
if __name__=="__main__":
    eval_treatment()