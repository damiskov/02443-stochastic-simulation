import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from task7 import CTMC_monte_carlo
from task3 import gen_ecdf




def theoretical_probability(t):
    Q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
                  [0, -0.014, 0.005, 0.004, 0.005],
                  [0, 0, -0.008, 0.003, 0.005],
                  [0, 0, 0, -0.009, 0.009],
                  [0, 0, 0, 0, 0]])
    
    Q_s = Q[:-1, :-1]

    p_0 = np.array([1,0,0,0])

    F_t = 1 - sum(np.dot(p_0, expm(Q_s*t)))

    return F_t


def compare_simulated_theoretical(n):
    lifetimes = []


    for i in range(n):

        if i%10==0:
            print(f"i: {i}")


        lifetime, _ = CTMC_monte_carlo()
        lifetimes.append(lifetime)

    #t = np.arange(1200)
    t = np.sort(lifetimes)
    theoretical_cdf = np.array([theoretical_probability(i) for i in t])


    simulated_ecdf = gen_ecdf(lifetimes)

    #plt.hist(lifetimes, 20, color="lightsteelblue", alpha=0.8, edgecolor='gray', density=True, label="Simulated")
    plt.plot(t, simulated_ecdf, color="cornflowerblue", label="Simulated")
    plt.plot(t, theoretical_cdf, color="lightcoral", alpha=0.8, label="Theoretical")
    # plt.vlines(mean,ymin=0, ymax=200, color="lightcoral", alpha=0.8, label=r"$\bar{x}$"+f"= {mean}")
    plt.grid()
    plt.legend()
    plt.ylabel(r"$P(T \leq t)$")
    plt.xlabel(r"$t$")
    plt.title("Cumulative Probability Density of Lifetimes (Theoretical vs Simulated)")
    plt.show()

    # Kolomogorov-Smirnov test ...
    
if __name__=="__main__":
    compare_simulated_theoretical(1000)