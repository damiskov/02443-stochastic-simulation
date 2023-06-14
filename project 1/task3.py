import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from task1 import MCMC


def calc_P_T_t(t):
    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    
    pi = np.array([1,0,0,0])
    
    P_s = P[:4, :4]
    
    p_s = P[:4, -1]


    return np.dot(pi, np.dot(np.linalg.matrix_power(P_s, t), p_s))

def calc_mean():
    P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001],
                  [0, 0.986, 0.005, 0.004, 0.005],
                  [0, 0, 0.992, 0.003, 0.005],
                  [0, 0, 0, 0.991, 0.009],
                  [0, 0, 0, 0, 1]])
    
    pi = np.array([1,0,0,0])
    
    P_s = P[:4, :4]
    
    a = np.linalg.inv(np.identity(4)-P_s)
    E_T = sum(np.dot(pi, a))
    
    
    return E_T


def compare_simulated_analytic():
    
    lifetimes_sim = []

    # Random sampling 5000 times

    for i in range(2000):
        
        if i%500==0:
            print(f"simulated {i} samples")

        # Perform MCMC
        
        lifetime, _ = MCMC()
        lifetimes_sim.append(lifetime)

    # Generating probability distribution:

    mean_sim = np.mean(lifetimes_sim)
    mean_a = calc_mean()

    x = np.arange(1, 1200)
    lifetimes_a = [calc_P_T_t(t) for t in x]

    plt.hist(lifetimes_sim, 20, color="lightsteelblue", alpha=0.8, edgecolor='gray',density=True)
    plt.plot(x, lifetimes_a, color="lightcoral")
    plt.vlines(mean_sim, ymin=0, ymax=0.0026, color="darkkhaki", label=r"$\bar{x}$"+f" simulated= {round(mean_sim, 2)}")
    plt.vlines(mean_a, ymin=0, ymax=0.0026, color="olivedrab", label=r"$\bar{x}$"+f" analytical= {round(mean_a, 2)}")
    plt.grid()
    plt.legend()
    plt.ylabel(r"$P(T=t$)")
    plt.xlabel(r"$t$"+" (months)")
    plt.title("Distribution of lifetimes")
    plt.show()



    
if __name__=="__main__":
    compare_simulated_analytic()
    



