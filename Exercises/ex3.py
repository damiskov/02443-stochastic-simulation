import numpy as np
import matplotlib.pyplot as plt
from math import log, floor
from scipy import stats


# Uniform distribution using inversion method:

def uniform(a, b, U):
    return [a+(b-a)*i for i in U]

# Exponential distribution

def exponential(l, U):
    return [-log(1-i)/l for i in U]

# Pareto 

def pareto(k, beta, U):
    return [beta*(i**(-1/k)-1) for i in U]

def box_muller(U1, U2):
    return np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2), np.sqrt(-2*np.log(U1))*np.sin(2*np.pi*U2)

def hyper_exp(probabilities, lambdas):
    
    # Finding I
    
    cdf = np.cumsum(probabilities)
    intervals = [(0, cdf[0])]+[(cdf[i-1], cdf[i]) for i in range(1, len(cdf))]
    i_continuous = np.random.random(low=0, high=1)
    i_discrete = 0

    for i, interval in enumerate(intervals):
        if i_continuous > interval[0] and i_continuous <= interval[1]:

            i_discrete = i

    Z = exponential(lambdas[i_discrete], np.random.random(size=10000))
    return Z


def erlang(l, n):
    U = np.random.random(low=1, high=10, size=n)
    prod = 1
    for i in range(n):
        prod *= U[i]
    return (-1/l)*log(prod)





def main():

    # 1 - Generating distributions and testing/comparing

    # a) Exponential

    print("----- Exponential -----")

    U = np.random.random(size=1000)
    l = 0.2
    exp_dist = exponential(l, U)


    plt.scatter(U, exp_dist, marker="x", color="r", label=" \u03BB = 0.2")
    plt.grid()
    plt.legend()
    plt.title("Exponential")
    plt.show()

    kres = stats.kstest(exp_dist, stats.expon.cdf)

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres.statistic}\n\t- P-value: {kres.pvalue}")

    print("Histograms:")


    plt.hist(stats.expon.rvs(size=1000, scale=1/l), alpha=0.5, label="Reference")
    plt.hist(exp_dist, alpha=0.5, label="Generated")
    plt.legend()
    plt.grid()
    plt.show()
    
    exit()
    # b) Normal

    print("----- Normal -----")

    U1, U2 = np.random.random(size=1000), np.random.random(size=1000)

    x, y = box_muller(U1, U2)

    plt.scatter(x, y)
    plt.grid()
    plt.title("Normal")
    plt.show()

    kres_x = stats.kstest(x, stats.norm.cdf)

    print(f"Kolmogorov-Smirnov test (x):\n\t- Test statistic: {kres_x.statistic}\n\t- P-value: {kres_x.pvalue}")
    
    kres_y = stats.kstest(y, stats.norm.cdf)

    print(f"Kolmogorov-Smirnov test (y):\n\t- Test statistic: {kres_y.statistic}\n\t- P-value: {kres_y.pvalue}")


    # c) Pareto
    print("----- Pareto -----")
    U = np.random.random(size=1000)
    beta = 1

    # i) k = 2.05

    p1 = pareto(2.05, beta, U)
    kres_1 = stats.kstest(p1, stats.pareto.cdf(U, 2.05))

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres_1.statistic}\n\t- P-value: {kres_1.pvalue}")

    

    # ii) k = 2.5

    p2 = pareto(2.5, beta, U)
    kres_2 = stats.kstest(p2, stats.pareto.cdf(U, 2.5))

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres_2.statistic}\n\t- P-value: {kres_2.pvalue}")

    # iii) k = 3

    p3 = pareto(3, beta, U)
    kres_3 = stats.kstest(p2, stats.pareto.cdf(U, 3))

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres_3.statistic}\n\t- P-value: {kres_3.pvalue}")

    # iv) k = 4
    p4 = pareto(4, beta, U)
    kres_4 = stats.kstest(p2, stats.pareto.cdf(U, 4))

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres_4.statistic}\n\t- P-value: {kres_4.pvalue}")

    plt.scatter(U, p1, label="k = 2.05",marker="x")
    plt.scatter(U, p2, label="k = 2.5",marker="x")
    plt.scatter(U, p3, label="k = 3",marker="x")
    plt.scatter(U, p4, label="k = 4",marker="x")
    plt.title("Pareto")
    plt.legend()
    plt.show()


    return


if __name__=="__main__":
    main()