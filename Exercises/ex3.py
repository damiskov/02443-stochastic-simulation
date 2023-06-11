import numpy as np
import matplotlib.pyplot as plt
from math import log
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

def composition_hyper_exp(probabilities, lambdas):
    
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


def composition_erlang(l, n):
    U = np.random.random(low=1, high=10, size=n)
    prod = 1
    for i in range(n):
        prod *= U[i]
    return (-1/l)*log(prod)

def pareto_hist_comparison(gen_sample, k):
    pareto_reference = np.random.pareto(k, size=1000)
    plt.hist(gen_sample, alpha=0.5, label="generated")
    plt.hist(pareto_reference, alpha=0.3, label="reference")
    plt.title(f"k = {k}")
    plt.legend()
    plt.show()

def CI_mean(sample):
    return np.mean(sample)-1.96*(np.std(sample)/np.sqrt(len(sample))), np.mean(sample)+1.96*(np.std(sample)/np.sqrt(len(sample)))

def CI_var(sample):
    n = len(sample)
    chi2_lower = stats.chi2.ppf(0.05/2, n-1)
    chi2_upper = stats.chi2.ppf(1 - 0.05/2, n-1)
    return (n-1)*np.var(sample)/chi2_lower, (n-1)*np.var(sample)/chi2_upper

def ex1():
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
    plt.title("Exponential distribution histograms")
    plt.grid()
    plt.show()
    
    
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

    plt.hist(x, 10, alpha=0.5, label="x")
    plt.hist(y, alpha=0.5, label="y")
    plt.hist(np.random.normal(0, 1, 1000), alpha =0.5, label="reference")
    plt.title("Normal distribution comparisons")
    plt.legend()
    plt.show()

    # c) Pareto
    print("----- Pareto -----")
    U = np.random.random(size=1000)
    beta = 1


    # i) k = 2.05

    p1 = pareto(2.05, beta, U)
    kres_1 = stats.kstest(p1, np.random.pareto(2.05, size=1000))

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres_1.statistic}\n\t- P-value: {kres_1.pvalue}")


    # ii) k = 2.5

    p2 = pareto(2.5, beta, U)
    kres_2 = stats.kstest(p2, np.random.pareto(2.5, size=1000))

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres_2.statistic}\n\t- P-value: {kres_2.pvalue}")

    # iii) k = 3

    p3 = pareto(3, beta, U)
    kres_3 = stats.kstest(p2, np.random.pareto(3, size=1000))

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres_3.statistic}\n\t- P-value: {kres_3.pvalue}")

    # iv) k = 4
    p4 = pareto(4, beta, U)
    kres_4 = stats.kstest(p2, np.random.pareto(4, size=1000))

    print(f"Kolmogorov-Smirnov test:\n\t- Test statistic: {kres_4.statistic}\n\t- P-value: {kres_4.pvalue}")

    plt.scatter(U, p1, label="k = 2.05",marker="x")
    plt.scatter(U, p2, label="k = 2.5",marker="x")
    plt.scatter(U, p3, label="k = 3",marker="x")
    plt.scatter(U, p4, label="k = 4",marker="x")
    plt.title("Pareto")
    plt.legend()
    plt.show()




    print("Histograms for Pareto")
    for sample, k in zip([p1,p2,p3,p4],[2.05, 2.5, 3, 4]):
        print(f"K: {k}")
        pareto_hist_comparison(sample, k)



def ex2():
    
    U = np.random.random(size=1000)
    ks = np.linspace(2.1, 5, 20)
    mean_gen, mean_analytical, var_gen, var_analytical = [], [], [], []

    for k in ks:
        # Simulating values
        p = pareto(k, 1, U)
        
        mean_gen.append(np.mean(p))
        mean_analytical.append(np.var(p))
        var_gen.append(k/(k-1))
        var_analytical.append(k/(((k-1)**2)*(k-2)))
    
    plt.scatter(ks, mean_gen, label="Simulated", marker="x")
    plt.scatter(ks, mean_analytical, label="Theoretical", marker="x")
    plt.title("Comparison between simulated mean and theoretical mean")
    plt.xlabel("k")
    plt.ylabel("E(x)")
    plt.legend()
    plt.show()

    plt.scatter(ks, var_gen, label="Simulated", marker="x")
    plt.scatter(ks, var_analytical, label="Theoretical", marker="x")
    plt.title("Comparison between simulated variance and heoretical variance")
    plt.xlabel("k")
    plt.ylabel("Var(x)")
    plt.legend()
    plt.show()



def ex3():
    high_means,low_means = [], []
    high_vars, low_vars = [], []


    for i in range(100):
        u1,u2 = np.random.uniform(size=10), np.random.uniform(size=10)
        X, _ = box_muller(u1,u2)
        l_m, h_m = CI_mean(X)
        l_v, h_v = CI_var(X)
        high_means.append(h_m)
        low_means.append(l_m)
        high_vars.append(h_v)
        low_vars.append(l_v)
    
    plt.scatter([i for i in range(100)], high_means)
    plt.show()
    plt.scatter([i for i in range(100)], high_vars)
    plt.show()

    # Perform some analysis

def ex4():
    mu = 2 # arbitrary mean for exponential distribution
    Y = np.random.exponential(scale=1/mu, size=10000) # Random exponential sample Y
    X = np.random.exponential(scale=1/Y) # Random exponential sampling of X, given Y

    # Creating CDF

    X_sort = sorted(X)
    F_x = 1. * np.arange(len(X)) / (len(X) - 1)


    
    # x = np.random.uniform(low=0, high=20,size=100)
    # F_x = [1-(1+i/mu)**(-1) for i in x]
    plt.scatter(X_sort, F_x, marker="x", color="g", label=f"\u03BC={mu}")
    plt.title("CDF of Simulated Pareto (composition method)")
    plt.xlabel("x")
    plt.xlim(0,100)
    plt.legend()
    plt.grid()
    plt.ylabel(r"$P(X \leq x)$")
    plt.show()

    




def main():

    # 1 - Generating distributions and testing/comparing

    ex1()
    
    # 2 - Comparison of statistical values of Generated Pareto distribution 

    ex2()

    # 3 - 100 Confidence Intervals 

    ex3()

    # 4 - Pareto via composition

    ex4()



    return


if __name__=="__main__":
    main()