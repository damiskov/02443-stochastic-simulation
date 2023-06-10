import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import chisquare
from math import log, floor
from collections import Counter

def geometric_dist(U, p):
    return [floor(log(i)/log(1-p)) for i in U]

# Helper function for crude method of discrete random sampling

def insert_into_categories(values, intervals):
    
    new_dist = [0 for i in range(len(intervals))]

    for value in values:
        for i, interval in enumerate(intervals): # Linear search    
            if interval[0] < value and value <= interval[1]:
    
                new_dist[i] += 1
    
    return new_dist

def crude_discrete(pdf, size):    
    
    u = np.random.random(size=size)
    cdf = np.cumsum(pdf)
    intervals = [(0, cdf[0])]+[(cdf[i-1], cdf[i]) for i in range(1, len(cdf))]
    
    return insert_into_categories(u, intervals)

def accept_reject_method(pdf, size, k=6, c=1):
    
    X = [0 for i in range(len(pdf))]
    
    while sum(X) < size:
    
        U1, U2 = np.random.random(), np.random.random()
        I = np.floor(k*U1)+1
        p_I = pdf[int(I)-1]
        if U2 <= p_I/c: X.append(I)
    
    
    return X

def gen_tables(pdf, k=6):
    L = [i for i in range(1, k+1)]
    F = [k*i for i in pdf]
    G = [i for i in range(len(F)) if F[i] >= 1]
    S = [i for i in range(len(F)) if i <= 1]
    while len(S) != 0:
        i = G[0]
        j = S[0]
        L[j] = i
        F[i] = F[i] - (1 - F[j])
        if F[i] < 1:
            G = G[1:]
            S.append(i)
        S = S[1:]
    
    return F, L

def alias_method(pdf, size, k=6):
    F, L = gen_tables(pdf)
    X = []
    while len(X) < size:
    
        U1, U2 = np.random.random(), np.random.random()
        I = np.floor(k*U1)+1

        if U2 <= F[int(I)-1]: X.append(I)
        else: X.append(L[int(I)-1])
    
    return X


def make_bar_plots(gen_dist, expected_dist):
    width = 0.8

    indices = np.arange(len(gen_dist))

    plt.bar(indices, expected_dist, width=width, 
            color='b', label='Given Distribution')
    plt.bar([i+0.25*width for i in indices], gen_dist, 
            width=0.5*width, color='r', alpha=0.5, label='Generated Distribution')

    plt.legend()

    plt.show()

def test(X, pdf):
    pdf_gen = [float(i/sum(X)) for i in X]
    print("----- Probability Distributions -----")
    print(f"Generated pdf: {pdf_gen}")
    print(f"Expected pdf: {pdf}")
    chisq, p = chisquare(X, f_exp=[i*sum(X) for i in pdf])
    print("----- Performing chi-squared test (alpha = 5%) -----\nH_0: Samples come from a common distribution.\nH_a: Samples do not share a common distribution.")
    print(f"Test statistic: {chisq}")
    print(f"P-value: {p}")
    if p < 0.05: print("Reject null hypothesis. Sufficient evidence to suggest samples come from different distributions.")
    else: print("Accept null hypothesis. Sufficient evidence to suggest that samples come from the same distribution.")
    print("----- Visual Comparison between PDFs -----")
    make_bar_plots(pdf_gen, pdf)

def make_freq_dist(X, keys=[1,2,3,4,5,6]):
    
    freq = []

    for i in keys:
    
        X.count(i)
        freq.append(X.count(i))

    return freq

def main():

    pdf = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]

    # Simulating distribution, using various methods

    crude_sample = crude_discrete(pdf, 10000)
    ar_sample = accept_reject_method(pdf, 10000)
    alias_sample = alias_method(pdf, 10000)
    
    print("----- Direct Method -----")
    test(crude_sample, pdf)
    print(".\n.\n.\n")
    print("----- Accept/Reject Method -----")
    test(make_freq_dist(ar_sample), pdf)
    print(".\n.\n.\n")
    print("----- Alias Method -----")
    test(make_freq_dist(alias_sample), pdf)


if __name__=="__main__":
    main()