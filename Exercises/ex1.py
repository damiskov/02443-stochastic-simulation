import matplotlib.pyplot as plt
import copy
import scipy.stats as stat
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Linear Congruential Generator

def LCG(M, a, c, seed, n):
    for i in range(n):
        seed = (a*seed+c)%M
        yield seed

def histogram(numbers, num_bins):
    sns.displot(numbers, bins=num_bins)
    plt.show()

def chi_squared(numbers, num_bins):
    n_expected = len(numbers)/num_bins
    l = min(numbers)
    step = int((max(numbers)-l)/num_bins)
    test_stat = 0
    for i in range(1, num_bins+1):
        n_observed = len([j for j in numbers if j >= l+step*(i-1) and j < l+step*i])
        test_stat += ((n_expected-n_observed)**2)/n_expected
    return test_stat

def kolmogorov_smirnov(numbers, num_bins):
    ecdf = [0]
    l = min(numbers)
    h = max(numbers)
    step = int((h-l)/num_bins)
    n = len(numbers)

    for i in range(1, num_bins+1):
        ecdf.append(len([j for j in numbers if j <= l+step*i]))
    
    ecdf = np.array(ecdf)
    ideal = np.linspace(l, h, n)

    D = max(abs(np.array(ecdf) - np.array(ideal)))
    test_stat =  (np.sqrt(n) + 0.12 + (0.11/np.sqrt(n))) * D

    return test_stat

def run_test_I(numbers):

    median = np.median(np.array(numbers))
    n_1 = len([i for i in numbers if i > median])
    n_2 = len([i for i in numbers if i < median])

    runs_expected = 2*n_1*n_2/(n_1+n_2) + 1 # Mean


    numerator = 2*n_1*n_2*(2*n_1*n_2-n_1-n_2)
    denominator = ((n_1+n_2)**2)*(n_1+n_2+1)

    s_R = np.sqrt(numerator/denominator) # Variance

    # calculating number of runs

    num_runs = 0
    prev_val = ""
    if numbers[0]>median: prev_val = "above"
    else: prev_val = "below"


    for i in numbers[1:]:
        if prev_val == "above" and i < median: 
            num_runs += 1
            prev_val = "below"
        elif prev_val == "below" and i > median:
            num_runs += 1
            prev_val = "above"

    Z = abs((num_runs-runs_expected)/s_R)
    if Z>1.96:
        print("Reject Null hypothesis. Numbers are not independent.")
    else:
        print("Accept null hypothesis. Numbers are sufficiently independent at 5%.")
    
def run_test_II(numbers):
    R = np.zeros(shape=6, dtype=np.int32)
    current_run = 1
    for i in range(1, len(numbers)):
        if numbers[i] < numbers[i-1]:
            if current_run >= 6:
                R[-1] += 1
            else:
                R[current_run-1] += 1
            current_run = 1
        else:
            current_run += 1
    
    B = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])
    A = np.array(
            [[4529.4, 9044.9, 13568, 18091, 22615, 27892],
             [9044.9, 18097, 27139, 36187, 45234, 55789], 
             [13568, 27139, 40721, 54281, 67852, 83685], 
             [18091, 36187, 54281, 72414, 90470, 111580], 
             [22615, 45234, 67852, 90470, 113262, 139476], 
             [27892, 55789, 83685, 111580, 139476, 172860]], dtype=float)
    n = len(numbers)

    Z = (1/(n-6))*np.matmul(np.matmul(np.transpose(R - n*B), A), (R-n*B))

    return Z


def run_test_III(numbers):
    # converting 
    new_lst = []
    for i in range(len(numbers)-1):
        if numbers[i] < numbers[i+1]:
            new_lst.append("<")
        else:
            new_lst.append(">")
    
    # counting total runs
    X = 0
    for i in range(len(new_lst)-1):
        if new_lst[i] != new_lst[i+1]:
            X+=1
        
    n = len(numbers)

    Z = (X-(2*n-1)/3)/np.sqrt((16*n-29)/90)

    return Z

def auto_corr(U, h):
    s = 0
    n = len(U)
    for i in range(n-h):
        s += U[i]*U[i+h]
    return s/(n-h)