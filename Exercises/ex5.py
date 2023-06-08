import numpy as np
import matplotlib.pyplot as plt
from math import log, floor
from scipy import stats


def monte_carlo_integral(U_i):
    return np.mea(np.exp(U_i))

def antithetic_var(U_i):
    Y_i = (np.exp(U_i)+np.exp([1-x for x in U_i]))/2
    return np.mean(Y_i)


def ex1():
    """1. Estimate the integral * by simulation (the crude Monte Carlo
    estimator). Use eg. an estimator based on 100 samples and present
    the result as the point estimator and a confidence interval."""


if __name__=="__main__":
    