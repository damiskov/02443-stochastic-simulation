from SIR import SIR
from SEIR import SEIR
import numpy as np
import matplotlib.pyplot as plt



def test_COVID():
    sier = SEIR("COVID-19", 0.328, 0.1, 0.1)
    s, e, i, r = sier.deterministic_simulation(1000, 1, 0, 0, 500)
    plt.style.use("ggplot")
    plt.plot(s, label="Susceptible", color='darkblue')
    plt.plot(e, label="Exposed", color="darkorange")
    plt.plot(i, label="Infected", color='darkred')
    plt.plot(r, label="Recovered", color='darkgreen')
    plt.legend()
    plt.show()

if __name__=="__main__":
    test_COVID()