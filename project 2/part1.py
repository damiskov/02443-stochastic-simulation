import numpy as np
import matplotlib.pyplot as plt


def gillespie(N, max_time, beta, gamma):

    S, I, R, t = [N-10], [10], [0], [0]
    props=[beta*I[-1]*S[-1]/N, gamma*I[-1]]

    while t[-1] < max_time and (S[-1] + I[-1] >= 1) and sum(props) > 0:

        N = S[-1] + I[-1] + R[-1]


        propensities = [beta*I[-1]*S[-1]/N, gamma*I[-1]]

        

        total_propensity = sum(props)

        if total_propensity == 0:
            break
        
        tau = np.random.exponential(scale=1/propensities)

        t.append(t[-1]+tau)


        rand = np.random.uniform(0,1)

        # Susceptible becomes Infected
        if rand * total_propensity <= props[0]:
            S.append(S[-1] - 1)
            I.append(I[-1] + 1)
            R.append(R[-1])

        # Infected becomes Recovered
        # elif rand * prop_sum > props[0] and rand * prop_sum <= sum(props[:2]):
        else:
            S.append(S[-1])
            I.append(I[-1] - 1)
            R.append(R[-1] + 1)
    
    return S, I, R

def discrete_ODE_SIR(N, max_time, time_frame, beta, gamma):
    
    # Initial set up
    S, I, R = [N-10], [10], [0]
    t = np.linspace(0,max_time, int(max_time/time_frame))

    for _ in t[1:]:
        s_t, i_t, r_t = S[-1], I[-1], R[-1]

        S.append(s_t*(1 - time_frame*beta*i_t/N))
        I.append((beta*s_t/N + (1-gamma))*i_t*time_frame)
        R.append(r_t+gamma*i_t*time_frame)
    
    return S, I, R

def discrete_ebola_SIR_simulation(N=1000, max_time=1000, time_frame=1, beta = 0.4, gamma = 0.3):

    S, I, R = discrete_ODE_SIR(N, max_time, time_frame, beta, gamma)
    x = np.linspace(0, max_time, int(max_time/time_frame))

    plt.plot(x, S, color="cornflowerblue", label="S")
    plt.plot(x, I, color="lightcoral", label="I")
    plt.plot(x, R, color="yellowgreen", label="R")

    plt.grid()
    plt.legend()
    plt.title("Discrete ODE SIR Model")
    plt.ylabel("Population")
    plt.xlabel(r"$t$")

    plt.show()

def stochastic_ebola_SIR_simulation(N=1000, max_time=1000, beta = 0.4, gamma = 0.3):

    S, I, R = gillespie(N, max_time, beta, gamma)
    x = np.linspace(0, max_time, len(S))

    plt.plot(x, S, color="cornflowerblue", label="S")
    plt.plot(x, I, color="lightcoral", label="I")
    plt.plot(x, R, color="yellowgreen", label="R")

    plt.grid()
    plt.legend()
    plt.title("Stochastic SIR Model")

    plt.ylabel("Population")
    plt.xlabel(r"$t$")

    plt.show()

def discrete_covid_SIR_simulation(N=1000, max_time=1000, time_frame=1, beta = 0.5, gamma = 0.3):

    S, I, R = discrete_ODE_SIR(N, max_time, time_frame, beta, gamma)
    x = np.linspace(0, max_time, int(max_time/time_frame))

    plt.plot(x, S, color="cornflowerblue", label="S")
    plt.plot(x, I, color="lightcoral", label="I")
    plt.plot(x, R, color="yellowgreen", label="R")

    plt.grid()
    plt.legend()
    plt.title("Discrete ODE SIR Model (COVID-19)")
    plt.ylabel("Population")
    plt.xlabel(r"$t$")

    plt.show()


def stochastic_covid_SIR_simulation(N=1000, max_time=1000, beta = 0.5, gamma = 0.3):

    S, I, R = gillespie(N, max_time, beta, gamma)
    x = np.linspace(0, max_time, len(S))

    plt.plot(x, S, color="cornflowerblue", label="S")
    plt.plot(x, I, color="lightcoral", label="I")
    plt.plot(x, R, color="yellowgreen", label="R")

    plt.grid()
    plt.legend()
    plt.title("Stochastic SIR Model (COVID-19)")

    plt.ylabel("Population")
    plt.xlabel(r"$t$")

    plt.show()


if __name__ == "__main__":
    discrete_covid_SIR_simulation()
    stochastic_covid_SIR_simulation()

