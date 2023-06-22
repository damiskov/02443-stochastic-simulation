import numpy as np
import matplotlib.pyplot as plt

class SEIR:

    def __init__(self, name, beta, gamma, sigma):
        self.name = name
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
    
    def deterministic_simulation(self, S0, I0, E0, R0, time):
        """
        Deterministic simulation of the SIR model
        
        Parameters
        ----------
        S0 : int
            Initial number of susceptible individuals
        I0 : int
            Initial number of infected individuals
        R0 : int
            Initial number of recovered individuals
        time : int
            Number of days to simulate
        
        Returns
        -------
        S : list
            List of the number of susceptible individuals each day
        I : list
            List of the number of infected individuals each day
        R : list
            List of the number of recovered individuals each day
            """
        S, I, E, R = [S0], [I0], [E0], [R0]
        dt = 1
        N = S0 + I0 + E0 + R0
        for _ in range(time):
            S.append(S[-1] + dt*(-self.beta*S[-1]*I[-1]/N))
            E.append(E[-1] + dt*(self.beta*S[-2]*I[-1]/N - self.sigma*E[-1]))
            I.append(I[-1] + dt*(self.sigma*E[-1] - self.gamma*I[-1]))
            R.append(R[-1] + dt*(self.gamma*I[-1]))
        return S, E, I, R
    
    def stochastic_simulation(self, S0, I0, E0, R0, time):
        """
        Stochastic simulation of the SIR model
        
        Parameters
        ----------
        S0 : int
            Initial number of susceptible individuals
        I0 : int
            Initial number of infected individuals
        R0 : int
            Initial number of recovered individuals
        time : int
            Number of days to simulate
        
        Returns
        -------
        S : list
            List of the number of susceptible individuals each day
        I : list
            List of the number of infected individuals each day
        R : list
            List of the number of recovered individuals each day
            """
        
        S, I, E, R = [S0], [I0], [E0], [R0]
        N = S0 + I0 + E0 + R0
        for _ in range(1, time+1):
            
            l = self.beta*(I[-1])/N # Rate of new exposures due to infected individuals

            exposure_probability = 1.0 - np.exp(-l) # Probability of exposure
            infection_probability = 1.0 - np.exp(-l) # Probability of infection 
            recovery_probability = 1.0 - np.exp(-self.Gamma) # Probability of recovery

            exposed = np.random.binomial(S[-1], exposure_probability) # Number of new exposed
            infection = np.random.binomial(S[-1],infection_probability) # Number of new infected
            recovery = np.random.binomial(I[-1],recovery_probability) # Number of new recovered

            S.append(S[-1]-exposed)
            E.append(E[-1]-infection+exposed)
            I.append(I[-1]+infection-recovery)
            R.append(R[-1]+recovery)
            
        return S, I, R
