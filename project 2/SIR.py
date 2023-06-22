import numpy as np

class SIR:

    def __init__(self, name, beta, gamma):
        self.Name = name
        self.Beta = beta
        self.Gamma = gamma
    
    def __str__(self):
        s = f"Name: {self.Name}\nBeta: {self.Beta}\nGamma: {self.Gamma}"
        return s
    
    def simulate_ODE(self, population, initial_infected, max_time):
        """
         Discrete simulation disease progression:
        - Population
        - initial number of infected
        - time over which to model
        
        Returns:
        - S, I, R
        """

        # initial set up
        N = population
        S, I, R = [N-initial_infected], [initial_infected], [0]
        t = np.linspace(0, max_time, max_time)

        for _ in t[1:]:
            S.append(S[-1]*(1 - self.Beta*I[-1]/N))
            I.append((self.Beta*S[-1]/N + (1-self.Gamma))*I[-1])
            R.append(R[-1]+self.Gamma*I[-1])

        return S, I, R
    
    def simulate_stochastic(self, population, initial_infected, max_time):
        """
        Stochastic simulation of disease progression
        - Population
        - initial number of infected
        - time over which to model
        
        Returns:
        - S, I, R
        """
        
        # iota = 0.01 - external infection rate
        t = np.linspace(0,max_time,max_time)
        S, I, R, Re = [population - initial_infected], [initial_infected], [0], [self.getRe(population, population)]

        for _ in t[1::]:

            l = self.Beta*(I[-1])/population # Rate of new infections
    
            infection_probability = 1.0 - np.exp(-l) # Probability of infection 
            recovery_probability = 1.0 - np.exp(-self.Gamma) # Probability of recovery
    
            infection = np.random.binomial(S[-1],infection_probability) # Number of new infected
            recovery = np.random.binomial(I[-1],recovery_probability) # Number of new recovered

            S.append(S[-1]-infection)
            I.append(I[-1]+infection-recovery)
            R.append(R[-1]+recovery)
            Re.append(self.getRe(S[-1], population))
            
        return S, I, R

    def getR0(self):
        return self.Beta/self.Gamma

    def getRe(self, num_susceptible, population):
        return self.getR0()*num_susceptible/population
    
