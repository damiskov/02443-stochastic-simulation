import numpy as np

class Population:
    """
    A class to represent a population consisting of SEIR individuals.
    """

    def __init__(self, name, S0, E0, I0, R0, beta, gamma, sigma, linked_populations = {}):
        """
        Constructor for Population class.
        """
        self.name = name
        self.S = [S0]
        self.E = [E0]
        self.I = [I0]
        self.R = [R0]
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.linked_populations = linked_populations

    def add_exposed(self, n):
        """
        Add exposed individuals to the population.
        """
        self.E[-1] += n
    
    def add_connection(self, pop, prob):
        """
        Add linked populations to the population.
        """
        self.linked_populations[pop] = prob

    def simulate_travel(self):
        """
        Stochastically simulate the travel of exposed people between all linked populations.
        """
        for pop, prob in self.linked_populations.items():
            # Modelling number of people travelling from this population to the linked population using poisson distribution
            # with probability of success being the product of the proportion of exposed people in this population and the probability of travelling to a certain population.
            n = np.random.poisson(self.E[-1]*prob)
            self.E[-1] -= n
            pop.add_exposed(n)

    def update(self):
        """
        Simulate one day of disease progression in the population (stochastically).
        """

        self.simulate_travel()

        N = (self.S[-1]+self.E[-1]+self.I[-1]+self.R[-1])
        l = self.beta*(self.I[-1])/N # Rate of new exposures due to infected individuals
        # rate of new infections due to exposed individuals
        r = self.sigma*(self.E[-1])/N




        exposure_probability = 1.0 - np.exp(-l) # Probability of exposure due to infected contact
        infection_probability = 1.0 - np.exp(-r) # Probability of exposed->infected
        recovery_probability = 1.0 - np.exp(-self.gamma) # Probability of recovery

        # print("exposure_probability: ", exposure_probability)
        # print("infection_probability: ", infection_probability)
        # print("recovery_probability: ", recovery_probability)


        exposed = np.random.binomial(self.S[-1], exposure_probability) # Number of new exposed
        infection = np.random.binomial(self.E[-1],infection_probability) # Number of new infected
        recovery = np.random.binomial(self.I[-1],recovery_probability) # Number of new recovered


        # print("exposed: ", exposed)
        # print("infection: ", infection)
        # print("recovery: ", recovery)
        # exit()

        self.S.append(self.S[-1]-exposed)
        self.E.append(self.E[-1]-infection+exposed)
        self.I.append(self.I[-1]+infection-recovery)
        self.R.append(self.R[-1]+recovery)


    def get_SEIR(self):
        """
        Return the number of susceptible, exposed, infected and recovered people in the population.
        """
        return self.S, self.E, self.I, self.R
    



