import numpy as np
import matplotlib.pyplot as plt
from population import Population


def simulation1():

    germany = Population("Germany", 1e4, 1e2, 0, 0, 0.5, 0.05, 0.1)
    uk = Population("UK",1e4, 2e2, 0, 0, 0.5, 0.05, 0.1)
    france = Population("France", 1e4, 5e2, 0, 0, 0.5, 0.05, 0.1)

    # adding connections

    germany.add_connection(uk, 0.1)
    germany.add_connection(france, 0.1)
    uk.add_connection(germany, 0.1)
    uk.add_connection(france, 0.1)
    france.add_connection(germany, 0.1)
    france.add_connection(uk, 0.1)

    # running simulation

    for _ in range(1000):
        germany.update()
        uk.update()
        france.update()


    # plotting
    # list of different colours
    # 3  shades of blue, 3 shades of orange,  3  shades of red, 3  shades of green

    colours = [["darkblue", "cornflowerblue", "cyan"], ["darkorange", "bisque", "orange"], ["darkred", "lightcoral", "red"], ["darkgreen", "palegreen", "springgreen"]]

    countries = [germany, uk, france]

    for i in range(3):
        plt.style.use("ggplot")
        plt.plot(countries[i].S, label=f"{countries[i].name} S", c=colours[0][i], alpha=0.5)
        plt.plot(countries[i].E, label=f"{countries[i].name} E", c=colours[1][i], alpha=0.5)
        plt.plot(countries[i].I, label=f"{countries[i].name} I", c=colours[2][i], alpha=0.5)
        plt.plot(countries[i].R, label=f"{countries[i].name} R", c=colours[3][i], alpha=0.5)

        print(f"{countries[i].name} S: {countries[i].S[-1]}")
        print(f"{countries[i].name} E: {countries[i].E[-1]}")
        print(f"{countries[i].name} I: {countries[i].I[-1]}")
        print(f"{countries[i].name} R: {countries[i].R[-1]}")


    plt.legend()
    plt.show()

if __name__ == "__main__":
    simulation1()


    
    