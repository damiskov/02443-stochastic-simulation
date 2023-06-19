import numpy as np
import matplotlib.pyplot as plt

def discrete_SIR(N, max_time, time_frame, beta, gamma):
    
    # Initial set up
    S, I, R = [N-1], [1], [0]
    t = np.linspace(0,max_time, int(max_time/time_frame))

    for _ in t[1:]:
        s_t, i_t, r_t = S[-1], I[-1], R[-1]

        S.append(s_t*(1 - time_frame*beta*i_t/N))
        I.append((beta*s_t/N + (1-gamma))*i_t*time_frame)
        R.append(R[-1]+gamma*i_t*time_frame)
    
    return S, I, R

def ebola_SIR_model(N=1000, max_time=100, time_frame=1, beta = 0.4, gamma = 0.3):

    S, I, R = discrete_SIR(N, max_time, time_frame, beta, gamma)
    x = np.linspace(0, max_time, int(max_time/time_frame))
    # print(len(x))
    # exit()
    plt.plot(x, S, color="cornflowerblue", label="S", marker="x")
    plt.plot(x, I, color="lightcoral", label="I", marker="x")
    plt.plot(x, R, color="yellowgreen", label="R", marker="x")
    plt.grid()
    plt.legend()
    # plt.ylabel()
    plt.xlabel(r"$t$")
    plt.show()

if __name__=="__main__":
    ebola_SIR_model()
    
    
        


