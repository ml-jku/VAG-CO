import numpy as np
from matplotlib import pyplot as plt


def cosine_frac(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = 0., N_cycles = 3):
    if (epoch < N_warmup):
        T_curr = T_max
    else:
        T_high = fractional_schedule(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = T_min)
        T_curr = 0.5*(T_high*np.cos((2*np.pi*N_cycles + np.pi)*(epoch- N_warmup)/(epochs- N_warmup)) + T_high)

    return T_curr

def cosine(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = 0., N_cycles = 3):
    if (epoch < N_warmup):
        T_curr = T_max
    else:
        T_high = linear_decrease(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = T_min)
        T_curr = 0.5*(T_high*np.cos((2*np.pi*N_cycles + np.pi)*(epoch- N_warmup)/(epochs- N_warmup)) + T_high)

    return T_curr

def cosine_repeat(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = 0., N_cycles = 5):
    if (epoch < N_warmup):
        T_curr = T_max
    else:
        T_high = linear_decrease(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = T_min)
        T_curr = 0.5*(T_high*np.cos(2*np.pi*N_cycles*(epoch- N_warmup)/(epochs- N_warmup)) + T_high)

    return T_curr

def linear_decrease(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = 0.):
    if (epoch < N_warmup):
        T_curr = T_max
    elif (epoch >= N_warmup and epoch < epochs - N_equil - 1):  # (epoch % N_anneal == 0):
        T_curr = max([T_max - (T_max - T_min)* (epoch - N_warmup) / (N_anneal), T_min])
    else:
        T_curr = T_min
    return T_curr

def hyperbel_schedule(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = 0., alpha = 0.001):
    alpha = (epochs - epoch + 1)/epochs
    alpha_max = min([T_max, 0.99])
    alpha = min([alpha_max*alpha, alpha_max])
    T_curr = alpha/(1-alpha)
    return T_curr


def triangular_schedule(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min):
    N_increase = int(N_anneal/2)
    N_decrease = N_anneal - N_increase
    if (epoch < N_anneal/2):
        T_curr = max([T_max * epoch/ (N_increase), 0])
    else:  # (epoch % N_anneal == 0):
        T_curr = max([T_max - T_max * (epoch - N_increase) / (N_decrease), 0])

    return T_curr

def fractional_schedule(epoch, epochs, N_warmup, N_equil, N_anneal, T_max, T_min = 0., k = 6):
    if (epoch < N_warmup):
        T_curr = T_max
    elif (epoch >= N_warmup):  # (epoch % N_anneal == 0):
        T_curr = T_max/( k*(epoch-N_warmup)/N_anneal+ 1)

    return T_curr

if("__main__" == __name__):
    T = 1.
    epochs = 1100
    N_anneal = 1000
    N_warmup = 200
    epochs = N_anneal + N_warmup

    T_curr_list = []
    epoch_arr = np.arange(0, epochs)
    for epoch in epoch_arr:
        T_curr = cosine_frac(epoch, epochs, N_warmup, 0, N_anneal, T, T_min = 0., N_cycles = 3)
        T_curr_list.append(T_curr)

    plt.figure()
    plt.xlabel(r"$N_\mathrm{epoch}$", fontsize = 22)
    plt.ylabel("Temperature", fontsize = 15)
    plt.ylim(bottom = 0., top = 1.05)
    plt.plot(np.arange(0, epochs), T_curr_list)
    plt.axvline(x=N_warmup, c='red', linestyle='-.', linewidth=2.)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    T_curr_list = []
    epoch_arr = np.arange(epochs, 2*epochs)
    for epoch in epoch_arr:
        T_curr = cosine_frac(epoch, epochs, N_warmup, 0, N_anneal, T, T_min = 0., N_cycles = 3)
        T_curr_list.append(T_curr)

    plt.figure()
    plt.xlabel("epochs")
    plt.ylabel("Temperature")
    plt.plot(np.arange(0, epochs), T_curr_list)
    plt.show()