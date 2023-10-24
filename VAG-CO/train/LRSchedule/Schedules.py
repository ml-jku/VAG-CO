import numpy as np


def cosine_warmup(epoch, epochs, N_warmup, N_equil, N_anneal, lr_max, lr_min =5*10 ** -4, N_cycles = 5, N_w_cycles = 2,N_eq_cycles = 2,factor = 0.8):
    mean_lr = (lr_max+lr_min)/2
    amplitude = factor * (lr_max - mean_lr)
    if (epoch < N_warmup):
        #amplitude = factor * lr_max
        lr_curr = cosine_linear(epoch, N_warmup, amplitude = amplitude, mean_lr= lr_max)
    elif(epoch < N_warmup + N_anneal):
        lr_high = linear_decrease(epoch, epochs, N_warmup, N_equil, N_anneal, lr_max, lr_min= mean_lr)
        #amplitude = factor * lr_high
        periodicity = int(N_anneal/N_cycles)
        lr_curr = cosine_linear((epoch- N_warmup)%periodicity, periodicity, amplitude = amplitude, mean_lr= lr_high)
    else:
        periodicity = int(N_equil/N_eq_cycles)
        lr_high = linear_decrease(epoch, epochs, N_warmup, N_equil, N_anneal, lr_max, lr_min=mean_lr)
        #amplitude = 2*factor * lr_high
        lr_curr = cosine_linear((epoch- N_warmup- N_anneal)%periodicity,  periodicity, amplitude = amplitude, mean_lr= lr_high)

    return lr_curr

def cosine_linear(n, n_steps, amplitude = 1, mean_lr = 1. , factor = 0.1):
    cos_periodicity = n_steps - 2*factor*n_steps
    lin_periodicity = factor*n_steps
    if(n < lin_periodicity):
        lr_curr = mean_lr + amplitude * n/lin_periodicity
    elif(n_steps - lin_periodicity < n):
        lr_curr = mean_lr - amplitude + amplitude * (n-lin_periodicity-cos_periodicity)/lin_periodicity
    else:
        lr_curr = mean_lr + amplitude*np.cos(np.pi*(n-lin_periodicity)/cos_periodicity)
    return lr_curr


def cosine(epoch, epochs, N_warmup, N_equil, N_anneal, lr_max, lr_min =10 ** -4, N_cycles = 10, N_w_cycles = 2,N_eq_cycles = 2):
    amplitude = 0.5 * (lr_max - lr_min)
    if (epoch < N_warmup):
        lr_curr = amplitude*np.sin(2*np.pi*N_w_cycles*epoch/N_warmup) + lr_max
    elif(epoch < N_warmup + N_anneal):
        lr_high = linear_decrease(epoch, epochs, N_warmup, N_equil, N_anneal, lr_max, lr_min= lr_min)
        lr_curr = amplitude*np.sin(2*np.pi*N_cycles*(epoch- N_warmup)/(N_anneal)) + lr_high
    else:
        lr_high = linear_decrease(epoch, epochs, N_warmup, N_equil, N_anneal, lr_max, lr_min=lr_min)
        lr_curr = amplitude*np.sin(2*np.pi*N_eq_cycles*(epoch- N_warmup- N_anneal)/(N_equil)) + lr_high

    return lr_curr

def linear_decrease(epoch, epochs, N_warmup, N_equil, N_anneal, lr_max, lr_min = 0.):
    if (epoch < N_warmup):
        lr_curr = lr_max
    elif (epoch >= N_warmup and epoch < epochs - N_equil - 1):  # (epoch % N_anneal == 0):
        lr_curr = max([lr_max - (lr_max - lr_min) * (epoch - N_warmup) / (N_anneal), lr_min])
    else:
        lr_curr = lr_min
    return lr_curr

if(__name__ == "__main__"):
    from matplotlib import pyplot as plt
    N_warmup = 1000
    N_anneal = 5000
    N_equil = 2000
    max_epochs = N_warmup + N_equil + N_anneal
    epochs = np.arange(0,max_epochs)
    schedule = []
    for epoch in epochs:
        lr = cosine_warmup(epoch,max_epochs, N_warmup, N_equil, N_anneal, 10**-3, lr_min=10**-4, N_cycles=6)
        schedule.append(lr)
    plt.figure()
    plt.plot(epochs, schedule)
    plt.show()

    pass