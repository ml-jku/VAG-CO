import numpy as np
from train.AnnealingSchedules import Schedules

class MovingAverage:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        self.step_count = 0
        self.mean_value = None
        self.std_value = None

    def update_mov_averages(self, data):
        if(self.alpha == -1):
            return 0., 1.
        else:
            self.mean_value = self.mean_step(data)
            self.std_value = self.std_step(data)
            self.step_count += 1
        return self.mean_value, self.std_value

    def update_rule(self, value, data, factor):
        if(self.step_count == 0):
            value = data
        else:
            value = factor * data + (1-factor)*value

        return value

    def mean_step(self, data):
        mean_data = np.mean(data)
        return self.update_rule(self.mean_value, mean_data, self.alpha)

    def std_step(self, data):
        std_data = np.std(data)
        return self.update_rule(self.std_value, std_data, self.beta)





if(__name__ == "__main__"):
    from matplotlib import pyplot as plt

    alphas = [0.99, 0.9, 0.8,0.5, 0.4,0.3, 0.2, 0.15, 0.1, 0.05,0.01]
    for alpha in alphas:
        T = 1.
        N_anneal = 2000
        N_warmup = 400
        epochs = N_anneal + N_warmup
        MovAvg = MovingAverage(alpha, alpha)

        T_curr_list = []
        mov_data = []
        buffer = []
        epoch_arr = np.arange(0, epochs)
        for epoch in epoch_arr:
            T_curr = Schedules.cosine_frac(epoch, epochs, N_warmup, 0, N_anneal, T, T_min=0., N_cycles=3)
            T_curr = T_curr + np.exp(-epoch/(epochs/4))*np.random.uniform()
            T_curr_list.append(T_curr)
            buffer.append(T_curr)

            if(len(buffer) > 0.001*epochs):
                mean, std = MovAvg.update_mov_averages(np.array(buffer))
                mov_data.append([epoch, mean, std])
                buffer = []

        plt.figure()
        plt.title(str(alpha))
        plt.xlabel(r"$N_\mathrm{epoch}$", fontsize = 22)
        plt.ylabel("Temperature", fontsize = 15)
        plt.ylim(bottom = 0., top = 1.05)
        plt.plot(np.arange(0, epochs), T_curr_list)
        plt.plot([data[0] for data in mov_data], [data[1] for data in mov_data], label = "mov average")
        plt.plot([data[0] for data in mov_data], [data[2] for data in mov_data], label = "std average")
        plt.axvline(x=N_warmup, c='red', linestyle='-.', linewidth=2.)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()


