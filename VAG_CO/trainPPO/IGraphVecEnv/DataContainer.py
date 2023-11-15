

import numba
import numpy as np


class DataContainer:

    def __init__(self, cfg,  time_horizon, Nb, num_spins, lam, mov_reward, n_sampled_spins = 1):
        self.EnergyFunction = cfg["Ising_params"]["EnergyFunction"]
        self.gamma = 1.
        self.lam = lam
        self.Nb = Nb

        if(num_spins%n_sampled_spins != 0):
            print(num_spins%n_sampled_spins, num_spins, n_sampled_spins)
            ValueError("somehting is wrong here")

        if(time_horizon < num_spins/n_sampled_spins):
            self.max_steps = time_horizon
        else:
            self.max_steps = int(num_spins/n_sampled_spins)

        # if(self.max_steps == 0):
        #     print(self.max_steps, num_spins, n_sampled_spins)
        #     print("here")

        self.values = np.zeros((self.max_steps+1, Nb), dtype = np.float32)
        self.rewards = np.zeros((self.max_steps, Nb), dtype = np.float32)
        self.log_probs = np.zeros((self.max_steps , Nb), dtype = np.float32)
        self.actions = np.zeros((self.max_steps, Nb), dtype = np.uint32)
        self.masks = np.zeros((self.max_steps, Nb, 2**n_sampled_spins), dtype = np.uint8)
        self.external_fields = np.zeros((self.max_steps, Nb), dtype = np.float32)
        #self.advantage = np.zeros((time_horizon + 1, Nb))
        self.Jgraphs = []
        self.compl_Jgraphs = []
        self.ext_field_list = []

        self.counter = 0

        self.mean_reward = mov_reward[0]
        self.std_reward = mov_reward[1]

    def create_dict(self):
        self.dict = {}
        self.dict["value_targets"] = self.value_target
        self.dict["advantage"] = self.advantage
        self.dict["log_probs"] = self.log_probs
        self.dict["external_fields"] = self.ext_field_list
        self.dict["graphs"] = self.Jgraphs
        self.dict["actions"] = self.actions
        self.dict["masks"] = self.masks

    def get_Data_dict(self):
        self.create_dict()
        return self.dict

    def append(self, Jgraph, external_fields, values, rewards, log_probs, actions, masks):
        time_step = self.counter
        if(time_step == self.max_steps):
            self.values[time_step,:] = values
            #self.value_target, self.advantage = calc_traces(self.values, self.rewards)
        else:
            self.Jgraphs.append(Jgraph)
            self.values[time_step,:] = values
            self.log_probs[time_step,:] = log_probs
            self.rewards[time_step,:] = rewards
            self.actions[time_step,:] = actions
            self.masks[time_step] = masks
            self.ext_field_list.append(external_fields)
            # print("time step", time_step, self.max_steps)
        # if(Jgraph.nodes.shape[0] != complJgraph.nodes.shape[0]):
        #     print("here")
        # print("compl", Jgraph.nodes.shape,  "norm",complJgraph.nodes.shape)
        # print("compl", Jgraph.edges.shape, "norm", complJgraph.edges.shape)

        self.counter += 1

    def update_traces(self):
        self.rewards = (self.rewards - self.mean_reward)/self.std_reward

        self.value_target, self.advantage = calc_traces(self.values, self.rewards, lam = self.lam)

@numba.jit
def calc_traces(values, rewards, gamma = 1. , lam = 0.95):
    max_steps = rewards.shape[0]
    Nb = rewards.shape[1]
    advantage = np.zeros((max_steps +1 , Nb))
    for t in range(max_steps):
        idx = max_steps - t - 1
        delta = rewards[idx] + gamma *values[idx+1] - values[idx]
        advantage[idx] = delta + gamma*lam *advantage[idx+1]

    value_target = (advantage + values)[0:max_steps]

    advantage = advantage[0:max_steps]

    return value_target, advantage[0:max_steps]