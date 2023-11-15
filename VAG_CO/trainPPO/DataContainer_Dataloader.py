from torch.utils.data import Dataset
import numpy as np
import jraph
from timeit import time
import math

class ContainerDataset(Dataset):
    def __init__(self, config):
        self.mini_Nb = config["Train_params"]["PPO"]["mini_Nb"]
        self.mini_Hb = config["Train_params"]["PPO"]["mini_Hb"]
        self.mini_Sb = config["Train_params"]["PPO"]["mini_Sb"]

        self.Sb = config["Train_params"]["PPO"]["time_horizon"]
        self.time_horizon = self.Sb
        self.Nb = config["Train_params"]["n_basis_states"]
        self.Hb = config["Train_params"]["H_batch_size"]
        self.EngeryFunction = config["Ising_params"]["EnergyFunction"]

        self.Hb_list = np.arange(0, self.Hb)
        self.Nb_list = np.arange(0, self.Nb)

        self.H_splits = int(self.Hb / self.mini_Hb)
        self.N_splits = int(self.Nb / self.mini_Nb)

        ### TODO for every H_idx there shoud be mini_Nb and mini_Sb indices
        self.tuple_idxs = []

    def normalize_advantage(self, Hb_advantages):
        Hb_batched_advantages_arr = np.concatenate(Hb_advantages, axis=-2)
        self.mean_adv = np.mean(Hb_batched_advantages_arr)
        self.std_adv = np.std(Hb_batched_advantages_arr)

    def shuffle_list(self):
        #np.random.shuffle(self.Hb_list)
        np.random.shuffle( self.Nb_list)
        #self.mini_Hb_lists = np.array_split(self.Hb_list, self.H_splits)
        self.mini_Nb_lists = np.array_split(self.Nb_list, self.N_splits)
        return self.mini_Nb_lists

    def reshuffle(self):
        ### TODO replaye this by
        start_shuffle = time.time()
        self.mini_Nb_lists = self.shuffle_list()

        #np.random.shuffle(self.H_idx)

        #Hb_advantages = []
        self.Hb_mini_Sb_lists = []
        self.tuple_idxs = []
        for i in range(len(self.H_idx)):
            H_idx = self.H_idx[i]
            graph_time_horizon = self.DataContainers[H_idx].max_steps

            time_idxs = np.arange(0, graph_time_horizon)
            np.random.shuffle(time_idxs)
            if(graph_time_horizon > self.mini_Sb):
                graph_mini_Sb_lists = np.split(time_idxs, [(i+1)*self.mini_Sb for i in range(math.ceil(graph_time_horizon/self.mini_Sb) - 1)], axis=-1)
            else:
                graph_mini_Sb_lists = [time_idxs]

            N_idx, S_idx = np.meshgrid(np.arange(0, len(self.mini_Nb_lists)),np.arange(0, len(graph_mini_Sb_lists)))
            self.Hb_mini_Sb_lists.append(graph_mini_Sb_lists)

            N_idx = np.ravel(N_idx)
            S_idx = np.ravel(S_idx)

            # print([len(el) for el in graph_mini_Sb_lists], graph_time_horizon, self.mini_Sb)
            # if(np.any(np.array([len(el) for el in graph_mini_Sb_lists]) == 0)):
            #     print("here")

            self.tuple_idxs.extend([ (i, H_idx,N_idx[j], S_idx[j]) for j in range(N_idx.shape[0]) ])

        #self.normalize_advantage(Hb_advantages)
        end_shuffle = time.time()
        print("shuffling time", end_shuffle - start_shuffle)

    def overwrite_data(self, DataContainers):
        self.DataContainers = DataContainers
        H_idx = np.arange(0, len(DataContainers))
        self.H_idx = np.ravel(H_idx)

    def __len__(self):
        #return len(self.jraph_graph_list)
        return len(self.tuple_idxs)

    def index_arr(self, Sb_Nb_arr, S_idxs, N_idxs):
        minib_arr = Sb_Nb_arr[S_idxs[:, np.newaxis], N_idxs]

        return minib_arr

    def concatenate_dicts(self, minib_dict, additional_dict):

        concat_dict = {}
        for key in minib_dict:
            data1 = minib_dict[key]
            data2 = additional_dict[key]

            if(key == "graphs" or key == "compl_graphs"):
                concat_dict[key] = jraph.batch_np([data1, data2])
            elif(key == "arrays"):
                #print(data1.shape, data2.shape)
                concat_dict[key] = {}
                for kkey in data1:
                    concat_dict[key][kkey] = np.concatenate([data1[kkey], data2[kkey]], axis = 1)
            else:
                concat_dict[key] = np.concatenate([data1, data2], axis = 1)

        return concat_dict

    def add_missing_graphs(self, n_missing_graphs):
        rand_idx = np.random.randint(0, len(self.tuple_idxs))
        (i, H_idx, N_idx, S_idx) = self.tuple_idxs[rand_idx]
        datacontainer = self.DataContainers[H_idx]
        N_idxs = self.mini_Nb_lists[N_idx]
        S_idxs = self.Hb_mini_Sb_lists[i][S_idx]
        np.random.shuffle(S_idxs)

        chosen_S_idxs = S_idxs[0:n_missing_graphs]

        additional_minib_dict = self.create_dict(datacontainer, N_idxs, chosen_S_idxs)
        return additional_minib_dict

    def create_dict(self, datacontainer, N_idxs, S_idxs):
        data_dict = datacontainer.get_Data_dict()

        minib_dict = {}
        minib_dict["arrays"] = {}
        for key in data_dict:
            if (key != "graphs" and key != "external_fields" and key != "compl_graphs"):
                arr = data_dict[key]

                mini_Sb_Nb_arr = self.index_arr(arr, S_idxs, N_idxs)

                mini_Nb_Sb_arr = np.swapaxes(mini_Sb_Nb_arr, 0, 1)
                # if(key == "advantage"):
                #     minib_dict["arrays"][key] = (mini_Nb_Sb_arr - self.mean_adv)/(self.std_adv+1e-10)
                # else:
                minib_dict["arrays"][key] = mini_Nb_Sb_arr
            elif (key == "graphs" or key == "compl_graphs"):
                graph_list = data_dict[key]
                graph_list_func = lambda S_idx: graph_list[S_idx]
                mini_Sb_graphs = list(map(graph_list_func, S_idxs))
                minib_dict[key] = jraph.batch_np(mini_Sb_graphs)
            else:
                ext_field_list = data_dict[key]

                t_Sb_Nb_idxs = np.repeat(S_idxs[:, np.newaxis], len(N_idxs), axis=-1)
                N_Sb_Nb_idxs = np.repeat(N_idxs[np.newaxis, :], len(S_idxs), axis=0)
                t_Nb_Sb_idxs = np.swapaxes(t_Sb_Nb_idxs, 0, 1)
                N_Nb_Sb_idxs = np.swapaxes(N_Sb_Nb_idxs, 0, 1)


                ext_field_list_func = lambda S_idx, N_idx: ext_field_list[S_idx][N_idx]


                mini_Sb_Nb_external_fields_list = list(
                        map(ext_field_list_func, np.ravel(t_Nb_Sb_idxs), np.ravel(N_Nb_Sb_idxs)))


                #print("len N-idxs",len(N_idxs), "len S-idx", len(S_idxs), len(mini_Sb_Nb_external_fields_list) )
                minib_ext_fields = np.concatenate(mini_Sb_Nb_external_fields_list, axis=0)
                res = np.reshape(minib_ext_fields, (len(N_idxs), int(minib_ext_fields.shape[0] / len(N_idxs)), minib_ext_fields.shape[-1]))
                minib_dict[key] = res

        return minib_dict

    def __getitem__(self, idx):

        (i, H_idx, N_idx, S_idx) = self.tuple_idxs[idx]
        datacontainer = self.DataContainers[H_idx]
        N_idxs = self.mini_Nb_lists[N_idx]
        S_idxs = self.Hb_mini_Sb_lists[i][S_idx]

        Sb_goal_graphs = self.mini_Sb
        missing_graphs = Sb_goal_graphs - len(S_idxs)

        minib_dict = self.create_dict(datacontainer, N_idxs, S_idxs)

        # print("missign graphs")
        # print(missing_graphs)
        # print(len(S_idxs), Sb_goal_graphs)
        while(missing_graphs > 0):
            additional_minib_dict = self.add_missing_graphs(missing_graphs)
            minib_dict = self.concatenate_dicts(minib_dict, additional_minib_dict)
            missing_graphs = Sb_goal_graphs - minib_dict["arrays"]["log_probs"].shape[1]
            # print("still missing graphs", minib_dict["arrays"]["log_probs"].shape)
            # print(missing_graphs)

        return minib_dict



### TODO define Collate function
def concatenate_arrays(array_list):

    key_list = list(array_list[0].keys())

    Hb_arr_dict = {}
    for key in key_list:
        Hb_arr = np.concatenate([array_dict[key] for array_dict in array_list], axis = 1)
        Hb_arr_dict[key] = Hb_arr


    return Hb_arr_dict

import jax.numpy as jnp
from jraph_utils import utils as jutils

def collate_data_dict(Hb_data_dict):
    Hb_ext_field_list = np.concatenate([data_dict["external_fields"] for data_dict in Hb_data_dict], axis=1)
    minib_H_graphs = jraph.batch_np([data_dict["graphs"] for data_dict in Hb_data_dict])
    array_list = [data_dict["arrays"] for data_dict in Hb_data_dict]

    Hb_array_dict = concatenate_arrays(array_list)

    minib_value_target = Hb_array_dict["value_targets"]
    minib_A_k = Hb_array_dict["advantage"]
    minib_actions = Hb_array_dict["actions"]
    minib_log_probs = Hb_array_dict["log_probs"]
    masks = Hb_array_dict["masks"]

    return minib_H_graphs, masks, minib_actions, minib_A_k, minib_log_probs, minib_value_target, Hb_ext_field_list


### TODO test Dataloader