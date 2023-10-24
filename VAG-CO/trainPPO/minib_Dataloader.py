from torch.utils.data import Dataset
import numpy as np
from trainPPO.npBuffer import DataBuffer
import jraph

class minibDataset(Dataset):
    def __init__(self, config):
        self.mini_Nb = config["Train_params"]["PPO"]["mini_Nb"]
        self.mini_Hb = config["Train_params"]["PPO"]["mini_Hb"]
        self.mini_Sb = config["Train_params"]["PPO"]["mini_Sb"]

        self.Sb = config["Train_params"]["PPO"]["time_horizon"]
        self.time_horizon = self.Sb
        self.Nb = config["Train_params"]["n_basis_states"]
        self.Hb = config["Train_params"]["H_batch_size"]

        self.ReplayBuffer = DataBuffer(self.Hb, self.Nb, self.Sb, self.mini_Hb, self.mini_Nb, self.mini_Sb)
        self.length = self.ReplayBuffer.H_splits*self.ReplayBuffer.N_splits*self.ReplayBuffer.S_splits
        self.reshuffle()

        H_idx, N_idx, S_idx = np.meshgrid(np.arange(0, len(self.mini_Hb_lists)), np.arange(0, len(self.mini_Nb_lists)), np.arange(0, len(self.mini_Sb_lists)))
        H_idx = np.ravel(H_idx)
        N_idx = np.ravel(N_idx)
        S_idx = np.ravel(S_idx)
        np.random.shuffle(H_idx)
        np.random.shuffle(N_idx)
        np.random.shuffle(S_idx)

        self.tuple_idxs = [ (H_idx[i], N_idx[i], S_idx[i]) for i in range(self.length)]

    def reshuffle(self):
        (self.mini_Hb_lists, self.mini_Nb_lists), (H_order, N_order) = self.ReplayBuffer.split_in_minibatches()

        time_idxs = np.arange(0, self.time_horizon)
        np.random.shuffle(time_idxs)
        self.mini_Sb_lists = np.array_split(time_idxs, int(self.time_horizon / self.mini_Sb), axis=-1)

    def overwrite_data(self, Sb_Hb_H_graph_list, Sb_Hb_Nb_ext_fields_list , Sb_Hb_Nb_A_k, Sb_Hb_Nb_log_probs, Sb_Hb_Nb_value_target, Sb_Hb_Nb_actions):
        self.list_indexing_func = lambda S_idx, H_idx: Sb_Hb_H_graph_list[S_idx][H_idx]
        self.list_ext_fields_indexing_func = lambda S_idx, H_idx, N_idx: Sb_Hb_Nb_ext_fields_list[S_idx][H_idx][N_idx]

        self.Sb_Hb_Nb_A_k = Sb_Hb_Nb_A_k
        self.Sb_Hb_Nb_log_probs = Sb_Hb_Nb_log_probs
        self.Sb_Hb_Nb_value_target = Sb_Hb_Nb_value_target
        self.Sb_Hb_Nb_actions = Sb_Hb_Nb_actions

    def __len__(self):
        #return len(self.jraph_graph_list)
        return self.length

    def __getitem__(self, idx):
        (H_idx, N_idx, S_idx) = self.tuple_idxs[idx]
        Hb_idxs = self.mini_Hb_lists[H_idx]
        Nb_idxs = self.mini_Nb_lists[N_idx]
        Sb_idxs = self.mini_Sb_lists[S_idx]

        t_Sb_Hb_idxs = np.repeat(Sb_idxs[:, np.newaxis], len(Hb_idxs), axis=-1)
        t_Sb_Hb_Nb_idxs = np.repeat(t_Sb_Hb_idxs[:, :, np.newaxis], len(Nb_idxs), axis=-1)

        N_Sb_Nb_idxs = np.repeat(Nb_idxs[np.newaxis, :], len(Sb_idxs), axis=0)
        N_Sb_Hb_Nb_idxs = np.repeat(N_Sb_Nb_idxs[:, np.newaxis, :], len(Hb_idxs), axis=1)

        H_Sb_Hb_idxs = np.repeat(Hb_idxs[np.newaxis, :], len(Sb_idxs), axis=0)
        H_Sb_Hb_Nb_idxs = np.repeat(H_Sb_Hb_idxs[:, :, np.newaxis], len(Nb_idxs), axis=-1)

        H_Sb_Nb_Hb_idxs = np.swapaxes(H_Sb_Hb_Nb_idxs, 1, 2)
        H_Nb_Sb_Hb_idxs = np.swapaxes(H_Sb_Nb_Hb_idxs, 0, 1)

        N_Sb_Nb_Hb_idxs = np.swapaxes(N_Sb_Hb_Nb_idxs, 1, 2)
        N_Nb_Sb_Hb_idxs = np.swapaxes(N_Sb_Nb_Hb_idxs, 0, 1)

        t_Sb_Nb_Hb_idxs = np.swapaxes(t_Sb_Hb_Nb_idxs, 1, 2)
        t_Nb_Sb_Hb_idxs = np.swapaxes(t_Sb_Nb_Hb_idxs, 0, 1)

        minib_H_graphs_list = list(map(self.list_indexing_func, np.ravel(t_Sb_Hb_idxs), np.ravel(H_Sb_Hb_idxs)))
        batched_minib_H_graphs = jraph.batch_np(minib_H_graphs_list)

        minib_ext_fields_list = list(map(self.list_ext_fields_indexing_func, np.ravel(t_Nb_Sb_Hb_idxs), np.ravel(H_Nb_Sb_Hb_idxs), np.ravel(N_Nb_Sb_Hb_idxs)))
        minib_ext_fields = np.concatenate(minib_ext_fields_list, axis=0)
        minib_ext_fields = np.reshape(minib_ext_fields, (len(Nb_idxs), batched_minib_H_graphs.nodes.shape[0], 1))

        minib_A_k = self.Sb_Hb_Nb_A_k[Sb_idxs[:, np.newaxis, np.newaxis], Hb_idxs[:, np.newaxis], Nb_idxs]
        minib_log_probs = self.Sb_Hb_Nb_log_probs[Sb_idxs[:, np.newaxis, np.newaxis], Hb_idxs[:, np.newaxis], Nb_idxs]
        minib_value_target = self.Sb_Hb_Nb_value_target[Sb_idxs[:, np.newaxis, np.newaxis], Hb_idxs[:, np.newaxis], Nb_idxs]
        minib_actions = self.Sb_Hb_Nb_actions[Sb_idxs[:, np.newaxis, np.newaxis], Hb_idxs[:, np.newaxis], Nb_idxs]

        return batched_minib_H_graphs, minib_ext_fields, minib_A_k, minib_log_probs, minib_value_target, minib_actions