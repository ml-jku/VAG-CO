import os
import pandas as pd
from torch.utils.data import Dataset
import pickle
import numpy as np
from jraph_utils import utils as jutils
import igraph
from loadGraphDatasets import GenerateRRGs
from utils import string_utils
import jax


class JraphSolutionDataset_fromMemory(Dataset):
    def __init__(self, cfg, mode = "val", seed = None, ordering = "BFS", indices = None): ### TODO add orderign to config
        self.ordering = ordering
        self.self_loops = cfg["Ising_params"]["self_loops"]

        if(not self.self_loops):
            self.H_graph_type = "no_norm_H_graph_sparse"
        else:
            self.H_graph_type = "normed_H_graph_sparse"

        self.normed_Energy_list, self.orig_igraph_list, self.normed_graph_list,self.ground_states = self.get_dataset(cfg, mode = mode, seed = seed)

        self.EnergyFunction = cfg["Ising_params"]["EnergyFunction"]
        self.B = 1.1
        self.seed = seed
        self.cfg = cfg

        self.random_node_features = cfg["Ising_params"]["n_rand_nodes"]
        self.ordering = cfg["Ising_params"]["ordering"]

    def get_dataset(self, cfg, mode = "", seed = None):
        EnergyFunction = cfg["Ising_params"]["EnergyFunction"]
        self.dataset_name = cfg["Ising_params"]["IsingMode"]
        path = os.path.join(cfg["Paths"]["work_path"],"loadGraphDatasets", "DatasetSolutions", self.H_graph_type, self.dataset_name, f"{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle")

        with open(path, "rb") as file:
            solution_dict = pickle.load(file)

        self.mean_energy = solution_dict["val_mean_Energy"]
        self.std_energy = solution_dict["val_std_Energy"]
        self.original_Energy_list = solution_dict["original_Energies"]
        self.self_loop_Energy_list = solution_dict["self_loop_Energies"]

        if("orig_igraph" not in solution_dict.keys()):
            orig_igraph_list = None
        else:
            orig_igraph_list = solution_dict["orig_igraph"]

        return solution_dict["normed_Energies"], orig_igraph_list, solution_dict["normed_igraph"], solution_dict["gs_bins"]


    def normalize_Energy(self, Energy_arr):
        return (Energy_arr - self.mean_energy) / (self.std_energy)

    def __len__(self):
        return len(self.normed_Energy_list)

    def __getitem__(self, idx):
        igraph = self.normed_graph_list[idx]
        igraph["gt_Energy"] = np.array([self.normed_Energy_list[idx]])
        igraph["original_Energy"] = np.array([self.original_Energy_list[idx]])
        igraph["self_loop_Energy"] = np.array([self.self_loop_Energy_list[idx]])


        if(self.orig_igraph_list == None):
            orig_igraph = None
        else:
            orig_igraph = self.orig_igraph_list[idx]

        return igraph, orig_igraph, self.ground_states[idx]

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
class JraphSolutionDataset(InMemoryDataset):
    def __init__(self, cfg, mode="val", seed=None, ordering="BFS", indices = None):  ### TODO add orderign to config
        self.ordering = ordering
        self.self_loops = cfg["Ising_params"]["self_loops"]

        if (not self.self_loops):
            self.H_graph_type = "no_norm_H_graph_sparse"
        else:
            self.H_graph_type = "normed_H_graph_sparse"

        self.get_dataset_paths(cfg, mode=mode, seed=seed)
        super().__init__(self.base_path, None, None, None)
        self.EnergyFunction = cfg["Ising_params"]["EnergyFunction"]
        self.seed = seed
        self.cfg = cfg

        self.random_node_features = cfg["Ising_params"]["n_rand_nodes"]
        self.ordering = cfg["Ising_params"]["ordering"]

        if(type(indices) is not np.ndarray):
            self.indices = np.arange(0, self.n_graphs)
        else:
            self.n_graphs = indices.shape[0]
            self.indices = indices

    def get_dataset_paths(self, cfg, mode="", seed=None):
        EnergyFunction = cfg["Ising_params"]["EnergyFunction"]
        self.dataset_name = cfg["Ising_params"]["IsingMode"]

        path = os.path.join(cfg["Paths"]["work_path"], "loadGraphDatasets", "DatasetSolutions", self.H_graph_type,
                            self.dataset_name, f"{mode}_{EnergyFunction}_seed_{seed}_solutions", f"_idx_{0}.pickle")
        self.base_path = os.path.join(cfg["Paths"]["work_path"], "loadGraphDatasets", "DatasetSolutions", self.H_graph_type,
                            self.dataset_name, f"{mode}_{EnergyFunction}_seed_{seed}_solutions")

        with open(path, "rb") as file:
            data_dummy_dict = pickle.load(file)

        self.mean_energy = data_dummy_dict["val_mean_Energy"]
        self.std_energy = data_dummy_dict["val_std_Energy"]
        self.n_graphs = data_dummy_dict["n_graphs"]

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, idx):

        with open(self.base_path + f"/_idx_{self.indices[idx]}.pickle", "rb") as file:
            graph_dict = pickle.load(file)

        if("orig_igraph" not in graph_dict.keys()):
            orig_igraph = None
        else:
            orig_igraph = graph_dict["orig_igraph"]

        igraph = graph_dict["normed_igraph"]
        igraph["gt_Energy"] = np.array([graph_dict["normed_Energy"]])
        igraph["original_Energy"] = np.array([graph_dict["original_Energy"]])
        igraph["self_loop_Energy"] = np.array([graph_dict["self_loop_Energy"]])
        return igraph, orig_igraph, graph_dict["gs_bins"]


class RRGDataset(Dataset):
    def __init__(self, cfg, ks = [3,7,10,20 ]):

        self.random_node_features = cfg["Ising_params"]["n_rand_nodes"]
        self.ordering = cfg["Ising_params"]["ordering"]

        self.num_graphs = 100
        self.num_nodes = 100
        self.min_nodes = 75
        self.max_nodes = 100
        self.num_nodes = 100#(self.min_nodes + self.max_nodes)/2
        self.graph_sizes = np.arange(80,self.max_nodes + 1, 2)
        self.ks = ks
        self.ks = [20]

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        k = np.random.choice(self.ks)
        i_graph = GenerateRRGs.generate_regular_graph(self.num_nodes, k)
        j_graph = jutils.igraph_to_jraph(i_graph, np_= np)

        # if(self.random_node_features > 0):
        #     random_node_features = np.random.normal(size = (j_graph.nodes.shape[0], self.random_node_features))
        #     new_nodes = np.concatenate([j_graph.nodes, random_node_features], axis = -1)
        #     j_graph = j_graph._replace(nodes=new_nodes)

        j_graph = j_graph._replace(edges = j_graph.edges/self.num_nodes)

        if(self.ordering == "BFS"):
            order = GenerateRRGs.do_bfs(i_graph)
            j_graph = jutils.order_jgraph(j_graph, order)
        elif(self.ordering == "DFS"):
            order = GenerateRRGs.do_dfs(i_graph)
            j_graph = jutils.order_jgraph(j_graph, order)
        else:
            senders, receivers = jutils.shuffle_senders_and_receivers(j_graph.nodes.shape[0], j_graph.senders,
                                                                      j_graph.receivers)
            j_graph = j_graph._replace(senders=senders, receivers=receivers)


        return j_graph


