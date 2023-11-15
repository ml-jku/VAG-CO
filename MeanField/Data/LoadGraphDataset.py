import os
import random
from torch.utils.data import Dataset
import pickle
import numpy as np
import igraph
import jraph
import torch
from torch.utils.data import DataLoader
from unipath import Path

class SolutionDatasetLoader:
    def __init__(self, config = {}, dataset="MIS", problem="MIS", batch_size=32, relaxed=False, seed=123, mode = "train"):
        self.dataset_name = dataset
        self.problem_name = problem
        self.batch_size = batch_size
        self.relaxed = relaxed
        self.seed = seed
        self.num_workers = 8
        self.config = config
        self.mode = mode

        torch.manual_seed(self.seed)
        self._init_mode()

    def _init_mode(self):
        if(self.mode == "train"):
            self.train_dataset = True
            self.val_dataset = True
            self.test_dataset = False
        else:
            self.train_dataset = False
            self.val_dataset = False
            self.test_dataset = True

    def dataloaders(self):
        def collate_function(batch):
            batch_transposed = list(zip(*batch))
            jraph_graphs = batch_transposed[0]
            gt_normed_energies = batch_transposed[1]
            gt_spin_states = batch_transposed[2]
            batched_jraph_graph = jraph.batch_np(jraph_graphs)
            return batched_jraph_graph, np.asarray(gt_normed_energies), np.concatenate(gt_spin_states).astype(int)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        TRAIN_DATASET = self.train_dataset
        TEST_DATASET = self.test_dataset
        VAL_DATASET = self.val_dataset

        dataset_train = SolutionDataset(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="train", relaxed=self.relaxed, seed=self.seed) if TRAIN_DATASET else None
        dataset_test = SolutionDataset(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="test", relaxed=self.relaxed, seed=self.seed) if TEST_DATASET else None
        dataset_val = SolutionDataset(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="val", relaxed=self.relaxed, seed=self.seed) if VAL_DATASET else None

        if(self.mode == "train"):
            mean_energy = dataset_train.val_mean_energy
            std_energy = dataset_train.val_std_energy
        else:
            mean_energy = dataset_test.val_mean_energy
            std_energy = dataset_test.val_std_energy

        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, collate_fn=collate_function, num_workers=self.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=generator) if TRAIN_DATASET else None
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, collate_fn=collate_function, num_workers=self.num_workers, worker_init_fn=seed_worker, generator=generator) if dataset_test != None else None
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, collate_fn=collate_function, num_workers=self.num_workers, worker_init_fn=seed_worker, generator=generator) if VAL_DATASET else None
        return dataloader_train, dataloader_test, dataloader_val, (mean_energy, std_energy)


class SolutionDataset(Dataset):
    def __init__(self, config = {}, dataset="ENZYMES", problem="MIS", mode="val", relaxed=False, seed=123):
        self.dataset_name = dataset
        self.problem_name = problem
        self.mode = mode
        self.seed = seed
        self.relaxed = relaxed

        if self.relaxed:
            self.path = os.getcwd() + "/loadGraphDatasets/DatasetSolutions/no_norm/"

            self.jraph_list, (self.normed_energies, self.gs_bin_states, self.val_mean_energy,
                              self.val_std_energy) = self.__load_dataset()

        else:
            self.path = os.getcwd() + "/loadGraphDatasets/DatasetSolutions/normed_H_graph_sparse/"

            self.jraph_list, (self.normed_energies, self.gs_bin_states, self.val_mean_energy,
                              self.val_std_energy) = self.__create_jraph_dataset()

    def __len__(self):
        return len(self.normed_energies)

    def __getitem__(self, item):
        jraph_graph = self.jraph_list[item]
        gt_normed_energy = np.array(self.normed_energies[item])
        gt_spin_state = np.array(self.gs_bin_states[item]) * 2 - 1
        return jraph_graph, gt_normed_energy, gt_spin_state

    def __load_dataset(self):
        base_path = os.path.join(self.path, self.dataset_name)
        path = os.path.join(base_path, f"{self.mode}_{self.problem_name}_seed_{self.seed}_solutions.pickle")
        with open(path, 'rb') as file:
            solution_dict = pickle.load(file)

        if self.relaxed:
            mean_energy = 0
            std_energy = 1
            return solution_dict["H_graphs"], (solution_dict["Energies"], solution_dict["gs_bins"], mean_energy, std_energy)
        else:
            mean_energy = solution_dict["val_mean_Energy"]
            std_energy = solution_dict["val_std_Energy"]
            return solution_dict["normed_igraph"], (solution_dict["normed_Energies"], solution_dict["gs_bins"], mean_energy, std_energy)

    def __create_jraph_dataset(self):
        if self.relaxed:
            raise ValueError('__create_jraph_dataset should not be called when using relaxed states as the dataset used is not normed!')
        i_graphs, (normed_energies, gs_bin_states, val_mean_energy, val_std_energy) = self.__load_dataset()
        jraph_list = []
        for i_graph in i_graphs:
            jraph_list.append(self.__return_jraph(i_graph))
        return jraph_list, (normed_energies, gs_bin_states, val_mean_energy, val_std_energy)

    def __return_jraph(self, i_graph: igraph.Graph):
        """
        Return current igraph as jraph
        The nodes are the external fields and the edges are the couplings
        """
        edges = np.array(i_graph.get_edgelist())
        n_edges = i_graph.ecount()
        if n_edges > 0:
            couplings = np.array(i_graph.es['couplings'])

            jraph_senders = edges[:, 0]
            jraph_receivers = edges[:, 1]

            if(not self.relaxed):
                jraph_senders = np.concatenate([edges[:, 0], edges[:, 1]], axis = 0)
                jraph_receivers = np.concatenate([edges[:, 1], edges[:, 0]], axis = 0)
                couplings = np.concatenate([couplings, couplings], axis = 0)


            external_fields = np.array(i_graph.vs['ext_fields'])
            jraph_graph = jraph.GraphsTuple(nodes=external_fields,
                                            edges=couplings,
                                            senders=jraph_senders,
                                            receivers=jraph_receivers,
                                            n_node=np.array([i_graph.vcount()]),
                                            n_edge=np.array([jraph_receivers.shape[0]]),
                                            globals=None)
        else:
            raise NotImplementedError("graph has no edges")
        #print("jraph", jraph_graph)
        return jraph_graph