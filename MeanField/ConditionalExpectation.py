import copy
import os
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import jraph
from tqdm import tqdm
from functools import partial
import wandb
import time

from Networks.policy import Policy
from Data.LoadGraphDataset import SolutionDatasetLoader
from jraph_utils import pad_graph_to_nearest_power_of_k, add_random_node_features

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


"""
Conditional Expectation

1. Sort spins by probability and set the most probable one
2. Calculate expected energy by sampling N states where the set spin is fixed (for both -1 and 1) -> use better for that spin
3. Set the spin with the next highest probability
4. Calculate expected energy by sampling N states where the two set spins are fixed (for both -1 and 1 for the second spin) -> use better for that spin
5. ...
"""


class ConditionalExpectation:
    def __init__(self, wandb_id, n_different_random_node_features):
        self.wandb_id = wandb_id
        self.wandb_id_CE = wandb.util.generate_id()

        path = os.getcwd()
        self.path_results = path + "/CE_results"

        self.path_to_models = path + "/Checkpoints"

        self.n_different_random_node_features = n_different_random_node_features

        WANDB = False
        if WANDB:
            self.wandb_mode = "online"
        else:
            self.wandb_mode = "disabled"


        self.__load_network()
        #self.__init_dataset()
        self.__vmap_get_energy = jax.vmap(self.__get_energy, in_axes=(None, 0), out_axes=(1))

    def __init_wandb(self):
        wandb.init(project=self.wandb_project, name=self.wandb_run, id=self.wandb_id_CE, mode=self.wandb_mode)

    def __load_params(self):
        file_name = f"best_{self.wandb_id}.pickle"

        with open(f'{self.path_to_models}/{self.wandb_id}/{file_name}', 'rb') as f:
            params, config, eval_dict = pickle.load(f)
        return params, config, eval_dict

    def __load_network(self):
        self.params, self.config, self.eval_dict = self.__load_params()

        print(f"wandb ID: {self.wandb_id}\nDataset: {self.config['dataset_name']} | Problem: {self.config['problem_name']}")
        self.path_dataset = self.config['dataset_name']

        n_features_list_prob = self.config["n_features_list_prob"]
        n_features_list_nodes = self.config["n_features_list_nodes"]
        n_features_list_edges = self.config["n_features_list_edges"]
        n_features_list_messages = self.config["n_features_list_messages"]
        n_features_list_encode = self.config["n_features_list_encode"]
        n_features_list_decode = self.config["n_features_list_decode"]
        n_message_passes = self.config["n_message_passes"]
        message_passing_weight_tied = self.config["message_passing_weight_tied"]
        linear_message_passing = self.config["linear_message_passing"]

        self.dataset_name = self.config["dataset_name"]
        self.problem_name = self.config["problem_name"]

        self.batch_size = 32
        self.N_basis_states = 100

        self.T_max = self.config["T_max"]

        self.seed = self.config["seed"]

        if "relaxed" in self.config.keys():
            self.relaxed = self.config['relaxed']
            print(f"Relaxed: {self.relaxed}")

            if self.relaxed:
                if self.problem_name == 'MVC':
                    self.__relaxed_energy = self.MVC_Energy
                    self.__vmap_relaxed_energy = jax.vmap(self.MVC_Energy, in_axes=(None, 0), out_axes=(1))
                elif self.problem_name == 'MIS':
                    self.__relaxed_energy = self.MIS_Energy
                    self.__vmap_relaxed_energy = jax.vmap(self.MIS_Energy, in_axes=(None, 0), out_axes=(1))
                else:
                    raise ValueError(
                        f'For {self.problem_name} exists no energy function that can be used with relaxed states!')
        else:
            self.relaxed = False

        #self.wandb_project = f"{self.config['dataset_name']}-{self.config['problem_name']}_ConditionalExpectation"
        self.wandb_project = f"_{self.config['problem_name']}_ConditionalExpectation_RERUNS"
        # self.wandb_run = f"{self.config['dataset_name']}_{self.seed}_Tmax{self.T_max}_{self.n_different_random_node_features}_originalrun_{self.wandb_id}_currentrun_{self.wandb_id_CE}"
        self.wandb_run = f"{self.config['dataset_name']}_{self.seed}_Tmax{self.T_max}_originalrun_{self.wandb_id}"

        self.random_node_features = self.config["random_node_features"]

        self.key = jax.random.PRNGKey(self.seed)
        self.n_random_node_features = self.config["n_random_node_features"] if "n_random_node_features" in self.config.keys() else 1

        self.model = Policy(n_features_list_prob=n_features_list_prob,
                            n_features_list_nodes=n_features_list_nodes,
                            n_features_list_edges=n_features_list_edges,
                            n_features_list_messages=n_features_list_messages,
                            n_features_list_encode=n_features_list_encode,
                            n_features_list_decode=n_features_list_decode,
                            n_message_passes=n_message_passes,
                            message_passing_weight_tied=message_passing_weight_tied,
                            linear_message_passing=linear_message_passing)

        self.__init_wandb()

    def init_dataset(self, dataset_name):
        if not isinstance(dataset_name, type(None)):
            self.dataset_name = dataset_name
        data_generator = SolutionDatasetLoader(dataset=self.dataset_name, problem=self.problem_name,
                                               batch_size=self.batch_size, relaxed=self.relaxed, seed=self.seed)
        self.dataloader_train, self.dataloader_test, self.dataloader_val, (
            self.mean_energy, self.std_energy) = data_generator.dataloaders()

        print(f"T_max: {self.T_max}, {self.dataset_name} - {self.problem_name}")

    @partial(jax.jit, static_argnums=(0,))
    def __get_energy(self, jraph_graph, state):
        state = jnp.expand_dims(state, axis=-1)
        nodes = jraph_graph.nodes
        n_node = jraph_graph.n_node
        n_graph = jraph_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        sum_n_node = jraph_graph.nodes.shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

        energy_messages = jraph_graph.edges * state[jraph_graph.senders] * state[jraph_graph.receivers]
        energy_per_node = 0.5 * jax.ops.segment_sum(energy_messages, jraph_graph.receivers,
                                                    sum_n_node) + state * jnp.expand_dims(jraph_graph.nodes[:, 0],
                                                                                          axis=-1)
        energy = jax.ops.segment_sum(energy_per_node, node_graph_idx, n_graph)
        return energy

    @partial(jax.jit, static_argnums=(0,))
    def MVC_Energy(self, H_graph, bins):
        A = 1.
        B = 1.1

        nodes = H_graph.nodes
        n_node = H_graph.n_node
        n_graph = n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        sum_n_node = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(
            graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        adjacency = jnp.ones_like(H_graph.edges)

        ### normalise through average number of nodes in dataset
        A = A
        B = B

        raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
        Energy_messages = adjacency * (1 - raveled_bins[H_graph.senders]) * (1 - raveled_bins[H_graph.receivers])
        Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

        Hb = B * (jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph))
        Ha = A * (jax.ops.segment_sum(raveled_bins, node_gr_idx, n_graph))

        Energy = Ha + Hb

        return Energy

    @partial(jax.jit, static_argnums=(0,))
    def MIS_Energy(self, H_graph, bins):
        A = 1.
        B = 1.1

        nodes = H_graph.nodes
        n_node = H_graph.n_node
        n_graph = n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        sum_n_node = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(
            graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        adjacency = jnp.ones_like(H_graph.edges)

        ### normalise through average number of nodes in dataset
        A = A
        B = B

        raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
        Energy_messages = adjacency * (raveled_bins[H_graph.senders]) * (raveled_bins[H_graph.receivers])
        Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

        Hb = B * (jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph))
        Ha = A * (jax.ops.segment_sum(raveled_bins, node_gr_idx, n_graph))

        Energy = - Ha + Hb

        return Energy

    def __conditional_expectation_relaxed(self, jraph_graph, probs):
        jax.config.update('jax_platform_name', 'cpu')

        # shape = probs.shape
        # probs = np.full(shape, 0.5)
        probs_idx = probs.copy()

        for s_i in range(len(probs)):
            highest_idx = np.argmax(probs_idx)
            probs_idx[highest_idx] = float('-inf')

            probs[highest_idx] = 1
            probs_up = probs.copy()
            probs[highest_idx] = 0
            probs_down = probs.copy()

            energy_up = self.__relaxed_energy(jraph_graph, probs_up).flatten()#[0]
            energy_down = self.__relaxed_energy(jraph_graph, probs_down).flatten()#[0]

            if energy_down > energy_up:
                probs = probs_up
            else:
                probs = probs_down

        jax.config.update('jax_platform_name', 'gpu')

        state = probs * 2 - 1
        return self.__relaxed_energy(jraph_graph, probs).flatten()[0], state

    def __conditional_expectation(self, jraph_graph, jnp_states, jnp_spin_log_probs):
        states = np.array(jnp_states)
        spin_log_probs = np.array(jnp_spin_log_probs)

        idxs = []
        jax.config.update('jax_platform_name', 'cpu')

        for s_i in range(len(spin_log_probs)):
            highest_idx = np.argmax(spin_log_probs)
            idxs.append(highest_idx)
            spin_log_probs[highest_idx] = float('-inf')

            states[:, highest_idx] = 1
            states_up = states.copy()
            states[:, highest_idx] = -1
            states_down = states.copy()

            energy_up = np.mean(self.__vmap_get_energy(jraph_graph, states_up))
            energy_down = np.mean(self.__vmap_get_energy(jraph_graph, states_down))

            if energy_down > energy_up:
                states = states_up
            else:
                states = states_down
        jax.config.update('jax_platform_name', 'gpu')

        return np.mean(self.__vmap_get_energy(jraph_graph, states).flatten()), states[0]


    def __unbatch(self, graph_batch, gt_normed_energies_batch, states_batch, spin_log_probs_batch, spin_logits_batch):
        nodes = graph_batch.nodes
        n_node = graph_batch.n_node
        n_graph = graph_batch.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        sum_n_node = graph_batch.nodes.shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

        states_batch = np.array(states_batch)
        to_unbatch_graph_batch = copy.deepcopy(graph_batch)
        graphs_list = jraph.unbatch_np(to_unbatch_graph_batch)
        graphs_list = graphs_list[:-1]
        states_list = []
        spin_log_probs_list = []
        gt_normed_energies_list = []
        spin_logits_list = []

        for idx, graph in enumerate(graphs_list):
            gt_normed_energies_list.append(gt_normed_energies_batch[idx])
            states_list.append(states_batch[:, node_graph_idx == idx])
            spin_log_probs_list.append(spin_log_probs_batch[:, node_graph_idx == idx])
            spin_logits_list.append(spin_logits_batch[node_graph_idx == idx])

        return graphs_list, states_list, spin_log_probs_list, gt_normed_energies_list, spin_logits_list

    def __generate_random_node_feature_batch(self, graph_batch, iter):
        graph_list = []
        for i in range(self.n_different_random_node_features):
            _graph_batch = add_random_node_features(graph_batch, n_random_node_features=self.n_random_node_features,
                                                    seed=self.seed + iter + i)
            graph_list.append(_graph_batch)
        return jraph.batch(graph_list)

    @partial(jax.jit, static_argnums=(0, 3))
    def forward(self, params, graph_batch, n_basis_states, key):
        return self.model.apply(params, graph_batch, n_basis_states, key)

    def run_time(self, p=None):

        if self.batch_size > 1:
            raise ValueError("The batch size should be one when timing the code")

        DATALOADER = self.dataloader_test
        dataset_len = len(DATALOADER.dataset)

        dataset_len_active = 0

        print(f"\nLength of dataset: {dataset_len}\n")

        time_inferce_list = []
        time_unbatch_list = []
        time_CE_list = []

        for iter, (graph_batch, gt_normed_energies_batch, gt_spin_states) in tqdm(enumerate(DATALOADER), total=len(DATALOADER)):
            dataset_len_active += len(gt_normed_energies_batch)
            # one batch

            _graph_batch = self.__generate_random_node_feature_batch(graph_batch, iter)
            _graph_batch = pad_graph_to_nearest_power_of_k(_graph_batch)

            t_inference_0 = time.time()
            states_batch, _, spin_log_probs_batch, spin_logits_batch, _, self.key = self.forward(self.params, _graph_batch,
                                                                                               self.N_basis_states,
                                                                                               self.key)
            gt_normed_energies_batch = np.repeat(gt_normed_energies_batch, self.n_different_random_node_features)
            t_inference_1 = time.time()

            t_inference = t_inference_1-t_inference_0
            time_inferce_list.append(t_inference)

            t_unbatch_0 = time.time()
            graphs_list, states_list, spin_log_probs_list, gt_normed_energies_list, spin_logits_list = self.__unbatch(_graph_batch, gt_normed_energies_batch, states_batch, spin_log_probs_batch, spin_logits_batch)
            t_unbatch_1 = time.time()
            t_unbatch = t_unbatch_1-t_unbatch_0
            time_unbatch_list.append(t_unbatch)

            time_CE_sub_list = []
            for idx, (graph, states, spin_log_probs, gt_normed_energies, spin_logits) in enumerate(zip(graphs_list, states_list, spin_log_probs_list, gt_normed_energies_list, spin_logits_list)):
                # one graph
                if self.relaxed:
                    probs = np.exp(spin_logits[:, 1])
                    t_CE_0 = time.time()
                    energy, state = self.__conditional_expectation_relaxed(graph, probs)
                    t_CE_1 = time.time()
                else:
                    t_CE_0 = time.time()
                    normed_energy, state = self.__conditional_expectation(graph, states, spin_logits[:, 1])
                    t_CE_1 = time.time()
                    energy = np.array(normed_energy * self.std_energy + self.mean_energy)

                time_CE_sub = t_CE_1-t_CE_0
                time_CE_sub_list.append(time_CE_sub)

                #print(time_CE_sub)
                #break
            #break
            time_CE_list.append(np.mean(time_CE_sub_list))

        print(f"{self.problem_name} {self.dataset_name}")
        print(f"Mean Inference Time: {np.mean(time_inferce_list)}")
        print(f"Mean Unbatch Time: {np.mean(time_unbatch_list)}")
        print(f"Mean CE Time: {np.mean(time_CE_list)}")


    def run(self, p=None):
        rel_errors = []
        rel_errors_og = []
        rel_errors_og_mean = []
        mean_rel_errors_og_mean = []
        APRs = []

        rel_error_matrix = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')
        rel_error_OG_matrix = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')
        APR_matrix = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')
        APR_OG_matrix = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')

        rel_errors_graph = np.array([])
        rel_error_og_graph = np.array([])
        APR_graph = np.array([])
        n_nodes_graph = np.array([])
        n_edges_graph = np.array([])

        DATALOADER = self.dataloader_test


        dataset_len = len(DATALOADER.dataset)

        dataset_len_active = 0

        print(f"\nLength of dataset: {dataset_len}\n")

        state_probs = np.zeros(shape=(dataset_len))
        mean_probs = np.zeros(shape=(dataset_len))
        best_rel_errors = np.ones(shape=(dataset_len)) * float('inf')

        for iter, (graph_batch, gt_normed_energies_batch, gt_spin_states) in tqdm(enumerate(DATALOADER), total=len(DATALOADER)):
            dataset_len_active += len(gt_normed_energies_batch)
            # one batch
            graph_batch = pad_graph_to_nearest_power_of_k(graph_batch)

            batch_rel_error = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')
            batch_rel_error_og = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')
            batch_rel_error_og_mean = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')
            batch_APR = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')
            batch_APR_mean = np.ones((self.n_different_random_node_features, self.batch_size)) * float('-inf')

            for i in range(self.n_different_random_node_features):
                # one batch with one set of random node features

                _graph_batch = add_random_node_features(graph_batch, n_random_node_features=self.n_random_node_features, seed=(self.seed, iter, i))
                states_batch, _, spin_log_probs_batch, spin_logits_batch, _, self.key = self.model.apply(self.params, _graph_batch,
                                                                               self.N_basis_states,
                                                                               self.key)

                graphs_list, states_list, spin_log_probs_list, gt_normed_energies_list, spin_logits_list = self.__unbatch(_graph_batch, gt_normed_energies_batch, states_batch, spin_log_probs_batch, spin_logits_batch)
                for idx, (graph, states, spin_log_probs, gt_normed_energies, spin_logits) in enumerate(zip(graphs_list, states_list, spin_log_probs_list, gt_normed_energies_list, spin_logits_list)):
                    # one graph
                    if self.relaxed:
                        probs = np.exp(spin_logits[:, 1])
                        energy, state = self.__conditional_expectation_relaxed(graph, probs)
                    else:
                        normed_energy, state = self.__conditional_expectation(graph, states, spin_logits[:, 1])
                        energy = np.array(normed_energy * self.std_energy + self.mean_energy)

                    if not self.problem_name == 'WMIS':
                        energy = np.round(energy, decimals=0)

                    gt_energy = np.array(gt_normed_energies * self.std_energy + self.mean_energy)

                    if not self.problem_name == 'WMIS':
                        gt_energy = np.round(gt_energy, decimals=0)

                    rel_error = abs((energy - gt_energy) / gt_energy)

                    bin_state = (state + 1)/2

                    one_hot_state = jax.nn.one_hot(state, num_classes=2)
                    CE_state_spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)


                    state_prob = np.exp(np.sum(CE_state_spin_log_probs))
                    mean_prob = np.mean(np.exp(CE_state_spin_log_probs))

                    if rel_error < best_rel_errors[iter*self.batch_size+idx]:
                        state_probs[iter*self.batch_size+idx] = state_prob
                        mean_probs[iter*self.batch_size+idx] = mean_prob
                        best_rel_errors[iter * self.batch_size + idx] = rel_error


                    if self.config["problem_name"] == "MVC":
                        if not np.sum(bin_state) == energy:
                            raise ValueError(f"np.sum(bin_state) {np.sum(bin_state)} should equal the energy {energy}!")
                    if self.config["problem_name"] == "MIS" or self.config["problem_name"] == "MaxCl":
                        if not np.sum(bin_state) == -energy:
                            raise ValueError(f"np.sum(bin_state) {np.sum(bin_state)} should equal the -energy {-energy}!")

                    if energy < gt_energy:
                        raise ValueError(
                            f"\nYeah, that's not supposed to be that way...\nenergy: {energy}; gt_energy: {gt_energy}\n Graph: {idx}\n")

                    batch_rel_error[i][idx] = rel_error

                    if self.relaxed:
                        states_bin = (states + 1) / 2
                        normed_energy_og = self.__vmap_relaxed_energy(graph, states_bin)
                    else:
                        normed_energy_og = self.__vmap_get_energy(graph, states)
                    _energy_og = np.array(normed_energy_og * self.std_energy + self.mean_energy).flatten()
                    energy_og = np.min(_energy_og)
                    energy_og_mean = np.mean(_energy_og)

                    if not self.problem_name == 'WMIS':
                        energy_og = np.round(energy_og, decimals=0)
                        energy_og_mean = np.round(energy_og_mean, decimals=0)

                    rel_error_og = abs((energy_og - gt_energy) / gt_energy)
                    rel_error_og_mean = abs((energy_og_mean - gt_energy) / gt_energy)
                    batch_rel_error_og[i][idx] = rel_error_og
                    batch_rel_error_og_mean[i][idx] = rel_error_og_mean

                    if self.config["problem_name"] == "MVC":
                        APR = energy / gt_energy
                        APR_mean = energy_og_mean / gt_energy
                        batch_APR[i][idx] = APR
                        batch_APR_mean[i][idx] = APR_mean


            n_nodes_graph = np.concatenate((n_nodes_graph, graph_batch.n_node[:-1]))
            n_edges_graph = np.concatenate((n_edges_graph, graph_batch.n_edge[:-1]))



            rel_error_matrix = np.concatenate((rel_error_matrix, batch_rel_error), axis=1)
            rel_error_OG_matrix = np.concatenate((rel_error_OG_matrix, batch_rel_error_og_mean), axis=1)

            if self.config["problem_name"] == "MVC":
                APR_matrix = np.concatenate((APR_matrix, batch_APR), axis=1)
                APR_OG_matrix = np.concatenate((APR_OG_matrix, batch_APR_mean), axis=1)


        rel_error_matrix = rel_error_matrix[:, self.batch_size:]
        rel_error_matrix = rel_error_matrix[:, :dataset_len]

        rel_error_OG_matrix = rel_error_OG_matrix[:, self.batch_size:]
        rel_error_OG_matrix = rel_error_OG_matrix[:, :dataset_len]

        APR_matrix = APR_matrix[:, self.batch_size:]
        APR_matrix = APR_matrix[:, :dataset_len]

        APR_OG_matrix = APR_OG_matrix[:, self.batch_size:]
        APR_OG_matrix = APR_OG_matrix[:, :dataset_len]


        rel_error_CE = np.mean(np.min(rel_error_matrix, axis=0))
        rel_error_OG_mean = np.mean(rel_error_OG_matrix[0])


        if self.config["problem_name"] == "MVC":
            APR_CE = np.mean(np.min(APR_matrix, axis=0))
            APR_OG_mean = np.mean(APR_OG_matrix[0], axis=0)

            print(f"Shapes: {rel_error_matrix.shape}; {rel_error_OG_matrix.shape}; {APR_matrix.shape}; {APR_OG_matrix.shape}")
            print(f"\n### rel_error_CE: {rel_error_CE}; rel_error_OG_mean: {rel_error_OG_mean}; APR_CE: {APR_CE}; APR_OG_mean: {APR_OG_mean}")
        else:
            print(f"Shapes: {rel_error_matrix.shape}; {rel_error_OG_matrix.shape}")
            print(f"\n### rel_error_CE: {rel_error_CE}; rel_error_OG_mean: {rel_error_OG_mean}")



        if not rel_error_matrix.shape[1] == dataset_len:
            raise ValueError(f'The result matrix shape is wrong. {rel_error_matrix.shape} and the length of the dataset is {dataset_len}')
        if np.isinf(rel_error_matrix).any():
            raise(ValueError(f'The result matrix contains inf'))


        #raise NotImplementedError()

        results = {
            'rel_error_matrix': rel_error_matrix,
            'rel_error_OG_matrix': rel_error_OG_matrix,
            'APR_CE_matrix': APR_matrix,
            'APR_OG_matrix': APR_OG_matrix,
            'n_nodes': n_nodes_graph,
            'n_edges': n_edges_graph,
            'state_probs': state_probs,
            'mean_probs': mean_probs,
        }

        table_state_probs = wandb.Table(data=np.expand_dims(state_probs, axis=-1), columns=["probs"])
        table_mean_probs = wandb.Table(data=np.expand_dims(mean_probs, axis=-1), columns=["mean_probs"])
        if self.config["problem_name"] == "MVC":
            wandb.log({"dataset/APR_CE": APR_CE,
                       "dataset/APR_val": self.eval_dict['eval/mean_APR'],
                       "dataset/rel_error_CE": rel_error_CE,
                       "dataset/rel_error_OG_mean": rel_error_OG_mean,
                       "dataset/rel_error_val": self.eval_dict['eval/rel_error'],
                       "dataset/state_probs_hist": wandb.plot.histogram(table_state_probs, 'probs'),
                       "dataset/mean_probs_hist": wandb.plot.histogram(table_mean_probs, 'mean_probs')})

        else:
            wandb.log({"dataset/rel_error_CE": rel_error_CE,
                       "dataset/rel_error_OG_mean": rel_error_OG_mean,
                       "dataset/rel_error_val": self.eval_dict['eval/rel_error'],
                       "dataset/state_probs_hist": wandb.plot.histogram(table_state_probs, 'probs'),
                       "dataset/mean_probs_hist": wandb.plot.histogram(table_mean_probs, 'mean_probs')})

        if not isinstance(p, type(None)):
            self.__save_result(results, p)
        else:
            self.__save_result(results, None)

        if self.config["problem_name"] == "MVC":
            return results, rel_error_CE, rel_error_OG_mean, APR_CE, APR_OG_mean

        return results, rel_error_CE, rel_error_OG_mean, None, None

    def __save_result(self, results, p):
        result_dict = {
            "wandb_run_id": self.wandb_id,
            "dataset_name": self.dataset_name,
            "problem_name": self.problem_name,
            "T": self.T_max,
            "n_different_random_node_features": self.n_different_random_node_features,
            "p": p,
            "results": results
            }

        path_folder = f"{self.path_results}/{self.path_dataset}/"
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        print(f'\nsaving to {os.path.join(path_folder, f"{self.n_different_random_node_features}_{self.wandb_id}_{self.dataset_name}_{self.seed}_.pickle")}\n')

        with open(os.path.join(path_folder, f"{self.n_different_random_node_features}_{self.wandb_id}_{self.dataset_name}_{self.seed}_.pickle"), 'wb') as f:
            pickle.dump(result_dict, f)


if __name__ == "__main__":
    device = 2

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    print(f"GPU: {device}")

    wandb_id = "3iut7t6v" #"k70oqu9p" #"z4jmevmj"#"q916v52s"
    CE = ConditionalExpectation(wandb_id=wandb_id, n_different_random_node_features=8)
    CE.init_dataset(dataset_name=None)
    CE.run(p=None)
    print('\n###\n')
    #CE.run_time(p=None)



    # ps = np.linspace(0.25, 1, 10)
    #
    # APR_dict = {}
    #
    # rel_errors_CE = []
    # APRs_CE = []
    # for p in ps:
    #     print(f"\np: {p}")
    #     rel_error_CEs = []
    #     APR_CEs = []
    #     CE = ConditionalExpectation(wandb_id=wandb_id, n_different_random_node_features=8)
    #     dataset_name = f"{CE.dataset_name}_p_{p}"
    #     CE.init_dataset(dataset_name=dataset_name)
    #
    #     results, rel_error_CE, rel_error_OG_mean, APR_CE, APR_OG_mean = CE.run(p=p)
    #     rel_errors_CE.append(rel_error_CE)
    #     APRs_CE.append(APR_CE)
    #
    # print("\n")
    # print(f"rel_error: {np.mean(rel_errors_CE)}")
    # print(f"APR: {np.mean(APRs_CE)}")

    #     seeds = np.arange(0, 12)
    #     for s in seeds:
    #         # print(f"\n\n ## now running seed {s}:\n")
    #         CE.seed = s
    #         results, rel_error_CE, rel_error_OG_mean, APR_CE, APR_OG_mean = CE.run(p=p)
    #         rel_error_CEs.append(rel_error_CE)
    #         APR_CEs.append(APR_CE)
    #         print(f"{np.round(np.mean(APR_CEs), 4)} \\pm {np.std(APR_CEs) / np.sqrt(len(APR_CEs))}")
    #     APR_dict[f'{p}'] = APR_CEs
    #
    #     print('\n\n############')
    #     print(f"mean rel_error* : {np.mean(rel_error_CEs)};   std rel_error* : {np.std(rel_error_CEs)/np.sqrt(len(rel_error_CEs))}")
    #     print(f"mean APR* : {np.mean(APR_CEs)};   std rel_error* : {np.std(APR_CEs)/np.sqrt(len(APR_CEs))}")
    #     print('############\n\n')
    #
    #
    # for p in ps:
    #     APR_CEs = APR_dict[f"{p}"]
    #     print(f"p: {p}; \tmean APR* : {np.mean(APR_CEs)}; \tstd rel_error* : {np.std(APR_CEs) / np.sqrt(len(APR_CEs))}")
    #     print(f"{np.round(np.mean(APR_CEs), 4)} \\pm {np.std(APR_CEs) / np.sqrt(len(APR_CEs))}")







