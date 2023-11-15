
import numpy as np
from numpy import random
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
import jraph
import os
import copy
from jraph_utils import utils as jutils
from .GraphEnv_train_configuration import IGraphEnv as RB_IGraphEnv
#import jax

#jax.config.update('jax_platform_name', "cpu")


class IGraphEnv(RB_IGraphEnv):
    ### use jax random keys?
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 4}

    # def __init__(self,cfg, H_seed, mode = "train"):
    #     super().__init__(self, cfg, H_seed, mode = mode)

    def init_beam(self, init_spins):
        prob_list = []
        self.beams = [{"spins":init_spins,"spin_config": None,"ext_fields": self.init_Nb_external_fields[0], "dEnergy": [],
                       "beam_prob":1.,"prob_list" :prob_list}]

    def determine_best_beams(self, class_log_probs):
        candidates = []
        for idx, beam in enumerate(self.beams):
            # Get the log probabilities of the next classes
            state = beam["spins"]
            Ext_fields = self.Nb_external_fields[idx]
            score = beam["beam_prob"]
            prob_list = beam["prob_list"]
            dEnergy = beam["dEnergy"]

            probs = np.exp(class_log_probs[idx])

            # Calculate the total scores for each possible class
            total_scores = probs * score
            total_scores_flat = total_scores.flatten()

            # Get the indices of the top k classes
            topk_indices = np.argsort(-total_scores_flat)[0:self.Nb]
            topk_scores = total_scores_flat[topk_indices]
            topk_probs = probs.flatten()[topk_indices]
            topk_classes = topk_indices

            # Create new beam candidates for each top k class
            for class_idx, class_score, class_probs in zip(topk_classes, topk_scores, topk_probs):
                new_state = np.copy(state)
                ### TODO map spins to configuration
                spin_configurations = self._from_class_to_spins(np.array(int(class_idx)))
                new_state[self.env_step: self.env_step + self.n_sampled_sites, 0] = spin_configurations
                new_prob_list = prob_list + [class_probs]
                new_beam = {"spins" :new_state, "spin_config":spin_configurations, "ext_fields":Ext_fields, "dEnergy": copy.deepcopy(dEnergy),
                            "beam_prob": class_score, "prob_list": new_prob_list}
                candidates.append(new_beam)

        # Select the top k candidates as the new beams
        topk_candidates = sorted(candidates, key=lambda x: x["beam_prob"], reverse=True)[:self.Nb]
        self.beams = topk_candidates
        beam_spins = np.array([beam["spins"] for beam in  self.beams])
        self.Nb_spins = beam_spins
        self.Nb_external_fields = np.array([beam["ext_fields"] for beam in self.beams])

    def reset(self):

        self.dEnergies = []
        self.env_step = 0

        Igraph, self.orig_Igraph = self.init_graph_func()
        self.num_edges = Igraph.ecount()
        self.gt_Energy = Igraph["gt_Energy"]
        self.original_Energy = Igraph["original_Energy"]
        self.self_loop_Energy = Igraph["self_loop_Energy"]

        self.Nb_spins = np.zeros((self.Nb, self.N_spin, 1), dtype = np.float32)
        self.EnergyJgraph, self.Nb_external_fields = self.from_Hb_Igraph_to_Nb_Hb_Jgraph(Igraph)

        self.Jgraph, self.Igraph = self._get_compl_graph(self.EnergyJgraph, self.EnergyIgraph)
        self.return_Jgraph = self.Jgraph
        self.init_EnergyJgraph = copy.deepcopy(self.EnergyJgraph)

        self.init_Nb_external_fields = copy.copy(self.Nb_external_fields)

        self.init_beam(np.zeros((self.N_spin, 1)))
        self.start_graph_time = time.time()
        #print("reset", end_reset-start_reset)
        return {"H_graph": {"Nb_ext_fields": self.Nb_external_fields, "jgraph": self.return_Jgraph}}

    def step(self, data):
        if(self.sampling_mode == "beam_search"):
            raise ValueError("This code is stale and has to be reimplemented")
            class_log_probs = data
            ### TODO mask out probability of padded nodes
            self.determine_best_beams(class_log_probs)
            spin_configurations = np.array([np.squeeze(beam["spin_config"], axis = 0) for beam in  self.beams])

            dEnergy = self.update_and_get_Energy(spin_configurations)

            for idx, beam in enumerate(self.beams):
                beam["dEnergy"].append(dEnergy[idx])

        class_log_probs = data[:, 0:self.n_classes]
        # print("orig probs", class_log_probs.shape)
        # print(class_log_probs)
        if (self.masking):
            mask = self._check_for_violations(class_log_probs)
        else:
            mask = np.ones_like(class_log_probs)

            ### TODO add greedy sampling if mode == test
        if(self.sampling_mode == "greedy"):
            class_log_probs = mask*class_log_probs

            sampled_bin_values = np.argmax(class_log_probs, axis=-1)
            sampled_bin_values = np.reshape(sampled_bin_values, (self.Nb, 1))

            idx_1 = np.arange(0, class_log_probs.shape[0])

            greedy_mask = np.zeros_like(class_log_probs)
            greedy_mask[idx_1[:, np.newaxis], sampled_bin_values] = 1.
            class_log_probs = greedy_mask * class_log_probs
        else:
            pass


        log_probs, sampled_class = self._sample_np(class_log_probs, mask)

        spin_configurations = self._from_class_to_spins(sampled_class)
        self.Nb_spins[:, self.env_step: self.env_step + self.n_sampled_sites, 0] = spin_configurations

        dEnergy = self.update_and_get_Energy(spin_configurations)
        self.dEnergies.append(dEnergy)

        self.env_step += self.n_sampled_sites
        self.dones = self.env_step >= self.N_spin

        if(self.dones):
            if(self.sampling_mode == "beam_search"):
                print([np.array(beam["dEnergy"]).shape for beam in self.beams])
                curr_dEnergy_arr = np.concatenate([np.array(beam["dEnergy"])[:,np.newaxis] for beam in self.beams], axis=-1)
                self.dEnergies = curr_dEnergy_arr

            #self.check_energy()

            self.finished_energy = self.compute_unormalized_Energy( np.sum(np.array(self.dEnergies), axis = 0)) + self.self_loop_Energy
            if(self.mode != "train"):
                self.old_gt_Energy = self.compute_unormalized_Energy( self.gt_Energy) + self.self_loop_Energy
            else:
                self.old_gt_Energy = self.gt_Energy + self.self_loop_Energy

            if(self.mode == "test"):
                self.end_graph_time = time.time()
                print("finished Energy before decoding",np.min(self.finished_energy), "n_node", self.N_spin, self.orig_n_nodes)
                if(self.EnergyFunction != "MaxCut"):
                    self.finished_energy, HA, HB = self.check_for_violations(self.init_EnergyJgraph, np.argmin(self.finished_energy))
                    if (HB != 0):
                        raise ValueError("violation occured HB =", HB)
                #self.finished_energy = np.expand_dims(self.finished_energy, axis = -1)
                self.finished_energy = np.round(self.compute_Energy_full_graph(self.Nb_spins) + self.self_loop_Energy, 0)
                print("finished Energy after decoding", self.finished_energy, "n_node", self.N_spin, self.orig_n_nodes)
                print("AR", np.min(self.finished_energy)/self.old_gt_Energy, np.min(self.finished_energy), self.old_gt_Energy)

                old_finished = self.finished
                old_num_edges = self.num_edges
                old_num_nodes = self.orig_n_nodes

                self.time_needed = self.end_graph_time - self.start_graph_time
                print("probs per config ", np.array([beam["beam_prob"] for beam in  self.beams]) )
                self.orig_graph_dict = {"num_edges": old_num_edges, "num_nodes": old_num_nodes, "gt_Energy": self.old_gt_Energy, "pred_Energy": self.finished_energy, "orig_Nb_ext_fields": copy.copy(self.init_Nb_external_fields),
                                        "orig_graph": copy.deepcopy(self.init_EnergyJgraph), "Nb_spins": copy.deepcopy(self.Nb_spins), "time_per_graph": self.time_needed}
            else:
                old_finished = self.finished
                old_num_edges = self.num_edges
                old_num_nodes = self.orig_n_nodes
                self.orig_graph_dict = {"num_edges": old_num_edges, "num_nodes": old_num_nodes, "gt_Energy": self.old_gt_Energy, "pred_Energy": self.finished_energy}
            self.reset()
            self.dEnergies = []
        else:
            old_finished = self.finished

        if(self.node_embedding_type == "eigenvector"):
            edge_embedding = np.array(self.Igraph.es["edge_embedding"])
            new_embedding = np.concatenate([self.Jgraph.edges, edge_embedding], axis = -1)
            self.return_Jgraph = self.Jgraph._replace(edges = new_embedding)
        else:
            self.return_Jgraph = self.Jgraph

        return self.dones, dEnergy, 0, {"H_graph": {"jgraph":self.return_Jgraph, "Nb_ext_fields": self.Nb_external_fields}, "finished": old_finished}
        ### num edges is here unique number of edges

    def compute_Energy_full_graph(self, Nb_spins):
        init_Nb_graph = jraph.batch_np([self.init_EnergyJgraph for i in range(self.Nb)])
        print("ext fields shape",self.init_Nb_external_fields.shape)
        flattened_init_Nb_external_field = np.expand_dims(np.ravel(self.init_Nb_external_fields[:,:,0]), axis = -1)
        Energy = compute_Energy_full_graph(flattened_init_Nb_external_field,init_Nb_graph,np.expand_dims(np.ravel(Nb_spins), axis = -1))
        unnormed_Energy = self.compute_unormalized_Energy(np.ravel(Energy))
        #print("round vs not round")
        #print(unnormed_Energy, np.round(self.compute_unormalized_Energy(np.ravel(Energy)),0))
        return np.round(unnormed_Energy,0)

    def check_energy(self):
        gs = self.gs[self.perm]
        Nb_gs = np.repeat(gs[np.newaxis,:], self.Nb, axis = 0)
        Nb_gs_spins = 2*Nb_gs-1
        self.init_Nb_graph = jraph.batch_np([self.init_EnergyJgraph for i in range(self.Nb)])
        print("ext fields shape",self.init_Nb_external_fields.shape)
        flattened_init_Nb_external_field = np.expand_dims(np.ravel(self.init_Nb_external_fields[:,:,0]), axis = -1)
        Energy_check_gs = compute_Energy_full_graph(flattened_init_Nb_external_field,self.init_Nb_graph,np.expand_dims(np.ravel(Nb_gs_spins), axis = -1), A= 1., B = 1.)
        Energy_check = compute_Energy_full_graph(flattened_init_Nb_external_field, self.init_Nb_graph,np.expand_dims(np.ravel(self.Nb_spins), axis = -1), A= 1., B = 1.)

        # ground_state = self.ground_state[self.perm]
        # ground_state = np.repeat(ground_state[np.newaxis,:], self.Nb, axis = 0)
        #
        # gs_Energy = compute_Energy_full_graph(self.init_Nb_graph,np.expand_dims(np.ravel(ground_state), axis = -1), A= 1., B = 1.)
        # gs_Energy = self.normalize_Energy(gs_Energy)

        T_Nb_Energy = np.array(self.dEnergies)
        #print(self.Nb_spins)
        print("Energy_check", self.compute_unormalized_Energy(np.ravel(Energy_check)))
        print("denergy check", self.compute_unormalized_Energy(np.sum(T_Nb_Energy, axis = 0)))
        print("Energy_check_gs", self.compute_unormalized_Energy(np.ravel(Energy_check_gs)))
        print("gs energy", self.compute_unormalized_Energy( self.gt_Energy), np.sum(gs))
        #print("gs energy", self.compute_unormalized_Energy(gs_Energy))
        #self.check_for_violations(self.init_jgraph)
        ### Max Cut calculations
        n_edges = self.init_EnergyJgraph.n_edge - 2*self.orig_n_nodes
        senders = self.init_EnergyJgraph.senders
        receivers = self.init_EnergyJgraph.receivers
        gs_spins = 2*gs-1
        MC_value = (n_edges - np.sum(gs_spins[senders]* gs_spins[receivers]) + 2*self.orig_n_nodes)/4
        MC_value_2 = (n_edges/4 -self.compute_unormalized_Energy( self.gt_Energy)/2)
        MaxCut_value_list = []
        for s, r in zip(senders, receivers):
            if(s!= r):
                MaxCut_value_list.append((1 - (2 * gs[int(s)] - 1) * (2 * gs[int(r)] - 1)) / 4)

        print("1", sum(MaxCut_value_list), MC_value, MC_value_2)

        print("end")

    def check_for_violations(self, H_graph, min_idx):
        if(False):
            best_spins = np.expand_dims(self.Nb_spins[min_idx], axis = 0)
        else:
            best_spins = self.Nb_spins

        Energies = []
        for i in range(best_spins.shape[0]):
            Nb_bins = np.expand_dims((best_spins[i]+1)/2, axis = 0)
            HB = 1
            while(HB != 0):
                if(self.EnergyFunction == "MVC"):
                    Energy, HA, HB, HB_per_node = MVC_Energy(H_graph, Nb_bins)
                    if(HB != 0):
                        violation_idx = np.argmax(HB_per_node, axis=-1)
                        Nb_bins[np.arange(0, Nb_bins.shape[0]), violation_idx] = 1
                elif(self.EnergyFunction == "MIS" or "MaxCl" in self.EnergyFunction):
                    Energy, HA, HB, HB_per_node = MIS_Energy(H_graph, Nb_bins)
                    if(HB != 0):
                        violation_idx = np.argmax(HB_per_node, axis=-1)
                        Nb_bins[np.arange(0, Nb_bins.shape[0]), violation_idx] = 0

            if (self.EnergyFunction == "MVC"):
                Energy, HA, HB, HB_per_node = MVC_Energy(H_graph, Nb_bins)
                Energy = np.sum(Nb_bins[0, self.N_spin - self.orig_n_nodes:], axis = 0)[0]
            elif (self.EnergyFunction == "MIS" or "MaxCl" in self.EnergyFunction):
                Energy, HA, HB, HB_per_node = MIS_Energy(H_graph, Nb_bins)
                Energy = -np.sum(Nb_bins[0, self.N_spin - self.orig_n_nodes :], axis = 0)[0]

            self.Nb_spins[i] = 2*np.squeeze(Nb_bins, axis = 0) - 1

            Energies.append(Energy)

        Energies = np.array(Energies)

        return Energies, HA, HB


def MVC_Energy(H_graph, Nb_bins, A= 1, B = 1.1):
    ### remove self loops that were not there in original graph
    iH_graph = jutils.from_jgraph_to_igraph(H_graph)
    H_graph = jutils.from_igraph_to_jgraph(iH_graph)
    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = B * (1-Nb_bins[:,H_graph.senders]) * (1-Nb_bins[:,H_graph.receivers])
    ### TODO add HB_per_node
    Nb_Energy_messages = np.squeeze(Energy_messages, axis=-1)
    Energy_messages_Nb = np.swapaxes(Nb_Energy_messages, 0, 1)
    HB_per_node_Nb = np.zeros((H_graph.nodes.shape[0], Nb_bins.shape[0]))
    np.add.at(HB_per_node_Nb, H_graph.receivers, Energy_messages_Nb)

    HB =  0.5 * np.sum(Energy_messages, axis = -2)
    HA = A* np.sum(Nb_bins, axis = -2)
    #HB = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)
    #HA = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)

    return float(HB + HA), float(HA), float(HB), np.swapaxes(HB_per_node_Nb, 0,1)

def MIS_Energy(H_graph, Nb_bins, A= 1, B = 1.1):
    ### remove self loops that were not there in original graph
    iH_graph = jutils.from_jgraph_to_igraph(H_graph)
    H_graph = jutils.from_igraph_to_jgraph(iH_graph)

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = B * Nb_bins[:,H_graph.senders] * Nb_bins[:,H_graph.receivers]

    Nb_Energy_messages = np.squeeze(Energy_messages, axis=-1)
    Energy_messages_Nb = np.swapaxes(Nb_Energy_messages, 0, 1)
    HB_per_node_Nb = np.zeros((H_graph.nodes.shape[0], Nb_bins.shape[0]))
    np.add.at(HB_per_node_Nb, H_graph.receivers, Energy_messages_Nb)

    HB =  0.5 * np.sum(Energy_messages, axis = -2)
    HA = -A* np.sum(Nb_bins, axis = -2)
    #HB = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)
    #HA = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)
    return float(HB + HA), float(HA), float(HB), np.swapaxes(HB_per_node_Nb, 0,1)


def compute_Energy_full_graph(Nb_external_fields,H_graph, spins, A = 1.0, B = 1.1):
    import jax
    jax.config.update('jax_platform_name', "cpu")
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = np.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = np.repeat(graph_idx, n_node, axis=0)

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = H_graph.edges * spins[H_graph.senders] * spins[H_graph.receivers]
    Energy_per_node =  jax.ops.segment_sum(Energy_messages, H_graph.receivers, sum_n_node) + spins*Nb_external_fields
    Energy = jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)
    return Energy


### TODO inherit test environments
def myfunc(cfg,H_seed, mode = "train"):
    return lambda: IGraphEnv(cfg, H_seed, mode = mode)

def envstep():
    from unipath import Path
    from omegaconf import OmegaConf
    from utils import split_dataset

    from loadGraphDatasets.jraph_Dataloader import JraphSolutionDataset
    # import warnings
    # warnings.filterwarnings('error', category=UnicodeWarning)
    # warnings.filterwarnings('error', message='*equal comparison failed*')
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    p = Path( os.getcwd())
    path = p.parent.parent
    cfg = OmegaConf.load(path + "/Experiment_configs/HydraBaseConfig.yaml")
    OmegaConf.update(cfg, "Paths.path", str(path) + "/model_checkpoints", merge=True)
    OmegaConf.update(cfg, "Paths.work_path", str(path), merge=True)
    cfg["Ising_params"]["IsingMode"] = "RB_iid_200"
    cfg["Ising_params"]["EnergyFunction"] = "MVC"
    cfg["Ising_params"]["ordering"] = "BFS"
    cfg["Ising_params"]["embedding_type"] = "None"
    cfg["Train_params"]["masking"] = True
    cfg["Train_params"]["pruning"] = True
    Hb = 1
    cfg["Test_params"]["n_test_graphs"] = Hb
    Nb = 30
    cfg["Train_params"]["n_basis_states"] = Nb
    cfg["Train_params"]["seed"] = 124
    n_sampled_sites = 5
    cfg["Network_params"]["policy_MLP_features"] = [120, 64, 2**n_sampled_sites]
    mode = "test"
    shuffle_seed = cfg["Ising_params"]["shuffle_seed"]
    # split_idxs = split_dataset.get_split_datasets(Dataset, Hb)
    #
    # Dataset_index_func = lambda indices: list(map(lambda x: Dataset[x], indices))
    #
    # start_fast = time.time()
    # SpinEnv = SubprocVecEnv([myfunc(Dataset_index_func(split_idxs[H_seed]),cfg, H_seed, mode = mode) for H_seed in range(Hb)])
    # end_fast = time.time()

    start_slow = time.time()
    SpinEnv = SubprocVecEnv([myfunc(cfg, H_seed, mode = mode) for H_seed in range(Hb)])
    end_slow = time.time()


    print("slow", end_slow - start_slow)

    run(SpinEnv, Hb, Nb, n_sampled_sites)
    #run(SpinEnv, Hb, Nb)

def run(SpinEnv, Hb, Nb, n_sampled_sites):
    from jraph_utils import utils as jutils
    import jax.numpy as jnp
    ### TODO check eglibilty traces
    SpinEnv.set_attr("global_reset", True)
    orig_H_graphs_dict = SpinEnv.reset()
    orig_H_graphs = [H_graph["H_graph"]["jgraph"] for H_graph in orig_H_graphs_dict]

    Hb_Nb_H_graphs = jraph.batch_np(orig_H_graphs)

    terminated = False
    Sb_minib_graphs = []
    Energies = []
    overall_time = []
    # while(not np.all(terminated)):
    Energy_list = []
    gt_Energy_list = []
    n_edges = []
    n_nodes = []
    ### TODO track spins here
    finished_list = [False for i in range(Hb)]
    overall_time_list = []
    while(True):
        spin_value = np.random.randint(0, 2**n_sampled_sites, (Hb, Nb, 1))
        class_log_probs = np.log(0.5 * np.ones((Hb, Nb, 2 ** n_sampled_sites)))

        start = time.time()
        done, reward, terminated, H_graph_dict = SpinEnv.step(class_log_probs)
        #print("reward", reward)

        start_for_loop = time.time()
        if(np.any(done)):
            g_idx = np.arange(0, Hb)[np.array(done)]
            # print(done)
            # print(g_idx)
            orig_graph_dict = SpinEnv.get_attr("orig_graph_dict", g_idx)
            for el,idx in enumerate(g_idx):
                if(H_graph_dict[idx]["finished"] != True):
                    Energy_list.append(orig_graph_dict[el]["pred_Energy"])
                    gt_Energy_list.append(orig_graph_dict[el]["gt_Energy"])
                    print(orig_graph_dict[el]["pred_Energy"],"vs", orig_graph_dict[el]["gt_Energy"])
                    # print(H_graph_dict[idx]["finished_Energies"],  "vs ", H_graph_dict[idx]["gt_Energy"])
                    # print(SpinEnv.get_attr("gt_Energy")[idx])
                    n_edges.append(orig_graph_dict[el]["num_edges"])
                    n_nodes.append(orig_graph_dict[el]["num_nodes"])
                    print(n_edges)
                else:
                    # print("else")
                    # print(idx, len(Energy_list))
                    pass

        end_for_loop = time.time()

        start_repeat = time.time()
        done = np.repeat(done[:, np.newaxis], Nb, axis=-1)
        end_repeat = time.time()
        end = time.time()
        print("repeat", end_repeat- start_repeat)
        print("for_loop", end_for_loop-start_for_loop)
        print("duration", end - start)
        finished_list = np.array([(H_graph["finished"] or prev_finished) for (H_graph, prev_finished) in zip(H_graph_dict,finished_list)])
       # print([H_graph["batched_H_graph"].nodes.shape for H_graph in H_graph_dict])


        if(np.all(finished_list)):
            print("all graphs finished")
            print(len(Energy_list))
            break

        overall_time_list.append(start_for_loop-start)

        start_batching = time.time()
        print([H_graph["H_graph"]["jgraph"].nodes.shape for H_graph in H_graph_dict])
        val_Hb_graphs = jraph.batch_np([H_graph["H_graph"]["jgraph"] for H_graph in H_graph_dict])
        val_Hb_graphs_jnp = jutils.cast_Tuple_to(copy.deepcopy(val_Hb_graphs), np_ = jnp)
        #b = jnp.ones((10,))

        end_batching = time.time()

        print("batching time v2", end_batching - start_batching)

        Energies.append(reward)
        overall_time.append(end - start)

    print("overall time needed", np.sum(overall_time))
    print("mean_time" , np.mean(np.array(end-start)))
    pred_Energies = np.array(Energy_list)
    gt_Energies = np.array(gt_Energy_list)
    min_Energy = np.expand_dims(np.min(pred_Energies, axis=-1), axis=-1)
    n_edges = np.expand_dims(np.array(n_edges), axis=-1)

    rel_error = np.mean(np.abs(gt_Energies - pred_Energies) / np.abs(gt_Energies))
    best_rel_error = np.mean(np.abs(gt_Energies - min_Energy) / np.abs(gt_Energies))
    pred_energy = np.mean(pred_Energies, axis = -1)
    gt_energy = np.mean(gt_Energies, axis = -1)

    MC_Value = np.mean(n_edges / 4 - pred_Energies / 2)
    best_MC_Value = np.mean(n_edges / 4 - min_Energy / 2)
    gt_MC_Value = np.mean(n_edges / 4 - gt_Energies / 2)

    rel_error = np.mean(np.abs(gt_Energies - pred_Energies) / np.abs(gt_Energies))
    best_rel_error = np.mean(np.abs(gt_Energies - min_Energy) / np.abs(gt_Energies))
    pred_energy = np.mean(pred_Energies)
    gt_energy = np.mean(gt_Energies)

    B = 1.1
    APR_per_graph = (pred_Energies) / (gt_Energies)
    best_APR_per_graph = (min_Energy) / (gt_Energies)
    APR = np.mean(APR_per_graph)
    best_APR = np.mean(best_APR_per_graph)

    print("here", gt_energy, gt_Energies.shape)



def calc_traces( rewards, values, not_dones, time_horizon =5, gamma = 1., lam = 1.):
    advantage = np.zeros_like(values)
    for t in reversed(range(time_horizon)):
        delta = rewards[t] + gamma * not_dones[t+1]*values[t+1] - values[t]
        advantage[t] = delta + gamma*lam *not_dones[t+1]*advantage[t+1]

    value_target = (advantage + values)[0:time_horizon]
    return value_target, advantage[0:time_horizon]

if(__name__ == "__main__"):

    envstep()
    pass