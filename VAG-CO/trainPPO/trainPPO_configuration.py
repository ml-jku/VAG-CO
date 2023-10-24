import copy
import pickle
import jax
import jax.random
import jraph
import wandb
import jax.numpy as jnp
from train.ConfigureRun import configure_run_ReplayBuffer
import train.LoadUtils as LoadUtils
from train.LogMetrics import SaveModel
from tqdm import tqdm
from functools import partial
from trainPPO.OwnVecEnv import SubprocVecEnv
#from trainPPO.Nb_H_graph_env import *

from trainPPO.IGraphVecEnv.GraphEnv_train_configuration import IGraphEnv as IGraphEnv_train
from trainPPO.IGraphVecEnv.GraphEnv_eval_configuration import IGraphEnv as IGraphEnv_eval
import numpy as np
from jraph_utils import utils as jutils
from collections import Counter
import time
from jax import config
from matplotlib import pyplot as plt
from train.LRSchedule.Schedules import cosine_warmup as LRCosineSchedule
from trainPPO.DataContainer_Dataloader import ContainerDataset, concatenate_arrays
import os
from torch.utils.data import DataLoader
from utils.moving_averages import MovingAverage

config.update('jax_enable_x64', False)

# from jax.config import config
# config.update('jax_disable_jit', True)

cpu_np = np
gpu_np = jnp

### todo add dE caluclation!
class HiVNAPPo:
    def __init__(self):
        self.gamma = 1.

        self.occured_padded_graphs_env = []
        self.occured_padded_graphs_backward = []

        self.vmapped_forward_env = jax.vmap(self.forward_env, in_axes = (None, None, None, 0,0), out_axes = (0,0,0,0,0, 0))
        self.jitted_vmapped_forward_env = jax.jit(self.vmapped_forward_env)

        self.jitted_forward_env = jax.jit(partial(self.forward_env))
        self.jitted_forward_env_unpadded = jax.jit(partial(self.forward_env_unpadded))
        self.edge_list = []
        self.edge_list_backward = []
        self.tracked_jit_tuples = {}
        self.tracked_jit_tuples["eval"] = {}
        self.tracked_jit_tuples["forw"] = {}
        self.tracked_jit_tuples["backw"] = {}
        self.mov_avrg_step = 0


    def track_jitting_forward(self, padded_minib_H_graph, padded_minib_compl_H_graph):
        self.edge_list.append((padded_minib_H_graph.nodes.shape[0], padded_minib_H_graph.edges.shape[0], padded_minib_compl_H_graph.nodes.shape[0], padded_minib_compl_H_graph.edges.shape[0]))

        if(padded_minib_H_graph.nodes.shape[0] != padded_minib_compl_H_graph.nodes.shape[0]):
            print((padded_minib_H_graph.nodes.shape[0], padded_minib_H_graph.edges.shape[0], padded_minib_compl_H_graph.nodes.shape[0], padded_minib_compl_H_graph.edges.shape[0]))
            ValueError("shapes do not match")
        print("number of occurances forward")
        print(Counter(self.edge_list).keys())
        print(Counter(self.edge_list).values())

    def track_jitting_backward(self, padded_minib_H_graph, padded_minib_compl_H_graph):
        self.edge_list_backward.append((padded_minib_H_graph.nodes.shape[0], padded_minib_H_graph.edges.shape[0], padded_minib_compl_H_graph.nodes.shape[0], padded_minib_compl_H_graph.edges.shape[0]))

        if(padded_minib_H_graph.nodes.shape[0] != padded_minib_compl_H_graph.nodes.shape[0]):
            print((padded_minib_H_graph.nodes.shape[0], padded_minib_H_graph.edges.shape[0], padded_minib_compl_H_graph.nodes.shape[0], padded_minib_compl_H_graph.edges.shape[0]))
            ValueError("shapes do not match")

        print("number of occurances")
        print(Counter(self.edge_list).keys())
        print(Counter(self.edge_list).values())

    def _get_closest_larger_value(self, number, values):
        closest_larger_value = None
        closest_distance = float('inf')
        for value in values:
            if value > number and value - number < closest_distance:
                closest_larger_value = value
                closest_distance = value - number
        return closest_larger_value

    def track_compl_graph_edges(self, padded_minib_H_graph, padded_minib_compl_H_graph, k = 1.4, mode = "eval"):
        H_graph_edges = padded_minib_H_graph.edges.shape[0]
        H_graph_nodes = padded_minib_H_graph.nodes.shape[0]

        compl_H_graph_edges = padded_minib_compl_H_graph.edges.shape[0]
        nearest_number_edges = jutils._nearest_bigger_power_of_k(compl_H_graph_edges, k=k)

        H_graph_tuple = (H_graph_nodes, H_graph_edges)

        if(H_graph_tuple in self.tracked_jit_tuples[mode]):
            if(nearest_number_edges > max(self.tracked_jit_tuples[mode][H_graph_tuple])):
                print("new n_edges are added")
                self.tracked_jit_tuples[mode][H_graph_tuple].append(nearest_number_edges)
            else:
                print("closest jit function can be taken")
                nearest_number_edges = self._get_closest_larger_value(nearest_number_edges, self.tracked_jit_tuples[mode][H_graph_tuple])
        else:
            print("jit function is initialised")
            self.tracked_jit_tuples[mode][H_graph_tuple] = [nearest_number_edges]
        return nearest_number_edges

    def check_if_jitted(self, graph_info_list, H_graph, mode = "env"):
        num_nodes = cpu_np.sum(H_graph.n_node)
        num_edges = cpu_np.sum(H_graph.n_edge)
        check_tuple = (num_nodes, num_edges)
        if(check_tuple not in graph_info_list):
            graph_info_list.append(check_tuple)

        print("there are ",len(graph_info_list), f"jitted {mode} graph functions")
        print("nodes", cpu_np.sum(H_graph.n_node), H_graph.nodes.shape)
        print("edges", cpu_np.sum(H_graph.n_edge), H_graph.edges.shape)
        print("receivers", H_graph.receivers.shape)
        print("senders", H_graph.senders.shape)
        print("globals", H_graph.globals.shape)

    def calc_traces(self, rewards, values, not_dones):
        advantage = cpu_np.zeros_like(values)
        for t in reversed(range(self.time_horizon)):
            delta = rewards[t] + self.gamma * not_dones[t+1]*values[t+1] - values[t]
            advantage[t] = delta + self.gamma*self.lam *not_dones[t+1]*advantage[t+1]

        value_target = (advantage + values)[0:self.time_horizon]
        return value_target, advantage[0:self.time_horizon]

    def collate_data_dict(self, Hb_data_dict):

        Hb_ext_field_arr = np.concatenate([data_dict["external_fields"] for data_dict in Hb_data_dict], axis=1)
        minib_H_graphs = jraph.batch_np([data_dict["graphs"] for data_dict in Hb_data_dict])
        array_list = [data_dict["arrays"] for data_dict in Hb_data_dict]

        Hb_array_dict = concatenate_arrays(array_list)

        minib_value_target = Hb_array_dict["value_targets"]
        minib_A_k = Hb_array_dict["advantage"]
        minib_actions = Hb_array_dict["actions"]
        minib_log_probs = Hb_array_dict["log_probs"]
        masks = Hb_array_dict["masks"]

        minib_A_k = (minib_A_k- np.mean(minib_A_k))/(np.std(minib_A_k) + 1e-10)

        return minib_H_graphs, masks, minib_actions, minib_A_k, minib_log_probs, minib_value_target, Hb_ext_field_arr

    def init_attributes(self, config):
        self.cfg = config
        self.n_test_graphs = config["Test_params"]["n_test_graphs"]
        self.updates_per_iterations = config["Train_params"]["PPO"]["updates_per_iteration"]
        self.clip = config["Train_params"]["PPO"]["clip_value"]
        self.mini_Nb = config["Train_params"]["PPO"]["mini_Nb"]
        self.mini_Hb = config["Train_params"]["PPO"]["mini_Hb"]
        self.mini_Sb = config["Train_params"]["PPO"]["mini_Sb"]

        self.lam = config["Train_params"]["PPO"]["lam"]
        self.alpha = config["Train_params"]["PPO"]["alpha"]
        self.Nb = config["Train_params"]["n_basis_states"]
        self.Hb = config["Train_params"]["H_batch_size"]
        self.time_horizon = config["Train_params"]["PPO"]["time_horizon"]

        self.EnergyFunction = config["Ising_params"]["EnergyFunction"]

        self.graph_padding_factor = config["Ising_params"]["graph_padding_factor"]
        self.compl_graph_padding_factor = config["Ising_params"]["compl_graph_padding_factor"]

        self.policy_global_features = config["Network_params"]["policy_MLP_features"]
        self.n_classes = self.policy_global_features[-1]
        self.n_sampled_sites = int(np.log2(self.n_classes))

        self.graph_padding_factor = config["Ising_params"]["graph_padding_factor"]

        if("pruning" not in config["Train_params"].keys()):
            config["Train_params"]["pruning"] = True
        if("masking" not in config["Train_params"].keys()):
            config["Train_params"]["masking"] = False
        if("reversed_disjoint_graph_ordering" not in config["Ising_params"].keys()):
            config["Ising_params"]["reversed_disjoint_graph_ordering"] = False
        if("self_loops" not in config["Ising_params"].keys()):
            config["Ising_params"]["self_loops"] = False
        if ("centrality" not in config["Ising_params"].keys()):
            config["Ising_params"]["centrality"] = False


    ### TODO check if graph generation is correct
    def train(self, config, load_path = None):

        ### Flags
        #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        #os.environ["WANDB_MODE"] = "offline"
        self.cfg = config
        config["sparse_graphs"] = True
        path = config["Paths"]["path"]
        group = config["group"]
        job_type = config["job_type"]
        N_warmup = config["Anneal_params"]["N_warmup"]
        N_anneal = config["Anneal_params"]["N_anneal"]
        N_equil = config["Anneal_params"]["N_equil"]
        seed = config["Train_params"]["seed"]
        batch_epochs = config["Train_params"]["batch_epochs"]
        self.n_test_graphs = config["Test_params"]["n_test_graphs"]
        T = config["Anneal_params"]["Temperature"]
        self.updates_per_iterations = config["Train_params"]["PPO"]["updates_per_iteration"]
        self.clip = config["Train_params"]["PPO"]["clip_value"]
        self.mini_Nb = config["Train_params"]["PPO"]["mini_Nb"]
        self.mini_Hb = config["Train_params"]["PPO"]["mini_Hb"]
        self.mini_Sb = config["Train_params"]["PPO"]["mini_Sb"]

        self.lam = config["Train_params"]["PPO"]["lam"]
        self.alpha = config["Train_params"]["PPO"]["alpha"]
        self.Nb = config["Train_params"]["n_basis_states"]
        self.Hb = config["Train_params"]["H_batch_size"]

        self.lr = config["Train_params"]["lr"]
        self.lr_min = self.lr*config["Train_params"]["lr_alpha"]

        self.time_horizon = config["Train_params"]["PPO"]["time_horizon"]
        self.EnergyFunction = config["Ising_params"]["EnergyFunction"]

        self.graph_padding_factor = config["Ising_params"]["graph_padding_factor"]
        self.compl_graph_padding_factor = config["Ising_params"]["compl_graph_padding_factor"]
        self.policy_global_features = config["Network_params"]["policy_MLP_features"]
        self.n_classes = self.policy_global_features[-1]
        self.n_sampled_sites = int(np.log2(self.n_classes))
        self.config = config
        alpha = config["Train_params"]["PPO"]["mov_avrg"]
        self.MovingAverage = MovingAverage(alpha, alpha)

        epochs = int(N_warmup + N_anneal + N_equil)

        cosine_schedule_func = lambda epoch: LRCosineSchedule(epoch,  epochs, N_warmup, N_equil, N_anneal, self.lr, lr_min = self.lr_min)
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.login()
        project_name = config["project"]
        run = wandb.init(mode = "online", project=project_name, reinit=True, group=group, job_type=job_type, config=config, settings=wandb.Settings(_service_wait=300))
        run.name = wandb.run.id


        wandb.define_metric("val/step")
        wandb.define_metric("val/*", step_metric="val/step")

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("schedules/*", step_metric="train/step")

        print("saving is", self.cfg["Save_settings"]["save_params"])
        if(self.cfg["Save_settings"]["save_params"] == True):
            run_path = SaveModel.create_save_path(project_name, "", N_anneal, run, path, config)

        if(load_path != "None"):
            params, opt_state, epoch = LoadUtils.checkpoint(load_path)
            print("Load model from checkpoint at epoch", epoch)
            params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
            ### TODO overwrite anneal scheduler
            RNN, ICGenerator, anneal_scheduler = configure_run_ReplayBuffer(config, sparse_graphs=True)

            self.RNN = RNN
            self.ICGenerator = ICGenerator

            RNN.key = jax.random.PRNGKey(seed + 10)
            RNN_key = RNN.key
            Nb_RNN_key = jax.random.split(RNN_key, self.Nb)
            epoch_arr = np.arange(epoch, epochs)
            self.train_loop(RNN, params, opt_state, Nb_RNN_key, epoch_arr, epochs, batch_epochs, N_warmup, N_equil, N_anneal, anneal_scheduler, T, run_path)
        else:
            print("Reinitialize Model")
            RNN, ICGenerator, anneal_scheduler = configure_run_ReplayBuffer(config, sparse_graphs=True)
            self.RNN = RNN
            self.ICGenerator = ICGenerator

            RNN_key = RNN.key
            Nb_RNN_key = jax.random.split(RNN_key, self.Nb)

            params = RNN.get_params()
            opt_state = RNN.get_opt_states()

            if(config["Anneal_params"]["schedule"] == "cosine_frac"):
                epoch_arr = np.arange(0, config["Anneal_params"]["reps"] *epochs)
            else:
                epoch_arr = np.arange(0, epochs)
            self.train_loop(RNN, params, opt_state, Nb_RNN_key,epoch_arr, epochs, batch_epochs, N_warmup, N_equil, N_anneal, anneal_scheduler, T, run_path)


    def train_loop(self, RNN, params, opt_state, Nb_RNN_key, epoch_arr, epochs, batch_epochs, N_warmup, N_equil, N_anneal, anneal_scheduler, T, run_path):
        self.init_VecEnv()
        RNN.T = T
        best_metrics = {}
        best_metrics["best_rel_error"] = np.inf
        best_metrics["rel_error"] = np.inf
        best_metrics["best_energy"] = np.inf
        best_metrics["mean_energy"] = np.inf

        print("Start Training")
        train_step = 1
        val_epoch = int(0.05 * epochs)
        finer_val_epoch = int(0.01 * epochs)
        rel_error_dict = {}
        rel_error_dict["epoch"] = []
        rel_error_dict["val_rel_error"] = []
        rel_error_dict["best_val_rel_error"] = []

        for epoch in tqdm(epoch_arr):

            ### nomalize A_k
            loss_dict = {}
            loss_dict["overall_losses"] = []
            loss_dict["actor_losses"] = []
            loss_dict["critic_losses"] = []

            RNN.T = anneal_scheduler(epoch*batch_epochs, epochs*batch_epochs, N_warmup*batch_epochs, N_equil*batch_epochs, N_anneal*batch_epochs, T)

            start_env = time.time()
            DataContainerList, t_Hb_Nb_Energy_per_action, t_Hb_Nb_log_prob, t_Hb_Nb_rewards, Hb_Nb_randomness, Nb_RNN_key = self.make_env_steps(params, RNN.T, Nb_RNN_key)
            end_env = time.time()


            if(epoch == epoch_arr[0]):
                self.ContainerDataset = ContainerDataset(self.config)
                self.ContainerDataset.overwrite_data(DataContainerList)
                self.ContainerDataset.reshuffle()
                self.ContainerDataloader = DataLoader(self.ContainerDataset, batch_size=self.mini_Hb, shuffle=True,  persistent_workers=False,
                                                      num_workers=2, collate_fn=self.collate_data_dict)
            else:
                self.ContainerDataset.overwrite_data(DataContainerList)

            minibatch_Entropy = -cpu_np.sum(t_Hb_Nb_log_prob, axis = 0)


            start_backprob = time.time()
            for batch_epoch in range(batch_epochs):
                log_dict, params, opt_state = self.train_on_minibatches_Dataset(RNN, params, opt_state, loss_dict)

                log_dict = {}
                for key in loss_dict:
                    metric = cpu_np.array(loss_dict[key])
                    mean_metric = cpu_np.mean(metric)

                    log_dict[key] = mean_metric

                wandb.log(log_dict)

            end_backprob = time.time()

            log_dict = {}
            ### TODO log time in more detail
            overall_env_time = end_env - start_env
            overall_backprob_time = end_backprob - start_backprob
            train_step += 1
            log_dict["time/overall_env_steps"] =  overall_env_time
            log_dict["time/overall_grad_steps"] =  overall_backprob_time
            ### TODO fix loging of log probs
            Hb_Nb_randomness = np.where(Hb_Nb_randomness != np.inf, Hb_Nb_randomness, np.zeros_like(Hb_Nb_randomness))
            mask = 1*(Hb_Nb_randomness != 0)
            log_dict["train/mean_action_prob"] =  cpu_np.sum(mask*Hb_Nb_randomness)/np.sum(mask)

            FreeEnergy = cpu_np.mean(cpu_np.sum( -t_Hb_Nb_rewards,axis = 0))
            log_dict["train/Free_Energy"] = FreeEnergy
            log_dict["train/reward"] = cpu_np.mean( t_Hb_Nb_rewards)
            log_dict["train/mean_reward"] = self.MovingAverage.mean_value
            log_dict["train/std_reward"] = self.MovingAverage.std_value
            log_dict["train/Entropy"] = cpu_np.mean(minibatch_Entropy)
            log_dict["train/Energy_per_action"] = cpu_np.mean(t_Hb_Nb_Energy_per_action)
            log_dict["train/step"] = train_step

            # new_lr = cosine_schedule_func(epoch)
            # RNN.update_optimizer(new_lr, params)

            log_dict["schedules/lr"] = RNN.curr_lr
            log_dict["schedules/T"] = RNN.T
            wandb.log(log_dict)

            log_dict = {}
            log_dict["val/step"] = epoch

            SaveModel.save_curr_model(params, opt_state, run_path, epoch, T, self.config)
            if(epoch % val_epoch == 0 or epoch == epochs -1 or epoch == 0 or (100*RNN.T/T < 2 and epoch % finer_val_epoch == 0) ):
                log_dixt, Nb_RNN_key = self.make_env_steps_validation(params, Nb_RNN_key, log_dict)

                best_rel_error = log_dict["val/best_rel_error"]
                rel_error =  log_dict["val/rel_error"]
                best_energy = log_dict[f"val/min_pred_mean_energy"]
                mean_energy = log_dict[f"val/pred_mean_energy"]

                rel_error_dict["epoch"].append(epoch)
                rel_error_dict["val_rel_error"].append(rel_error)
                rel_error_dict["best_val_rel_error"].append(best_rel_error)

                with open(os.path.join(run_path, f"rel_error_dict.pickle"), "wb") as file:
                    pickle.dump(rel_error_dict, file)

                if(self.cfg["Save_settings"]["save_params"] == True):
                    if(self.cfg["Save_settings"]["save_mode"] != "best"):
                        SaveModel.save_best_model_params(params, opt_state, "val_rel_error", run_path, add_string= f"_T_{RNN.T}_epoch_{epoch}")
                    else:
                        if (best_energy < best_metrics["best_energy"]):
                            best_metrics["best_energy"] = best_energy
                            SaveModel.save_best_model_params(params, opt_state, "val_best_rel_error", run_path)
                        if (mean_energy < best_metrics["mean_energy"]):
                            best_metrics["mean_energy"] = mean_energy
                            SaveModel.save_best_model_params(params,opt_state, "val_rel_error", run_path)

            wandb.log(log_dict)

    def train_on_minibatches_Dataset(self, RNN, params, opt_state, loss_dict):
        self.ContainerDataset.reshuffle()
        start_time = time.time()
        for Cidx , (minib_H_graphs, minib_masks, minib_actions, minib_A_k, minib_log_probs, minib_value_target, concatenated_Nb_external_fields) in enumerate(self.ContainerDataloader):
            pad_time = time.time()

            padded_minib_H_graph, padded_concatenated_Nb_external_fields = jutils.pad_graph_and_ext_fields_to_nearest_power_of_k(
                minib_H_graphs, concatenated_Nb_external_fields,
                k=self.graph_padding_factor, min_nodes = self.n_sampled_sites)
            end_pad_time = time.time()


            concatenated_arrays = np.concatenate( [minib_actions[:,:,np.newaxis], minib_A_k[:,:,np.newaxis], minib_log_probs[:,:,np.newaxis], minib_value_target[:,:,np.newaxis]], axis = -1)
            empty_globals = np.zeros((minib_log_probs.shape[1]+1, minib_log_probs.shape[0], 4))
            empty_globals[:-1,:,:] = np.swapaxes(concatenated_arrays, 0,1)
            padded_minib_H_graph = padded_minib_H_graph._replace(globals = empty_globals)

            concat_time = time.time()

            ### TODO chekc if commenting jnp casting works
            padded_minib_H_graph = jutils.cast_Tuple_to_float32(padded_minib_H_graph, np_=jnp)
            cast_time = time.time()

            padded_minib_masks = np.ones((minib_masks.shape[1]+1, minib_masks.shape[0], minib_masks.shape[-1]), dtype=np.float32)
            padded_minib_masks[:-1,:,:] = np.swapaxes(minib_masks, 0,1)
            padded_minib_masks = jnp.asarray(padded_minib_masks)

            start_jit_time = time.time()

            params, opt_state, (overall_loss, (actor_loss, critic_loss)) = RNN.backward_sparse_reduced_Nb_Hb_loss(params, opt_state, padded_minib_H_graph, padded_minib_masks, padded_concatenated_Nb_external_fields)

            #print("ceck device of array", actor_loss.device_buffer.device())
            # if(np.isnan(overall_loss) ):
            #     print("Loss is Nan")
            #     while(True):
            #         pass
            ### TODO check if deleting is neccesary
            del padded_minib_H_graph
            del padded_minib_masks

            # end_jit_time = time.time()
            # overall_time = end_jit_time-pad_time
            # print("jit_time", end_jit_time - start_jit_time, (end_jit_time - start_jit_time)/overall_time)
            # print("padding time", end_pad_time-pad_time, (end_pad_time-pad_time)/overall_time)
            # print("concat time", concat_time - end_pad_time, (concat_time - end_pad_time)/overall_time)
            # print("cast time", cast_time-concat_time, (cast_time-concat_time)/overall_time)
            # print("overall time", overall_time)

            print("losses")
            print(overall_loss, actor_loss, critic_loss)

            loss_dict["overall_losses"].append(overall_loss)
            loss_dict["actor_losses"].append(actor_loss)
            loss_dict["critic_losses"].append(critic_loss)
            #wandb.log({"inner_loop/overall_losses":overall_loss, "inner_loop/actor_losses": actor_loss, "inner_loop/critic_losses": critic_loss})
            start_time = time.time()

        return loss_dict, params, opt_state


    def init_VecEnv(self):
        shuffle_seed = self.cfg["Ising_params"]["shuffle_seed"]
        #val_Dataset = JraphSolutionDataset(self.cfg, mode="val", seed=shuffle_seed)
        #train_Dataset = JraphSolutionDataset(self.cfg, mode="train", seed=shuffle_seed)
        self.SpinEnv = SubprocVecEnv([myfunc(self.cfg, H_seed, mode = "train") for H_seed in range(self.Hb)])
        self.ValSpinEnv = SubprocVecEnv([myfunc(self.cfg, H_seed, mode = "val") for H_seed in range(self.n_test_graphs)])

        H_graph_dict = self.SpinEnv.reset()
        # self.init_graph_list = self.SpinEnv.get_attr("init_EnergyJgraph")
        # self.Nb_spin_list = [jnp.ones((self.Nb,graph.nodes.shape[0],1)) for graph in self.init_graph_list]
        self.Hb_graph_list = [H_graph["H_graph"]["jgraph"] for H_graph in H_graph_dict]
        self.Hb_graphs = jraph.batch_np(self.Hb_graph_list)
        self.Hb_Nb_ext_fields_list = [H_graph["H_graph"]["Nb_ext_fields"] for H_graph in H_graph_dict]
        self.Energy_list = []

    def init_TestEnv(self, dataset = "Test"):
        self.TestSpinEnv = SubprocVecEnv([myfunc(self.cfg, H_seed, mode = dataset) for H_seed in range(self.n_test_graphs)])
        pass

    def eval_CE(self, params, cfg, n_test_graphs, padding_factor = 2., Nb = 30):
        self.graph_padding_factor = padding_factor
        cfg["Test_params"]["n_test_graphs"] = n_test_graphs
        self.init_attributes(cfg)
        self.n_test_graphs = n_test_graphs
        self.graph_padding_factor = padding_factor
        self.Nb = Nb
        self.cfg["Train_params"]["n_basis_states"] = Nb
        RNN, _, _ = configure_run_ReplayBuffer(self.cfg, sparse_graphs=True)
        RNN_key = RNN.key
        self.RNN = RNN
        Nb_RNN_key = jax.random.split(RNN_key, self.Nb)
        log_dict = {}
        gt_Energy_list = []
        pred_Energy_list = []
        log_dict, Nb_RNN_key = self.make_env_steps_test(params, Nb_RNN_key, log_dict, sampling_mode = "CE")
        return log_dict

    def eval_on_testdata(self, params, cfg, n_test_graphs, padding_factor = 2., n_perm = 0, Nb = 8, mode = "perm", dataset = "test", rand_params = False): # modes = perm, perm+Nb, perm+beam_search, Nb, beam_search
        self.n_test_graphs = n_test_graphs
        if(mode == "normal"):
            self.Nb = Nb
            sampling_mode = "normal"
            self.n_perm = 1
        elif(mode == "OG"):
            self.Nb = 1
            self.n_perm = n_perm
            self.n_test_graphs = n_test_graphs
            sampling_mode = "greedy"
        elif(mode == "perm+Nb"):
            self.Nb = Nb
            sampling_mode = "normal"
            self.n_perm = n_perm
        elif(mode == "perm+beam_search"):
            self.Nb = Nb
            sampling_mode = "beam_search"
            self.n_perm = n_perm
        elif(mode == "beam_search"):
            self.Nb = Nb
            sampling_mode = "beam_search"
            self.n_perm = 1
        else:
            ValueError("The mode", mode, "is not implemented")

        self.graph_padding_factor = padding_factor
        cfg["Test_params"]["n_test_graphs"] = self.n_test_graphs
        cfg["Train_params"]["n_basis_states"] = self.Nb
        self.init_attributes(cfg)
        self.graph_padding_factor = padding_factor

        params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
        print("loaded params dtype", jax.tree_util.tree_map(lambda x: type(x), params))

        self.cfg["Train_params"]["n_basis_states"] = self.Nb
        RNN, ICGenerator, _ = configure_run_ReplayBuffer(self.cfg, sparse_graphs=True)
        ### todo remove random params from rnn again
        if(rand_params):
            print("random params are used")
            params = RNN.params

        RNN_key = RNN.key
        self.RNN = RNN
        Nb_RNN_key = jax.random.split(RNN_key, self.Nb)

        self.init_TestEnv(dataset = dataset)
        self.TestSpinEnv.set_attr("sampling_mode", sampling_mode)

        gt_Energy_list = []
        pred_Energy_list = []
        time_per_graph_list = []
        n_nodes = []
        n_edges = []
        for i in range(self.n_perm + 1):
            log_dict = {}
            old_log_dict, Nb_RNN_key = self.make_env_steps_test(params, Nb_RNN_key, log_dict, sampling_mode = sampling_mode)
            if(i == 0):
                ### reset to measure jit time
                del self.TestSpinEnv
                self.init_TestEnv(dataset = dataset)
                self.TestSpinEnv.set_attr("sampling_mode", sampling_mode)
            else:
                gt_Energies = old_log_dict["test/gt_Energies"]
                pred_Energies = old_log_dict["test/pred_Energies"]
                time_per_graph_list.append(old_log_dict["test/time_per_graph"])

                gt_Energy_list.append(gt_Energies)
                pred_Energy_list.append(pred_Energies)
                n_edges.append(old_log_dict["test/n_edges"])
                n_nodes.append(old_log_dict["test/n_nodes"])



        gt_Energy_arr = np.array(gt_Energy_list)
        print("gt_Energy_arr", gt_Energy_arr.shape)
        Hb_Nb_gt_Energy_arr = np.swapaxes(gt_Energy_arr, 0,1)

        pred_Energy_arr = np.array(pred_Energy_list)
        print("pred_Energy_arr", pred_Energy_arr.shape)
        Hb_Nb_pred_Energy_arr = np.swapaxes(pred_Energy_arr, 0,1)

        log_dict = {}

        log_dict["pred_Energy_per_graph"] = Hb_Nb_pred_Energy_arr
        log_dict["gt_Energy_per_graph"] = Hb_Nb_gt_Energy_arr[:,0,:]
        log_dict["gt_Energy_arr"] = gt_Energy_arr
        log_dict["pred_Energy_arr"] = pred_Energy_arr
        log_dict["n_edges"] = np.array(n_edges)
        log_dict["n_nodes"] = np.array(n_nodes)
        log_dict["time_per_graph"] = np.array(time_per_graph_list)

        return log_dict

    ### TODO vmap this
    def forward_env(self, params, padded_H_graph, compl_graph, Ext_fields, RNN_key):
        return self.RNN.SampleGRNN.apply(params, padded_H_graph, compl_graph, Ext_fields, None, RNN_key, None)

    def forward_env_unpadded(self, params, padded_H_graph, compl_graph, RNN_key):
        return self.RNN.SampleUnpaddedGRNN.apply(params, padded_H_graph, compl_graph, None, RNN_key, None)

    def make_env_steps(self, params, T, RNN_key):
        print("start env steps")
        DataContainerList = []

        t_Hb_Nb_Energy_per_action = np.zeros((self.time_horizon, self.Hb, self.Nb, 1))
        t_Hb_Nb_log_prob = np.zeros((self.time_horizon, self.Hb, self.Nb,1))
        t_Hb_Nb_rewards = np.zeros((self.time_horizon, self.Hb, self.Nb,1))
        t_Hb_Nb_randomness = np.zeros((self.time_horizon, self.Hb, self.Nb,1))

        for i in range(self.time_horizon + 1):
            concatenated_Nb_external_fields = np.concatenate(self.Hb_Nb_ext_fields_list, axis=1)
            padded_batched_H_graph, padded_concatenated_Nb_external_fields = jutils.pad_graph_and_ext_fields_to_nearest_power_of_k(self.Hb_graphs,concatenated_Nb_external_fields,
                                                                            k=self.graph_padding_factor)

            padded_Hb_compl_graphs = padded_batched_H_graph
            padded_batched_H_graph = jutils.cast_Tuple_to_float32(padded_batched_H_graph, np_=gpu_np)


            padded_concatenated_Nb_external_fields = jnp.array(padded_concatenated_Nb_external_fields)
            #print("MAKE TRAIN STEP")
            value, log_probs, sampled_bin_values, log_probs_classes, logits, RNN_key = self.jitted_vmapped_forward_env(params, padded_batched_H_graph,padded_Hb_compl_graphs, padded_concatenated_Nb_external_fields, RNN_key)

            ### TODO masking on gpu


            Nb_Hb_logits = cpu_np.reshape(logits, (self.Nb, self.Hb, self.n_classes))
            Hb_Nb_logits = np.swapaxes(Nb_Hb_logits, 0, 1)
            Nb_Hb_values = cpu_np.reshape(value, (self.Nb, self.Hb,1))
            Hb_Nb_values = np.swapaxes(Nb_Hb_values,0,1)

            Hb_Nb_Temperatures = T*np.ones_like(Hb_Nb_values)

            data = np.concatenate([Hb_Nb_logits, Hb_Nb_values, Hb_Nb_Temperatures], axis=-1)

            done, Hb_Nb_Energy, _, H_graph_dict = self.SpinEnv.step(data)

            if(np.any(done)):
                g_idx = np.arange(0, self.Hb)[np.array(done)]
                # new_init_EnergyJgraph_list = self.SpinEnv.get_attr("init_EnergyJgraph", g_idx)
                #
                # for el in g_idx:
                #     self.init_graph_list[el] = new_init_EnergyJgraph_list[el]
                #     self.Nb_spin_list[el] = jnp.ones((self.Nb,new_init_EnergyJgraph_list[el].nodes.shape[0],1))
                # print(done)
                # print(g_idx)
                dc = self.SpinEnv.get_attr("filled_DataContainer", g_idx)
                DataContainerList.extend(dc)

            self.Hb_graph_list = [H_graph["H_graph"]["jgraph"] for H_graph in H_graph_dict]
            self.Hb_graphs = jraph.batch_np(self.Hb_graph_list)
            self.Hb_Nb_ext_fields_list = [H_graph["H_graph"]["Nb_ext_fields"] for H_graph in H_graph_dict]

            # print("forward pass", end-start)
            # print("for loops", end_list2 - start_list2)
            # print("env step", end_env_step-start_env_step)
            # print("padding step", end_padding-start_padding)
            # print("cast to GPU", end_cast_to_GPU- start_cast_to_GPU)
            if(i != self.time_horizon):
                Hb_Nb_Energy = np.expand_dims(Hb_Nb_Energy, axis = -1)
                t_Hb_Nb_Energy_per_action[i,:,:] = Hb_Nb_Energy
                ### TODO fix these logging metrics
                Hb_Nb_log_probs = np.array([H_graph["log_probs"] for H_graph in H_graph_dict])
                Hb_Nb_log_probs = np.expand_dims(Hb_Nb_log_probs, axis = -1)
                t_Hb_Nb_log_prob[i,:,:] = Hb_Nb_log_probs
                t_Hb_Nb_rewards[i,:,:] = -(Hb_Nb_Energy + T*Hb_Nb_log_probs)
                randomness = np.array([H_graph["randomness"] for H_graph in H_graph_dict])
                Hb_Nb_randomness = np.expand_dims(randomness, axis = -1)
                t_Hb_Nb_randomness[i,:,:] = Hb_Nb_randomness

        mean_reward, std_reward = self.MovingAverage.update_mov_averages(t_Hb_Nb_rewards)
        self.SpinEnv.set_attr("mov_reward", [mean_reward, std_reward])

        return DataContainerList, t_Hb_Nb_Energy_per_action, t_Hb_Nb_log_prob, t_Hb_Nb_rewards, t_Hb_Nb_randomness, RNN_key

    def make_evaluation_steps(self, params, RNN_key, SpinEnv, log_dict, mode = "val", sampling_mode = "normal"):
        ### TODO adapt test env to logit changes
        SpinEnv.set_attr("global_reset", True)
        H_graph_dict = SpinEnv.reset()
        val_Hb_graphs = jraph.batch_np([H_graph["H_graph"]["jgraph"] for H_graph in H_graph_dict])
        ### TODO concatenate here
        val_Hb_Nb_ext_fields_list = [H_graph["H_graph"]["Nb_ext_fields"] for H_graph in H_graph_dict]

        Energy_list = []
        gt_Energy_list = []
        finished_list = [False for i in range(self.n_test_graphs)]
        n_edges = []
        n_nodes = []
        time_per_graph = []

        finished = False
        while(not finished):
            concatenated_Nb_external_fields = np.concatenate(val_Hb_Nb_ext_fields_list, axis=1)
            padded_batched_H_graph, padded_concatenated_Nb_external_fields = jutils.pad_graph_and_ext_fields_to_nearest_power_of_k(val_Hb_graphs,concatenated_Nb_external_fields,
                                                                            k=self.graph_padding_factor)

            padded_val_Hb_compl_graphs = padded_batched_H_graph


            ### TODO test if this is unnecessary and makes code slower? bacuase casting may be faster in jit?
            padded_batched_H_graph = jutils.cast_Tuple_to_float32(padded_batched_H_graph, np_=gpu_np)
            padded_concatenated_Nb_external_fields = jnp.array(padded_concatenated_Nb_external_fields)

            value, log_probs, sampled_bin_values, log_probs_classes, logits, RNN_key = self.jitted_vmapped_forward_env(params, padded_batched_H_graph, padded_val_Hb_compl_graphs,padded_concatenated_Nb_external_fields ,
                                                                                                  RNN_key)


            logits = np.asarray(logits)
            Nb_Hb_logits = np.reshape(logits, (self.Nb, self.n_test_graphs, self.n_classes))
            Hb_Nb_logits = np.swapaxes(Nb_Hb_logits, 0, 1)

            Hb_Nb_actions = Hb_Nb_logits

            # print("MAKE EVAL STEP")
            # print(type(Hb_Nb_actions))
            done, Energy, _, H_graph_dict = SpinEnv.step(Hb_Nb_actions)

            if (np.any(done)):
                g_idx = np.arange(0, self.n_test_graphs)[np.array(done)]
                orig_graph_dict = SpinEnv.get_attr("orig_graph_dict", g_idx)
                for k, idx in enumerate(g_idx):
                    if (H_graph_dict[idx]["finished"] != True):
                        ### TODO track HB
                        Energy_list.append(orig_graph_dict[k]["pred_Energy"])
                        gt_Energy_list.append(orig_graph_dict[k]["gt_Energy"])
                        n_edges.append(orig_graph_dict[k]["num_edges"])
                        n_nodes.append(orig_graph_dict[k]["num_nodes"])
                        # print("pred", orig_graph_dict[k]["pred_Energy"])
                        # print("gt", orig_graph_dict[k]["gt_Energy"])
                        if("time_per_graph" in orig_graph_dict[k].keys()):
                            time_per_graph.append(orig_graph_dict[k]["time_per_graph"])
                            print(np.array(Energy_list).shape, np.array(gt_Energy_list).shape)
                            print("mean AR", np.mean(np.expand_dims(np.min(np.array(Energy_list), axis = -1), axis = -1)/np.array(gt_Energy_list)))
                        #print(H_graph_dict[idx]["finished_Energies"], "vs ", H_graph_dict[idx]["gt_Energy"])
                    else:
                        # print("else")
                        # print(idx, len(Energy_list))
                        pass

            finished_list = np.array([(H_graph["finished"] or prev_finished) for (H_graph, prev_finished) in zip(H_graph_dict, finished_list)])

            if(np.all(finished_list) == True):
                break

            val_Hb_graphs = jraph.batch_np([ H_graph["H_graph"]["jgraph"] for H_graph in H_graph_dict])
            val_Hb_Nb_ext_fields_list = [H_graph["H_graph"]["Nb_ext_fields"] for H_graph in H_graph_dict]

        ### TODO report to WANDB
        pred_Energies = np.array(Energy_list)
        gt_Energies = np.array(gt_Energy_list)
        min_Energy = np.expand_dims(np.min(pred_Energies, axis = -1), axis = -1)
        n_edges = np.expand_dims(np.array(n_edges), axis = -1)
        n_nodes = np.expand_dims(np.array(n_nodes), axis = -1)

        if(mode == "test"):
            log_dict[f"{mode}/gt_Energies"] = gt_Energies
            log_dict[f"{mode}/pred_Energies"] = pred_Energies
            log_dict[f"{mode}/n_edges"] = n_edges
            log_dict[f"{mode}/n_nodes"] = n_nodes
            log_dict[f"{mode}/time_per_graph"] = time_per_graph

        rel_error = np.mean(np.abs(gt_Energies - pred_Energies)/ np.abs(gt_Energies))
        best_rel_error_per_graph = np.abs(gt_Energies - min_Energy)/ np.abs(gt_Energies)
        best_rel_error = np.mean(best_rel_error_per_graph)
        pred_energy = np.mean(pred_Energies)
        gt_energy = np.mean(gt_Energies)

        #print("finished", rel_error)
        log_dict[f"{mode}/rel_error"] = rel_error
        log_dict[f"{mode}/best_rel_error"] = best_rel_error
        log_dict[f"{mode}/pred_mean_energy"] = pred_energy
        log_dict[f"{mode}/min_pred_mean_energy"] = np.mean(min_Energy)
        log_dict[f"{mode}/gt_mean_energy"] = gt_energy


        APR_per_graph = (pred_Energies)/(gt_Energies)
        best_APR_per_graph = (min_Energy)/(gt_Energies )
        APR = np.mean(APR_per_graph)
        best_APR = np.mean(best_APR_per_graph)

        if(self.EnergyFunction == "MVC"):
            log_dict[f"{mode}/APR"] = APR
            log_dict[f"{mode}/best_APR"] = best_APR
            log_dict[f"{mode}/APR_error"] = APR - 1.
            log_dict[f"{mode}/best_APR_error"] = best_APR - 1.
        if(self.EnergyFunction == "MaxCut"):
            n_edges_no_self_loops = n_edges - 2*n_nodes
            MC_Value = np.mean(n_edges_no_self_loops/4- pred_Energies/2)
            best_MC_Value = np.mean(n_edges_no_self_loops/4- min_Energy/2)
            gt_MC_Value = np.mean(n_edges_no_self_loops/4- gt_Energies/2)
            log_dict[f"{mode}/MaxCutValue"] = MC_Value
            log_dict[f"{mode}/gt_MaxCutValue"] = gt_MC_Value
            log_dict[f"{mode}/best_MaxCutValue"] = best_MC_Value

        best_APR_per_graph = list(np.squeeze(best_APR_per_graph))
        n_edges = list(np.squeeze(n_edges))
        n_nodes = list(np.squeeze(n_nodes))
        best_rel_error_per_graph = list(np.squeeze(best_rel_error_per_graph))

        best_APR_per_graph, n_edges, n_nodes, best_rel_error_per_graph = zip(*sorted(zip(best_APR_per_graph, n_edges, n_nodes, best_rel_error_per_graph), key = lambda x: x[0]))

        fig1 = plt.figure()
        plt.subplot(3,1, 1)
        plt.title("best_rel_error per graph")
        plt.plot(np.arange(0, APR_per_graph.shape[0]), best_rel_error_per_graph, "-x", label = "APR")
        plt.xlabel("sorted Graph id")
        plt.legend()
        plt.subplot(3,1, 2)
        plt.plot(np.arange(0, APR_per_graph.shape[0]), n_nodes, "-x", label = "num_nodes")
        plt.xlabel("sorted Graph id")
        plt.legend()
        plt.subplot(3,1, 3)
        plt.plot(np.arange(0, APR_per_graph.shape[0]), n_edges, "-x", label = "num_edges")
        plt.legend()
        log_dict["val/AR_ratio"] = wandb.Image(fig1)

        fig2 = plt.figure()
        plt.title("best rel_error per graph")
        plt.plot(np.arange(0, APR_per_graph.shape[0]), best_rel_error_per_graph, "-x")

        log_dict[f"{mode}/err"] = wandb.Image(fig2)
        plt.close("all")

        return log_dict, RNN_key

    def CE_env_steps(self, params, RNN_key, SpinEnv, log_dict):
        SpinEnv.global_reset = True
        H_graph_dict = SpinEnv.reset()

        val_Hb_graphs = H_graph_dict["H_graph"]["jgraph"]
        val_Hb_Nb_ext_fields_list = [H_graph_dict["H_graph"]["Nb_ext_fields"]]


        Energy_list = []
        gt_Energy_list = []
        n_edges = []
        n_nodes = []

        initial_predictions = []

        min_energy = np.inf
        initial_pred = True
        best_Energy_found = False
        finished = False
        while(not finished):
            concatenated_Nb_external_fields = np.concatenate(val_Hb_Nb_ext_fields_list, axis=1)
            padded_batched_H_graph, padded_concatenated_Nb_external_fields = jutils.pad_graph_and_ext_fields_to_nearest_power_of_k(val_Hb_graphs,concatenated_Nb_external_fields,
                                                                            k=self.graph_padding_factor)

            padded_val_Hb_compl_graphs = padded_batched_H_graph

            padded_batched_H_graph = jutils.cast_Tuple_to_float32(padded_batched_H_graph, np_=gpu_np)

            padded_concatenated_Nb_external_fields = jnp.asarray(padded_concatenated_Nb_external_fields)

            value, log_probs, sampled_bin_values, log_probs_classes, logits, RNN_key = self.jitted_vmapped_forward_env(params, padded_batched_H_graph, padded_val_Hb_compl_graphs,padded_concatenated_Nb_external_fields ,
                                                                                                  RNN_key)

            Nb_Hb_log_probs_classes = cpu_np.reshape(log_probs_classes, (self.Nb, self.n_test_graphs, self.n_classes))
            Hb_Nb_log_probs_classes = np.swapaxes(Nb_Hb_log_probs_classes, 0,1)

            if(best_Energy_found):
                sampled_bin_values = cpu_np.reshape(sampled_bin_values, (self.Nb, self.n_test_graphs, 1))
                Hb_Nb_actions = np.swapaxes(sampled_bin_values, 0, 1)
            else:
                prev_set_spins = []
                for spin_idx in range(self.n_sampled_sites):
                    args_0, args_1 = get_args_0_and_args_1(self.n_sampled_sites, prev_set_spins, spin_idx)
                    mask_0 = np.zeros((self.n_classes))
                    mask_0[args_0] = 1.
                    Hb_Nb_probs_0 = np.exp(Hb_Nb_log_probs_classes[0, 0,:])*mask_0
                    if(args_0.shape[0] == 1):
                        probs_0 = np.zeros_like(Hb_Nb_probs_0)
                        probs_0[args_0] = 1.
                    else:
                        probs_0 = Hb_Nb_probs_0/np.sum(Hb_Nb_probs_0)
                    sampled_bin_values_0 = np.random.choice(np.arange(0, self.n_classes), p = probs_0, size = (self.Nb, 1))

                    mask_1 = np.zeros((self.n_classes))
                    mask_1[args_1] = 1.
                    Hb_Nb_probs_1 = np.exp(Hb_Nb_log_probs_classes[1, 0, :])*mask_1

                    if(args_1.shape[0] == 1):
                        probs_1 = np.zeros_like(Hb_Nb_probs_1)
                        probs_1[args_1] = 1.
                    else:
                        probs_1 = Hb_Nb_probs_1/np.sum(Hb_Nb_probs_1)

                    sampled_bin_values_1 = np.random.choice(np.arange(0, self.n_classes), p = probs_1, size = (self.Nb, 1))

                    ### TODO return spinEnv etc
                    pred_Energy_0, gt_Energy, _ , _, RNN_key = self.make_forward_pass_until_finished(SpinEnv, sampled_bin_values_0, params, RNN_key)
                    pred_Energy_1, _, _ , _, RNN_key = self.make_forward_pass_until_finished(SpinEnv, sampled_bin_values_1, params, RNN_key)

                    mean_pred_Energy_0 = np.min(pred_Energy_0)
                    mean_pred_Energy_1 = np.min(pred_Energy_1)

                    if(mean_pred_Energy_0 < mean_pred_Energy_1):
                        Hb_Nb_actions = sampled_bin_values_0
                        prev_set_spins.append(0)
                        print("take Energy 0", mean_pred_Energy_0, "probs", np.sum(Hb_Nb_probs_0))
                    elif(mean_pred_Energy_0 == mean_pred_Energy_1):
                        print("prob condition", np.sum(Hb_Nb_probs_0), np.sum(Hb_Nb_probs_1))
                        if(np.sum(Hb_Nb_probs_0) > np.sum(Hb_Nb_probs_1)):
                            Hb_Nb_actions = sampled_bin_values_0
                            prev_set_spins.append(0)
                        else:
                            Hb_Nb_actions = sampled_bin_values_1
                            prev_set_spins.append(1)
                    else:
                        Hb_Nb_actions = sampled_bin_values_1
                        prev_set_spins.append(1)
                        print("take Energy 1", mean_pred_Energy_1, "probs", np.sum(Hb_Nb_probs_1))

                    if(min([np.min(pred_Energy_0), np.min(pred_Energy_1)]) < min_energy):
                        min_energy = min([np.min(pred_Energy_0), np.min(pred_Energy_1)])

                    if(initial_pred):
                        initial_pred = False
                        initial_predictions.append(min([np.min(pred_Energy_0), np.min(pred_Energy_1)]))
                    if(min([np.min(pred_Energy_0), np.min(pred_Energy_1)]) == gt_Energy):
                        best_Energy_found = True
                        print("best Energy has been found", min([np.min(pred_Energy_0), np.min(pred_Energy_1)]), gt_Energy)
                        break

                    print("compare mean Energies")
                    print("0",mean_pred_Energy_0, "1",mean_pred_Energy_1)
                    print("spins", prev_set_spins)

            #print("origSpinEnv", SpinEnv.env_step, SpinEnv.gt_Energy )
            ### TODO can this be removed?
            done, Energy, _, H_graph_dict = SpinEnv.step(Hb_Nb_actions)

            if (np.any(done)):
                orig_graph_dict = SpinEnv.orig_graph_dict

                min_final_energy = np.min(orig_graph_dict["pred_Energy"])
                print("min final Energy", min_final_energy, "min energy", min_energy)
                if( min_final_energy < min_energy):
                    min_energy = min_final_energy
                Energy_list.append(min_energy)
                gt_Energy_list.append(orig_graph_dict["gt_Energy"])
                n_edges.append(orig_graph_dict["num_edges"])
                n_nodes.append(orig_graph_dict["num_nodes"])

                print("finished AR for this graph", orig_graph_dict["pred_Energy"]/orig_graph_dict["gt_Energy"])
                best_Energy_found = False
                initial_pred = True
                min_energy = np.inf

                print("pred energy")
                print(Energy_list)
                print("gt energy")
                print(gt_Energy_list)
                print(initial_predictions)
                print(np.array(gt_Energy_list).shape, np.array(Energy_list).shape,np.array(initial_predictions).shape)
                gt_Energy_arr = np.squeeze(np.array(gt_Energy_list), axis = -1)
                inital_pred_arr = np.array(initial_predictions)
                print(np.abs(np.array(Energy_list) / gt_Energy_arr))
                print("curr AR is", np.mean(np.abs(np.array(Energy_list) / gt_Energy_arr)), "n finished graphs", len(Energy_list))
                print("initial AR is", np.mean(np.abs(inital_pred_arr / gt_Energy_arr)))


            val_Hb_graphs =  H_graph_dict["H_graph"]["jgraph"]
            val_Hb_Nb_ext_fields_list = [H_graph_dict["H_graph"]["Nb_ext_fields"] ]

        pred_Energies = np.array(Energy_list)
        gt_Energies = np.array(gt_Energy_list)
        log_dict["pred_Energies"] = pred_Energies
        log_dict["gt_Energies"] = gt_Energies
        log_dict["inital_Energies"] = np.array(initial_predictions)
        log_dict["n_edges"] = n_edges
        log_dict["n_nodes"] = n_nodes
        print("AR is", np.mean(pred_Energies/gt_Energies))
        return log_dict, RNN_key

    def make_forward_pass_until_finished(self, orig_SpinEnv, Hb_Nb_actions, params, RNN_key):
        import itertools
        loader_copy, loader_copy_copy = itertools.tee(orig_SpinEnv.loader)
        delattr(orig_SpinEnv, "loader")
        SpinEnv = copy.deepcopy(orig_SpinEnv)
        orig_SpinEnv.loader = loader_copy
        SpinEnv.loader = loader_copy_copy

        #print("copy SpinEnv", SpinEnv.env_step , SpinEnv.gt_Energy)
        while(True):

            done, Energy, _, H_graph_dict = SpinEnv.step(Hb_Nb_actions)

            if (np.any(done)):
                orig_graph_dict = SpinEnv.orig_graph_dict
                return orig_graph_dict["pred_Energy"], orig_graph_dict["gt_Energy"], orig_graph_dict["num_edges"], orig_graph_dict["num_nodes"], RNN_key
                        # print("pred", orig_graph_dict[k]["pred_Energy"])
                        # print("gt", orig_graph_dict[k]["gt_Energy"])
                        # print(H_graph_dict[idx]["finished_Energies"], "vs ", H_graph_dict[idx]["gt_Energy"])



            val_Hb_graphs = H_graph_dict["H_graph"]["jgraph"]
            val_Hb_Nb_ext_fields_list = [H_graph_dict["H_graph"]["Nb_ext_fields"]]

            concatenated_Nb_external_fields = np.concatenate(val_Hb_Nb_ext_fields_list, axis=1)
            padded_batched_H_graph, padded_concatenated_Nb_external_fields = jutils.pad_graph_and_ext_fields_to_nearest_power_of_k(val_Hb_graphs,concatenated_Nb_external_fields,
                                                                            k=self.graph_padding_factor)

            padded_val_Hb_compl_graphs = padded_batched_H_graph

            padded_batched_H_graph = jutils.cast_Tuple_to_float32(padded_batched_H_graph, np_=gpu_np)

            padded_concatenated_Nb_external_fields = jnp.array(padded_concatenated_Nb_external_fields)

            value, log_probs, sampled_bin_values, log_probs_classes, logits, RNN_key = self.jitted_vmapped_forward_env(params, padded_batched_H_graph, padded_val_Hb_compl_graphs,padded_concatenated_Nb_external_fields ,
                                                                                                  RNN_key)

            sampled_bin_values = cpu_np.reshape(sampled_bin_values, (self.Nb, self.n_test_graphs, 1))
            Hb_Nb_actions = np.swapaxes(sampled_bin_values, 0, 1)



    def make_env_steps_validation(self, params, RNN_key, log_dict):
        print("start evaluation on val set")
        self.make_evaluation_steps(params, RNN_key, self.ValSpinEnv, log_dict, mode = "val")

        return log_dict, RNN_key

    def make_env_steps_test(self, params, RNN_key, log_dict, sampling_mode = "beam_search"):
        print("start evaluation on test set")
        if(sampling_mode != "CE"):
            self.make_evaluation_steps(params, RNN_key, self.TestSpinEnv, log_dict, mode="test",
                                       sampling_mode=sampling_mode)

        else:
            self.cfg["Train_params"]["H_batch_size"] = 1
            self.n_test_graphs = 1
            self.graph_padding_factor = 2
            TestSpinEnv = IGraphEnv_eval(self.cfg, 0, mode="test")
            self.CE_env_steps(params, RNN_key, TestSpinEnv, log_dict)

        return log_dict, RNN_key


def return_batched_graph(H_graph_dict):
    return [ H_graph["batched_H_graph"] for H_graph in H_graph_dict]


def return_unbatched_graph(H_graph_dict):
    return [H_graph["unbatched_H_graph"] for H_graph in H_graph_dict]

def myfunc(cfg,H_seed, mode = "train"):
    print("init", H_seed)
    if(mode == "train"):
        return lambda: IGraphEnv_train(cfg,  H_seed, mode = mode)
    else:
        return lambda: IGraphEnv_eval(cfg,  H_seed, mode = mode)


def swap_axes_of_arrays(Sb_Hb_Nb_list_of_arrays):
    Nb_Sb_Hb_list_of_arrays = (np.swapaxes(np.swapaxes(arr, 1,2), 0,1) for arr in Sb_Hb_Nb_list_of_arrays)
    return Nb_Sb_Hb_list_of_arrays

def get_args_0_and_args_1(n_sampled_spins,prev_set_spins, spin_idx):
    n_sampled_spins = n_sampled_spins
    n_classes = 2**n_sampled_spins

    sampled_class = np.arange(0, n_classes)
    bin_arr = np.unpackbits(sampled_class.reshape((-1, 1)).astype(np.uint8), axis=1, count=n_sampled_spins,
                            bitorder="little")

    condition_0 = [bin_arr[:,idx] == value for idx, value in enumerate(prev_set_spins) ]
    condition_0.append(bin_arr[:,spin_idx] == 0 )
    condition_0 = np.array(condition_0)

    args_0 = np.argwhere(np.all(condition_0, axis = 0) )


    condition_1 = [bin_arr[:,idx] == value for idx, value in enumerate(prev_set_spins) ]
    condition_1.append(bin_arr[:,spin_idx] == 1 )
    condition_1 = np.array(condition_1)
    args_1 = np.argwhere(np.all(condition_1, axis = 0)  )
    return args_0, args_1

def mask_out_padded_probs(n_sampled_spins, spins_to_be_set):
    n_sampled_spins = n_sampled_spins
    n_classes = 2**n_sampled_spins

    sampled_class = np.arange(0, n_classes)
    bin_arr = np.unpackbits(sampled_class.reshape((-1, 1)).astype(np.uint8), axis=1, count=n_sampled_spins,
                            bitorder="little")

    condition_0 = [bin_arr[:,n_sampled_spins - idx - 1] == value for idx, value in enumerate(spins_to_be_set) ]
    condition_0 = np.array(condition_0)

    args_0 = np.argwhere(np.all(condition_0, axis = 0) )

    return args_0

if(__name__ == "__main__"):
    ### TODO inplement this to BFS
    n = 5
    logits = np.arange(0, 2**n)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits/(np.sum(exp_logits))

    args_0 = mask_out_padded_probs(n, [1, 0])

    filtered_logits = probs[args_0]
    filtered_prob_logits = np.exp(filtered_logits - np.max(filtered_logits))
    filtered_probs = filtered_prob_logits/np.sum(filtered_prob_logits)

    print(filtered_probs)
    print(np.sum(filtered_probs))
    pass

    n_sampled_spins = 5
    n_classes = 2**5

    get_args_0_and_args_1(5,[1,1], 2)
    pass