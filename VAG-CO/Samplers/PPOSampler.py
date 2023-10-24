import jax
import jax.numpy as jnp
import jax.lax
import optax

from Networks.AutoregressiveModels.GNN_PPO_spin_selector import GNN_PPO_SpinSelector
from Networks.AutoregressiveModels.GNN_PPO_spin_drop import GNN_PPO_SpinDrop
from Samplers.BaseSamplers.BaseSampler import BaseRNNSampler
from matplotlib import pyplot as plt
import numpy as np
from functools import partial

# from jax.config import config
# config.update('jax_disable_jit', True)

### TODO pay attention this currently does not work with external fields
class PPOSampler(BaseRNNSampler):

    def __init__(self, GraphDataLoader, epoch_dict, cfg, sparse_graphs = True):

        self.sparse_graphs = sparse_graphs

        self.clip = cfg.Train_params.PPO.clip_value
        self.GraphDataLoader = GraphDataLoader
        self.n_nodes = self.GraphDataLoader.n_nodes
        self.x = GraphDataLoader.x
        self.y = GraphDataLoader.y

        self.n_basis_states = cfg.Train_params.n_basis_states
        self.num_classes = 2

        self.init_key = jax.random.PRNGKey(cfg.Train_params.seed)
        self.key, self.dropout_key = jax.random.split(self.init_key, num=2)

        self.cfg = cfg

        self.lr = cfg.Train_params.lr
        self.lr_alpha = cfg.Train_params.lr_alpha

        self.N_warmup = epoch_dict["N_warmup"]*epoch_dict["batch_epochs"]
        self.N_anneal = epoch_dict["N_anneal"]*epoch_dict["batch_epochs"]
        self.N_equil = epoch_dict["N_equil"]*epoch_dict["batch_epochs"]
        self.epochs = self.N_warmup + self.N_anneal + self.N_equil
        print("epoch_dict", epoch_dict)

        self.network_type = cfg.Network_params.network_type

        ### INITIALIZE PROB AND PHASE NETWORK
        self.back_transform = jnp.array(self.GraphDataLoader.back_transform)
        self.spiral_transform = jnp.array(self.GraphDataLoader.spiral_transform)
        print("back_transform", self.back_transform)
        print("spiral_transform", self.spiral_transform)
        self.schedule = cfg.Train_params.lr_schedule
        self.initialize_networks()

    def initialize_networks(self):
        print("init ", self.network_type)
        if(self.network_type == "GNN_SpinSelector"):
            print("GNN is used")
            net = GNN_PPO_SpinSelector
        elif (self.network_type == "GNN_SpinDrop"):
            print("GNN is used")
            net = GNN_PPO_SpinDrop
        else:
            ValueError(f"Network type {self.network_type} is not valid.")

        if(self.sparse_graphs):
            self.SampleGRNN = net(self.spiral_transform, self.cfg, SampleMode = "sample_sparse")
            self.EvalSpinGRNN = net(self.spiral_transform, self.cfg, SampleMode = "eval_spin_sparse")
            self.SampleUnpaddedGRNN = net(self.spiral_transform, self.cfg, SampleMode = "sample_unpadded")
        else:
            self.SampleGRNN = net(self.spiral_transform, self.cfg, SampleMode = "sample")
            self.EvalSpinGRNN = net(self.spiral_transform, self.cfg, SampleMode = "eval_spin")
        self.initEvalSpinGRNN = net(self.spiral_transform, self.cfg, SampleMode="eval_spin")

        self.init_params()

        print("Networks are initialized")

        ### INITIALIZE OPTAX
        print("learning rate scheduler, lr , lr_alpha", self.lr, self.lr_alpha, self.schedule, self.epochs)

        if(self.schedule == "cosine_warmup"):
            warmup_steps = 500
            self.lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=0., peak_value=10 * self.lr,
                                                             end_value=self.lr_alpha * self.lr,
                                                             warmup_steps=warmup_steps,
                                                             decay_steps=self.epochs - warmup_steps)
        elif(self.schedule ==  "cosine"):
            self.lr_scheduler = optax.cosine_decay_schedule(self.lr, self.epochs, alpha=self.lr_alpha)
        elif(self.schedule ==  "cosine_restart"):
            min_lr = self.lr * self.lr_alpha
            mean_lr = (self.lr + min_lr) / 2
            n = 2
            peak_factor = 2
            if(self.N_warmup > 0):
                # for freq in range(n)
                lrs = [self.lr,self.lr/n]
                lrs_end = [self.lr/n, self.lr]
                warmup_cosine_decay_scheduler_list = [optax.warmup_cosine_decay_schedule(init_value=lrs[i], peak_value=peak_factor * self.lr,
                                                   warmup_steps=int(0.2 * self.N_warmup / n),
                                                   decay_steps=int(self.N_warmup / n),
                                                   end_value=lrs_end[i]) for i in range(n)]
                boundaries = [int(self.N_warmup / n), 2*int(self.N_warmup / n)]
            else:
                boundaries = []
                warmup_cosine_decay_scheduler_list = []

            if(self.N_anneal > 0):
                freq_width = int(0.1 * self.N_anneal)
                n_cycles = int(self.N_anneal / freq_width)
                print("cosine with restarts is used with ", n_cycles, " frequencies")
                cos_anneal = lambda x: (self.lr - mean_lr) * np.cos(np.pi * x / n_cycles) + mean_lr
                lrs = [cos_anneal(0)]
                lrs.extend([ cos_anneal(freq)/n for freq in range(n_cycles)])
                warmup_cosine_decay_scheduler_list.extend([optax.warmup_cosine_decay_schedule(init_value=lrs[freq], peak_value= peak_factor*cos_anneal(freq),
                                                                                   warmup_steps=int(0.2*freq_width),
                                                                                   decay_steps=freq_width,
                                                                                   end_value=lrs[freq + 1]) for freq in range(n_cycles)])
                boundaries.extend([self.N_warmup + freq_width * (i + 1) for i in range(len(warmup_cosine_decay_scheduler_list) - 1 - len(boundaries))])
            if(self.N_equil > 0):
                freq_width = int(0.1 * self.N_equil)
                n_cycles = int(self.N_equil / freq_width)
                print("cosine with restarts is used with ", n_cycles, " frequencies")
                cos_anneal = lambda x: min_lr
                lrs = [cos_anneal(0)]
                lrs.extend([ cos_anneal(freq)/n for freq in range(n_cycles)])
                warmup_cosine_decay_scheduler_list.extend([optax.warmup_cosine_decay_schedule(init_value=lrs[freq], peak_value= peak_factor*cos_anneal(freq),
                                                                                   warmup_steps=int(0.2*freq_width),
                                                                                   decay_steps=freq_width,
                                                                                   end_value=lrs[freq + 1]) for freq in range(n_cycles)])
                boundaries.extend([self.N_warmup + self.N_anneal + freq_width * (i + 1) for i in range(len(warmup_cosine_decay_scheduler_list) - 1 - len(boundaries))])

            self.lr_scheduler = optax.join_schedules(warmup_cosine_decay_scheduler_list, boundaries=boundaries)

            if(False):
                lrs = [self.lr_scheduler(i) for i in range(self.epochs)]
                print("finished")
                #print(lrs)
                plt.figure()
                plt.scatter(range(self.epochs), lrs)
                plt.title("SGD With Warm Restarts Scheduler")
                plt.ylabel("Learning Rate")
                plt.xlabel("Epochs/Steps")
                plt.show()

        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(), optax.scale(-self.lr))
        self.opt_state = self.optimizer.init(self.params)

        # opt_init, self.opt_update = optax.radam(learning_rate=self.lr_scheduler)
        # self.opt_state = opt_init(self.params)

        self.init_Nb_sample()
        #self.init_Nb_eval()
        self.init_Hb_Nb_sample()
        #self.init_Hb_Nb_eval()
        #self.init_Nb_eval_spin()
        #self.init_Hb_Nb_eval_spin()

        self.init_idxb_eval()
        self.init_Nb_idxb_eval()
        self.init_Hb_Nb_idxb_eval()
        self.init_backward_Hb_Nb_loss()
        self.init_Nb_sample_sparse()
        self.init_Nb_sample_reduce()
        #self.init_Nb_sparse_eval()
        self.init_idxb_sparse_eval()
        self.init_Nb_idxb_sparse_eval()
        self.init_backward_sparse_reduced_Hb_Nb_loss()
        self.init_backward_sparse_Hb_Nb_loss()

        jitting = True
        update_params = lambda par, grad, opt_state: self.update_params(par, grad, opt_state)

        if (jitting):
            update_params = jax.jit(update_params)
        self.update_params_jit = update_params


    def init_params(self):
        self.key, subkey = jax.random.split(self.key)
        H_graph = self.GraphDataLoader.H_graph
        actions = jnp.ones((self.n_nodes,2), dtype = jnp.int32)
        self.params = self.initEvalSpinGRNN.init({"params": subkey}, H_graph, actions, None, None)

    def return_batched_key(self, batch_size):
        subkeys = jax.random.split(self.key, batch_size)
        return subkeys

    ### TODO add this to BaseSampler
    def get_params(self):
        return self.params

    def get_opt_states(self):
        return self.opt_state

    def save_opt_state(self, opt_state):
        self.opt_state = opt_state

    ### TODO add this to BaseSampler
    def update_optimizer(self, lr, params):
        self.curr_lr = lr
        #self.optimizer = optax.radam(learning_rate=self.curr_lr)
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(),
                                              optax.scale(-self.curr_lr))
        opt_state = self.optimizer.init(params)
        return opt_state

    ### TODO add this to BaseSampler
    def update_params(self, params, grads, opt_state):
        grad_update, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, grad_update)
        return params, opt_state

    def Hb_Nb_loss(self, params, Hb_Nb_binaries, Hb_Nb_observations, Hb_H_graph, Hb_Nb_idx, Hb_Nb_A_k, Hb_Nb_log_probs, Hb_Nb_rtg, c1 = 0.5):

        Hb_Nb_Values, curr_Hb_Nb_log_probs = self.Hb_Nb_idxb_eval(params, Hb_H_graph, Hb_Nb_binaries, Hb_Nb_observations, Hb_Nb_idx)
        ratios = jnp.exp(curr_Hb_Nb_log_probs - Hb_Nb_log_probs)

        surr1 = ratios * Hb_Nb_A_k
        surr2 = jnp.clip(ratios, 1 - self.clip, 1 + self.clip) * Hb_Nb_A_k

        actor_loss = jnp.mean((-jnp.minimum(surr1, surr2)))
        critic_loss = jnp.mean((Hb_Nb_Values - Hb_Nb_rtg) ** 2)
        overall_loss = (1-c1) * actor_loss + c1 *critic_loss
        return overall_loss, (actor_loss, critic_loss)


    def sparse_Nb_Hb_loss(self, params, Nb_Hb_binaries, Nb_Hb_observations, Hb_H_graph, Nb_Hb_idxb, Nb_Hb_A_k, Nb_Hb_log_probs, Nb_Hb_rtg, c1 = 0.5):
        Nb_idxb_Hb = jnp.swapaxes(Nb_Hb_idxb, 1 ,2)
        Nb_idxb_Hb_Values, curr_Nb_idxb_Hb_log_probs = self.Nb_idxb_sparse_eval(params, Hb_H_graph, Nb_Hb_binaries, Nb_Hb_observations, Nb_idxb_Hb)

        curr_Nb_Hb_idxb_log_probs = jnp.swapaxes(curr_Nb_idxb_Hb_log_probs, 1, 2)

        Nb_Hb_idxb_Values = jnp.swapaxes(Nb_idxb_Hb_Values, 1, 2)

        ratios = jnp.exp(curr_Nb_Hb_idxb_log_probs - Nb_Hb_log_probs)

        surr1 = ratios * Nb_Hb_A_k
        surr2 = jnp.clip(ratios, 1 - self.clip, 1 + self.clip) * Nb_Hb_A_k

        actor_loss = jnp.mean((-jnp.minimum(surr1, surr2)))
        critic_loss = jnp.mean((Nb_Hb_idxb_Values - Nb_Hb_rtg) ** 2)
        overall_loss = (1-c1) * actor_loss + c1 *critic_loss

        return overall_loss, (actor_loss, critic_loss)

    def sparse_reduced_Nb_Hb_loss_minibatched(self, params, Sb_minibatch_H_graph, Sb_minibatch_actions, Sb_minibatch_A_k, Sb_minibatch_log_probs, Sb_minibatch_rtg, c1 = 0.5):

        Sb_minibatch_actions = jnp.reshape(Sb_minibatch_actions, (Sb_minibatch_actions.shape[0]*Sb_minibatch_actions.shape[1],2))
        Sb_minibatch_A_k = jnp.reshape(Sb_minibatch_A_k, (Sb_minibatch_A_k.shape[0]*Sb_minibatch_A_k.shape[1]))
        Sb_minibatch_log_probs = jnp.reshape(Sb_minibatch_log_probs, (Sb_minibatch_log_probs.shape[0]*Sb_minibatch_log_probs.shape[1]))
        print("before reshaping",Sb_minibatch_rtg.shape)
        Sb_minibatch_rtg = jnp.reshape(Sb_minibatch_rtg, (Sb_minibatch_rtg.shape[0]*Sb_minibatch_rtg.shape[1],1))

        Sb_minibatch_Values, curr_Sb_minibatch_log_probs = self.EvalSpinGRNN.apply(params, Sb_minibatch_H_graph, Sb_minibatch_actions, None, None)

        print("log", curr_Sb_minibatch_log_probs.shape, Sb_minibatch_log_probs.shape)
        ratios = jnp.exp(curr_Sb_minibatch_log_probs - Sb_minibatch_log_probs)

        surr1 = ratios * Sb_minibatch_A_k
        surr2 = jnp.clip(ratios, 1 - self.clip, 1 + self.clip) * Sb_minibatch_A_k

        actor_loss = jnp.mean((-jnp.minimum(surr1, surr2)))
        print("rewards to go", Sb_minibatch_Values.shape, Sb_minibatch_rtg.shape)
        print("Ak", Sb_minibatch_A_k.shape )
        critic_loss = jnp.mean((Sb_minibatch_Values - Sb_minibatch_rtg) ** 2)
        print((Sb_minibatch_Values - Sb_minibatch_rtg).shape, (-jnp.minimum(surr1, surr2)).shape, "losses")
        overall_loss = (1-c1) * actor_loss + c1 *critic_loss

        return overall_loss, (actor_loss, critic_loss)

    def sparse_reduced_Nb_Hb_loss(self, params, Sb_minibatch_H_graph, Sb_minibatch_actions, Sb_minibatch_A_k, Sb_minibatch_log_probs, Sb_minibatch_rtg, c1 = 0.5):

        Sb_minibatch_actions = jnp.reshape(Sb_minibatch_actions, (Sb_minibatch_actions.shape[0]*Sb_minibatch_actions.shape[1]*Sb_minibatch_actions.shape[2],2))
        Sb_minibatch_A_k = jnp.reshape(Sb_minibatch_A_k, (Sb_minibatch_A_k.shape[0]*Sb_minibatch_A_k.shape[1]*Sb_minibatch_A_k.shape[2]))
        Sb_minibatch_log_probs = jnp.reshape(Sb_minibatch_log_probs, (Sb_minibatch_log_probs.shape[0]*Sb_minibatch_log_probs.shape[1]*Sb_minibatch_log_probs.shape[2]))
        print("before reshaping",Sb_minibatch_rtg.shape)
        Sb_minibatch_rtg = jnp.reshape(Sb_minibatch_rtg, (Sb_minibatch_rtg.shape[0]*Sb_minibatch_rtg.shape[1]*Sb_minibatch_rtg.shape[2],1))

        Sb_minibatch_Values, curr_Sb_minibatch_log_probs = self.EvalSpinGRNN.apply(params, Sb_minibatch_H_graph, Sb_minibatch_actions, None, None)

        print("log", curr_Sb_minibatch_log_probs.shape, Sb_minibatch_log_probs.shape)
        ratios = jnp.exp(curr_Sb_minibatch_log_probs - Sb_minibatch_log_probs)

        surr1 = ratios * Sb_minibatch_A_k
        surr2 = jnp.clip(ratios, 1 - self.clip, 1 + self.clip) * Sb_minibatch_A_k

        actor_loss = jnp.mean((-jnp.minimum(surr1, surr2)))
        print("rewards to go", Sb_minibatch_Values.shape, Sb_minibatch_rtg.shape)
        print("Ak", Sb_minibatch_A_k.shape )
        critic_loss = jnp.mean((Sb_minibatch_Values - Sb_minibatch_rtg) ** 2)
        print((Sb_minibatch_Values - Sb_minibatch_rtg).shape, (-jnp.minimum(surr1, surr2)).shape, "losses")
        overall_loss = (1-c1) * actor_loss + c1 *critic_loss

        return overall_loss, (actor_loss, critic_loss)


    def init_backward_Hb_Nb_loss(self):
        self.Hb_Nb_backward_func = jax.jit(jax.value_and_grad(self.Hb_Nb_loss, argnums = 0, has_aux=True))

    def backward_Hb_Nb_loss(self, params, opt_state, Hb_Nb_binaries, Hb_Nb_observations, Hb_H_graph, Hb_Nb_indeces, Hb_Nb_A_k, Hb_Nb_log_probs, Hb_Nb_rtg):
        (overall_loss, (actor_loss, critic_loss)), grad = self.Hb_Nb_backward_func(params, Hb_Nb_binaries, Hb_Nb_observations, Hb_H_graph, Hb_Nb_indeces, Hb_Nb_A_k, Hb_Nb_log_probs, Hb_Nb_rtg)
        params, opt_state = self.update_params_jit(params, grad, opt_state)
        return params, opt_state, (overall_loss, (actor_loss, critic_loss))

    def init_backward_sparse_Hb_Nb_loss(self):
        self.sparse_Nb_Hb_backward_func = jax.jit(jax.value_and_grad(self.sparse_Nb_Hb_loss, argnums = 0, has_aux=True))

    def init_backward_sparse_reduced_Hb_Nb_loss(self):
        self.sparse_reduced_Nb_Hb_backward_func = jax.jit(jax.value_and_grad(self.sparse_reduced_Nb_Hb_loss, argnums=0, has_aux=True))

    @partial(jax.jit, static_argnames=['self'])
    def backward_sparse_Nb_Hb_loss(self, params, opt_state, Hb_Nb_binaries, Hb_Nb_observations, Hb_H_graph, Hb_Nb_indeces, Hb_Nb_A_k, Hb_Nb_log_probs, Hb_Nb_rtg):
        (overall_loss, (actor_loss, critic_loss)), grad = self.sparse_Nb_Hb_backward_func(params, Hb_Nb_binaries, Hb_Nb_observations, Hb_H_graph, Hb_Nb_indeces, Hb_Nb_A_k, Hb_Nb_log_probs, Hb_Nb_rtg)
        params, opt_state = self.update_params_jit(params, grad, opt_state)
        return params, opt_state, (overall_loss, (actor_loss, critic_loss))

    @partial(jax.jit, static_argnames=['self'])
    def backward_sparse_reduced_Nb_Hb_loss(self, params, opt_state, Sb_minibatch_H_graph, Sb_minibatch_actions, Sb_minibatch_A_k, Sb_minibatch_log_probs, Sb_minibatch_rtg):
        (overall_loss, (actor_loss, critic_loss)), grad = self.sparse_reduced_Nb_Hb_backward_func(params, Sb_minibatch_H_graph, Sb_minibatch_actions, Sb_minibatch_A_k, Sb_minibatch_log_probs, Sb_minibatch_rtg)
        params, opt_state = self.update_params_jit(params, grad, opt_state)
        return params, opt_state, (overall_loss, (actor_loss, critic_loss))

    def init_Nb_sample_sparse(self):
        self.Nb_sample_sparse_Hb_func = jax.jit(jax.vmap(self.sample, in_axes = (None, 0, None, 0 , None), out_axes = (0,0,0,0,0,0,0,0,0)))

    @partial(jax.jit, static_argnames=['self'])
    def Nb_sample_sparse_Hb(self, params, H_graph, key, T):
        one_vec = jnp.ones((self.n_basis_states, H_graph.n_node.shape[0]-1, self.n_nodes))
        subkeys = jax.random.split(key, self.n_basis_states)
        Nb_values, Nb_log_probs, Nb_binaries, Nb_observations, Nb_Energy, Nb_rewards, Nb_value_target, Nb_A_k, subkeys = self.Nb_sample_sparse_Hb_func(params, subkeys, H_graph, one_vec, T)
        unbatched_key, __ = jax.random.split(subkeys[-1])
        return Nb_values, Nb_log_probs, Nb_binaries, Nb_observations, Nb_Energy, Nb_rewards, Nb_value_target, Nb_A_k, unbatched_key

    @partial(jax.jit, static_argnames=['self'])
    def Nb_sample_sparse_Hb_test(self, params, H_graph, key, T = 0.):
        one_vec = jnp.ones((self.n_basis_states, H_graph.n_node.shape[0]-1, self.n_nodes))
        subkeys = jax.random.split(key, self.n_basis_states)
        Nb_values, Nb_log_probs, Nb_binaries, Nb_observations, Nb_Energy, Nb_rewards, Nb_value_target, Nb_A_k, subkeys = self.Nb_sample_sparse_Hb_func(params, subkeys, H_graph, one_vec, T)
        unbatched_key, __ = jax.random.split(subkeys[-1])
        return {"spin_log_probs": Nb_log_probs, "bins": Nb_binaries, "Energy": Nb_Energy, "key": unbatched_key}

    def sample(self, params, key, H_graph, ones, T):  ## dimension batch, time, features
        return self.SampleGRNN.apply(params, H_graph, ones, key, T)

    def init_Nb_sample_reduce(self):
        self.Nb_sample_reduce = jax.vmap(self.sample, in_axes= (None, 0, None, 0, None), out_axes= (0,0,0,0,0,0,0,0,0,0))

    def Nb_sample_reduce_H_sparse(self, params, H_graph, key, T):
        one_vec = jnp.ones((self.n_basis_states, H_graph.nodes.shape[0]))
        subkeys = jax.random.split(key, self.n_basis_states)
        Nb_values, Nb_log_probs, Nb_binaries, Nb_observations, Nb_actions, Nb_Energy, Nb_rewards, Nb_value_target, Nb_A_k, subkeys = self.Nb_sample_reduce(params, subkeys, H_graph, one_vec, T)
        unbatched_key, __ = jax.random.split(subkeys[-1])
        return Nb_values, Nb_log_probs, Nb_binaries, Nb_observations, Nb_Energy, Nb_rewards, Nb_value_target, Nb_A_k, unbatched_key

    def sparse_eval_spin(self,  params, H_graph,  binaries, observations, idx):
        return self.EvalSpinGRNN.apply(params, H_graph, binaries, observations, idx)

    def init_idxb_sparse_eval(self):
        self.idxb_sparse_eval = jax.jit(jax.vmap(self.sparse_eval_spin , in_axes=(None, None, None, None,0), out_axes=(0,0)))

    def init_Nb_idxb_sparse_eval(self):
        self.Nb_idxb_sparse_eval = jax.jit(jax.vmap(self.idxb_sparse_eval , in_axes=(None, None, 0, 0,0), out_axes=(0,0)))

    # def init_Nb_sparse_eval(self):
    #     self.Nb_sparse_eval = jax.jit(jax.vmap(self.sparse_eval_spin , in_axes=(None, None, 0, 0,0), out_axes=(0,0)))

    def eval_spin(self,  params, H_graph,  binaries, observations, idx):
        return self.EvalSpinGRNN.apply(params, H_graph, binaries, observations, idx)

    def init_idxb_eval(self):
        self.idxb_eval = jax.jit(jax.vmap(self.eval_spin , in_axes=(None, None, None, None,0), out_axes=(0,0)))

    def init_Nb_idxb_eval(self):
        self.Nb_idxb_eval = jax.jit(jax.vmap(self.idxb_eval , in_axes=(None, None, 0, 0,0), out_axes=(0,0)))

    def init_Hb_Nb_idxb_eval(self):
        self.Hb_Nb_idxb_eval = jax.jit(jax.vmap(self.idxb_stoquastic_evaluate_spins , in_axes=(None, 0, 0, 0,0), out_axes=(0,0)))

    def stoquastic_samples(self, params, H_graph, key, T):
        one_vec = jnp.ones((self.n_basis_states, self.n_nodes))
        subkeys = jax.random.split(key, self.n_basis_states)
        Nb_values, Nb_log_probs, Nb_binaries, Nb_observations, Nb_Energy, Nb_rewards, Nb_value_target, Nb_A_k, subkeys = self.Nb_sample(params, subkeys, H_graph, one_vec, T)
        unbatched_key, __ = jax.random.split(subkeys[-1])
        return Nb_values, Nb_log_probs, Nb_binaries, Nb_observations, Nb_Energy, Nb_rewards, Nb_value_target, Nb_A_k, unbatched_key

    def stoquastic_evaluate_spins(self, params, H_graph, Nb_binaries, Nb_observations, Nb_idx):
        Nb_values, Nb_log_probs = self.Nb_eval_spin(params, H_graph, Nb_binaries, Nb_observations, Nb_idx)
        return Nb_values, Nb_log_probs

    def idxb_stoquastic_evaluate_spins(self, params, H_graph, Nb_binaries, Nb_observations, Nb_idx):
        Nb_values, Nb_log_probs = self.Nb_idxb_eval(params, H_graph, Nb_binaries, Nb_observations, Nb_idx)
        return Nb_values, Nb_log_probs

    def stoquastic_evaluate_log_probs(self, params, H_graph, Nb_binaries):
        Nb_values, Nb_log_probs = self.Nb_eval(params, H_graph, Nb_binaries)
        return Nb_values, Nb_log_probs

    def init_Hb_Nb_sample(self):
        self.Hb_Nb_sample = jax.jit(jax.vmap(self.stoquastic_samples, in_axes=(None, 0, 0,  None), out_axes=(0,0,0,0,0,0,0,0,0)))

    def init_Nb_sample(self):
        sample = jax.jit(self.sample)
        self.Nb_sample = jax.jit(jax.vmap(sample, in_axes=(None, 0, None, 0, None), out_axes=(0,0,0, 0,0,0,0,0,0)))

    def init_Hb_Nb_eval(self):
        self.Hb_Nb_eval = jax.jit(jax.vmap(self.stoquastic_evaluate_log_probs , in_axes=(None, 0, 0), out_axes=(0,0)))

    def init_Nb_eval(self):
        evaluate = jax.jit(self.eval)
        self.Nb_eval = jax.jit(jax.vmap(evaluate , in_axes=(None, None, 0, 0), out_axes=(0,0)))



if __name__ == "__main__":
    pass