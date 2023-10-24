import jax
import jax.numpy as jnp
import jax.lax
import optax

from Networks.AutoregressiveModels.GNN_PPO_sample_configuration import GNN_PPO_SpinDrop as GNN_sample_configuration
from Networks.AutoregressiveModels.GNN_PPO_spin_drop_vectorised import GNN_PPO_SpinDrop
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
        self.schedule = cfg.Train_params.lr_schedule
        self.initialize_networks()

    def initialize_networks(self):
        print("init ", self.network_type)
        if(self.network_type == "GNN_sample_configuration"):
            print("GNN that samples spin configuration is used")
            net = GNN_sample_configuration
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

        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(), optax.scale(-self.lr))
        self.curr_lr = self.lr
        self.opt_state = self.optimizer.init(self.params)

        self.init_backward_sparse_reduced_Hb_Nb_loss()
        self.init_vmapped_eval()

        jitting = True
        update_params = lambda par, grad, opt_state: self.update_params(par, grad, opt_state)

        if (jitting):
            update_params = jax.jit(update_params)
        self.update_params_jit = update_params


    def init_params(self):

        self.key, subkey = jax.random.split(self.key)

        H_graph = self.GraphDataLoader.H_graph
        EnergyFunction = self.cfg["Ising_params"]["EnergyFunction"]
        pruning = self.cfg["Train_params"]["pruning"]
        if("MaxCl_compl" in EnergyFunction or not pruning):
            external_fields = np.zeros((H_graph.nodes.shape[0], 2))
        else:
            external_fields = np.zeros((H_graph.nodes.shape[0], 1))
        actions = jnp.ones((self.n_nodes,1), dtype = jnp.int32)

        self.params = self.initEvalSpinGRNN.init({"params": subkey}, H_graph, H_graph, external_fields, actions, None, None)
        print("params dtype", jax.tree_util.tree_map(lambda x: x.dtype, self.params))

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

    def sparse_reduced_Nb_Hb_loss(self, params, Sb_Hb_H_graph, minib_masks, Nb_ext_fields, c1 = 0.5):

        Nb_actions = jnp.swapaxes(Sb_Hb_H_graph.globals[:-1,:,0], 0,1)
        Nb_A_k = jnp.swapaxes(Sb_Hb_H_graph.globals[:-1,:,1], 0,1)
        Nb_log_probs = jnp.swapaxes(Sb_Hb_H_graph.globals[:-1,:,2], 0,1)
        Nb_rtg = jnp.swapaxes(Sb_Hb_H_graph.globals[:-1,:,3], 0,1)

        Nb_minibatch_actions = jnp.expand_dims(Nb_actions, axis = -1)
        Nb_minibatch_A_k = Nb_A_k
        Nb_minibatch_log_probs = Nb_log_probs
        Nb_minibatch_rtg = jnp.expand_dims(Nb_rtg, axis = -1)
        minib_masks = np.swapaxes(minib_masks, 0, 1)

        Nb_minibatch_Values, curr_Nb_minibatch_log_probs = self.vmapped_eval_apply(params, Sb_Hb_H_graph, minib_masks, Nb_ext_fields, Nb_minibatch_actions)

        # print("old probs")
        # print(Nb_log_probs)
        # print("new probs")
        # print(curr_Nb_minibatch_log_probs)

        ratios = jnp.exp(curr_Nb_minibatch_log_probs - Nb_minibatch_log_probs)

        surr1 = ratios * Nb_minibatch_A_k
        surr2 = jnp.clip(ratios, 1 - self.clip, 1 + self.clip) * Nb_minibatch_A_k

        print("ratios", ratios.shape)
        print("adv", surr1.shape, surr2.shape)
        print("value loss", Nb_minibatch_Values.shape, Nb_minibatch_rtg.shape)
        actor_loss = jnp.mean((-jnp.minimum(surr1, surr2)))
        critic_loss = jnp.mean((Nb_minibatch_Values - Nb_minibatch_rtg) ** 2)
        overall_loss = (1-c1) * actor_loss + c1 *critic_loss

        return overall_loss, (actor_loss, critic_loss)

    def init_vmapped_eval(self):
        self.vmapped_eval_apply = jax.jit(jax.vmap(self.eval_spin, in_axes=(None, None, 0, 0, 0), out_axes=(0,0)))

    def vmapped_eval(self):
        pass

    def init_backward_sparse_reduced_Hb_Nb_loss(self):
        self.sparse_reduced_Nb_Hb_backward_func = jax.jit(jax.value_and_grad(self.sparse_reduced_Nb_Hb_loss, argnums=0, has_aux=True))


    @partial(jax.jit, static_argnames=['self'])
    def backward_sparse_reduced_Nb_Hb_loss(self, params, opt_state, Sb_minibatch_H_graph, minib_masks, Nb_ext_fields):
        (overall_loss, (actor_loss, critic_loss)), grad = self.sparse_reduced_Nb_Hb_backward_func(params, Sb_minibatch_H_graph, minib_masks, Nb_ext_fields)
        params, opt_state = self.update_params_jit(params, grad, opt_state)
        return params, opt_state, (overall_loss, (actor_loss, critic_loss))


    def sample(self, params, key, H_graph, Ext_fields, ones, T):  ## dimension batch, time, features
        return self.SampleGRNN.apply(params, H_graph, ones, key, T)

    def sparse_eval_spin(self,  params, H_graph,  binaries, observations, idx):
        return self.EvalSpinGRNN.apply(params, H_graph, binaries, observations, idx)

    def eval_spin(self,  params, H_graph, compl_graph,ext_fields, actions):
        return self.EvalSpinGRNN.apply(params, H_graph, compl_graph, ext_fields,actions,None, None)



if __name__ == "__main__":
    pass