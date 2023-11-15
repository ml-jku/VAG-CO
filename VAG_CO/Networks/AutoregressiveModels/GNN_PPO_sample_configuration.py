from ..BuildingBlocks.EncodeProcessDecodeGNNs import EncodeProcess, EncodeProcessNew, EncodeProcess_node_embedding
from ..BuildingBlocks.GNNetworks import ProbMLP, ValueMLP
import flax.linen as nn
import jax.numpy as jnp
import jax
import jax.tree_util as tree
from jraph_utils import utils as jutils
from utils import softmax_utils

### TODO add observations
class GNN_PPO_SpinDrop(nn.Module):
    spiral_transform: list
    cfg: dict
    SampleMode: str = "sample"
    training: bool = False
    weight_tied: bool = False

    def setup(self):

        self.message_MLP_features = self.cfg.Network_params.GNNs.message_MLP_features
        self.node_MLP_features = self.cfg.Network_params.GNNs.node_MLP_features
        self.edge_MLP_features = self.cfg.Network_params.GNNs.edge_MLP_features
        self.encode_node_features = self.cfg.Network_params.GNNs.encode_node_features
        self.encode_edge_features = self.cfg.Network_params.GNNs.encode_edge_features
        self.n_GNN_layers = self.cfg.Network_params.GNNs.n_GNN_layers
        self.policy_MLP_features = self.cfg.Network_params.policy_MLP_features
        self.value_MLP_features = self.cfg.Network_params.value_MLP_features

        self.policy_global_features = self.cfg.Network_params.GNNs.policy_global_features
        self.value_global_features = self.cfg.Network_params.GNNs.value_global_features

        self.EnergyFunction = self.cfg.Ising_params.EnergyFunction

        self.layer_norm = self.cfg.Network_params.layer_norm
        self.edge_updates = self.cfg.Network_params.GNNs.edge_updates
        self.GNN_name = self.cfg.Network_params.GNNs.GNN_name

        self.lam = self.cfg.Train_params.PPO.lam
        self.GNN_mode = self.cfg.Network_params.GNNs.mode

        self.n_classes = self.policy_MLP_features[-1]
        self.n_sampled_sites = self.cfg.Ising_params.sampled_sites
        self.aranged_sampled_sites = jnp.array(jnp.arange(0, self.n_sampled_sites), dtype=jnp.int32)
        #RNN = lambda x: GRNN_inside_loop(SampleMode = self.SampleMode,out_features = self.out_features, hidden_RNN_features = self.hidden_RNN_features, n_RNN_layers = self.n_RNN_layers, training = self.training)
        self.cell = nn.scan(
            nn.jit(GNN_actor_critic),
            variable_broadcast="params",
            split_rngs={"params": False})

        self.apply_cell = self.cell(self.n_classes, self.n_sampled_sites, self.aranged_sampled_sites, self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features, self.encode_node_features, self.encode_edge_features, self.n_GNN_layers,
                                    self.policy_MLP_features, self.value_MLP_features, self.EnergyFunction, self.policy_global_features, self.value_global_features, GNN_name = self.GNN_name, GNN_mode = self.GNN_mode,
                                    layer_norm = self.layer_norm, edge_updates = self.edge_updates, SampleMode = self.SampleMode, training = self.training)

        self.gamma = 1.


    @nn.compact
    def __call__(self, H_graph, compl_H_graph, Ext_fields, ones, key, T):
        if (self.SampleMode == "sample"):
            return self.sample(H_graph, compl_H_graph, Ext_fields, key)
        elif(self.SampleMode == "sample_unpadded"):
            actions = ones
            return self.sample_unpadded(H_graph, compl_H_graph, Ext_fields, key)
        elif(self.SampleMode == "eval_spin"):
            actions = ones
            return self.eval_spin(H_graph, compl_H_graph, Ext_fields, actions)
        elif(self.SampleMode == "sample_sparse"):
            return self.sample(H_graph, compl_H_graph, Ext_fields, key)
        elif(self.SampleMode == "eval_spin_sparse"):
            actions = ones
            return self.eval_spin_padded(H_graph, compl_H_graph, Ext_fields, actions)#

    def sample(self, H_graphs, compl_H_graph, Ext_fields, key):
        return self.apply_cell.sample(H_graphs, compl_H_graph,Ext_fields, key)

    def sample_unpadded(self, H_graphs, compl_H_graph, Ext_fields, key):
        return self.apply_cell.sample_unpadded(H_graphs, compl_H_graph, Ext_fields, key)

    def eval_spin(self, H_graph, compl_H_graph, Ext_fields, actions):
        values, log_probs = self.apply_cell.eval(H_graph, compl_H_graph, Ext_fields, actions)

        return values, log_probs

    def eval_spin_padded(self, H_graph, compl_H_graph, Ext_fields, actions):
        values, log_probs = self.apply_cell.eval_padded(H_graph, compl_H_graph, Ext_fields, actions)

        return values, log_probs


class GNN_actor_critic(nn.Module):
    n_classes: int
    n_sampled_sites: int
    aranged_sampled_sites: list
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    encode_node_features: list
    encode_edge_features: list
    n_GNN_layers: int
    policy_MLP_features: list
    value_MLP_features: list

    EnergyFunction: str
    value_global_features: bool
    policy_global_features: bool

    layer_norm: bool = True
    edge_updates: bool = True
    SampleMode: str = "sample"
    training: bool = False
    GNN_mode: str = "non_linear"
    GNN_name: str = "newEncoder"

    def setup(self):
        #self.aranged_sampled_sites = jnp.array(self.aranged_sampled_sites, jnp.int32)
        #self.aranged_sampled_sites = jnp.array(jnp.arange(0, self.n_sampled_sites ), dtype=jnp.int32)
        # self.value_GNN = EncodeProcessDecode(self.GNN_MLP_features, n_layers = self.n_GNN_layers,
        #                                message_passing = self.GNN_mode, training = self.training, weight_tied = False)

        print("used GNN Mode", self.GNN_mode)
        print("used Encoder", self.GNN_name)
        if(self.GNN_name == "newEncoder_embedding"):
            self.prob_GNN = EncodeProcess_node_embedding(self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features, self.encode_node_features, self.encode_edge_features, layer_norm = self.layer_norm,
                                               edge_updates = self.edge_updates,  n_layers = self.n_GNN_layers,
                                               message_passing = self.GNN_mode, training = self.training, weight_tied = False)
        else:
            raise ValueError("Network Type is not implemented")


        self.probMLP = ProbMLP(features= self.policy_MLP_features, training = self.training)
        self.valueMLP = ValueMLP(features= self.value_MLP_features, training = self.training)

    def __call__(self, carry, xs):
        if(self.SampleMode == "sample"):
            return self.sample(carry, xs)
        elif (self.SampleMode == "sample_sparse"):
            return self.sample_batched_graphs(carry, xs)

    def forward_padded(self, H_graph, spin_sites):
        GNN_embeddings = self.prob_GNN(H_graph)
        sampled_spin_embeddings = GNN_embeddings[spin_sites]
        concat_node_embeddings = jnp.reshape(sampled_spin_embeddings, (H_graph.n_node.shape[0], self.n_sampled_sites* sampled_spin_embeddings.shape[-1]))

        if(self.value_global_features or self.policy_global_features):
            nodes = H_graph.nodes
            n_node = H_graph.n_node
            n_graph = n_node.shape[0]
            graph_idx = jnp.arange(n_graph)
            sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
            node_gr_idx = jnp.repeat(
                graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)

            print(jax.ops.segment_sum(GNN_embeddings, node_gr_idx, n_graph).shape)
            sum_embedding = jax.ops.segment_sum(GNN_embeddings, node_gr_idx, n_graph)
            print("node embedding shape", concat_node_embeddings.shape)
            print("Overall gnn-embeddings",GNN_embeddings.shape)
            print("sum embedding shape", sum_embedding.shape)
            concat_embedding = jnp.concatenate([sum_embedding, concat_node_embeddings], axis = -1)

        if(self.value_global_features):
            values = self.valueMLP(concat_embedding)
        else:
            values = self.valueMLP(concat_node_embeddings)

        if(self.policy_global_features):
            log_prob, logits = self.probMLP(concat_embedding)
        else:
            log_prob, logits = self.probMLP(concat_node_embeddings)

        return values, log_prob, logits

    def forward(self, H_graph, spin_sites):
        GNN_embeddings = self.prob_GNN(H_graph)
        sampled_spin_embeddings = GNN_embeddings[spin_sites]
        concat_node_embeddings = jnp.squeeze(jnp.reshape(sampled_spin_embeddings, (H_graph.n_node.shape[0], self.n_sampled_sites* sampled_spin_embeddings.shape[-1])), axis = -2)

        if(self.value_global_features or self.policy_global_features):
            nodes = H_graph.nodes
            n_node = H_graph.n_node
            n_graph = n_node.shape[0]
            graph_idx = jnp.arange(n_graph)
            sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
            node_gr_idx = jnp.repeat(
                graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)

            print(jax.ops.segment_sum(GNN_embeddings, node_gr_idx, n_graph).shape)
            sum_embedding = jnp.squeeze(jax.ops.segment_sum(GNN_embeddings, node_gr_idx, n_graph), axis = -2)
            print("node embedding shape", concat_node_embeddings.shape)
            print(GNN_embeddings.shape)
            print("sum embedding shape", sum_embedding.shape)
            concat_embedding = jnp.concatenate([sum_embedding, concat_node_embeddings], axis = -1)

        if(self.value_global_features):
            values = self.valueMLP(concat_embedding)
        else:
            values = self.valueMLP(concat_node_embeddings)

        if(self.policy_global_features):
            log_prob, logits  = self.probMLP(concat_embedding)
        else:
            log_prob, logits  = self.probMLP(concat_node_embeddings)

        return values, log_prob, logits

    def _make_H_graph_input(self, H_graph, Ext_fields):
        n_node = H_graph.n_node

        first_spin_indexes = jutils.get_first_node_idxs(n_node)

        ### TODO check this line of code
        spin_sites_index_mat = first_spin_indexes[:, jnp.newaxis] + self.aranged_sampled_sites[jnp.newaxis, :]  # jnp.cumsum(graph_idx * n_node)
        spin_sites_index = jnp.ravel(spin_sites_index_mat)
        spin_input_token = jnp.ravel(spin_sites_index_mat - first_spin_indexes[:, jnp.newaxis] + 1)

        rand_nodes = H_graph.nodes
        num_nodes = rand_nodes.shape[0]
        Spin_identifier = jnp.zeros(num_nodes)
        Spin_identifier = Spin_identifier.at[spin_sites_index].set(spin_input_token)

        one_hot_identifier = jax.nn.one_hot(Spin_identifier, num_classes=self.n_sampled_sites + 1)
        H_embedding = jnp.concatenate([Ext_fields, rand_nodes, one_hot_identifier], axis=-1)

        H_graph = H_graph._replace(nodes=H_embedding)
        return H_graph, spin_sites_index


    def sample_unpadded(self, H_graph, compl_H_graph, Ext_fields, key):
        H_graph, spin_sites = self._make_H_graph_input(H_graph, Ext_fields)

        values, log_probs, logits  = self.forward(H_graph, spin_sites)

        key, subkey = jax.random.split(key)

        sampled_classes = jax.random.categorical(subkey, log_probs, axis=-1)

        one_hot_spins = jax.nn.one_hot(sampled_classes, num_classes=self.n_classes)

        spin_log_prob = jnp.sum(log_probs * one_hot_spins, axis=-1)

        return values, spin_log_prob, sampled_classes, log_probs, logits, key

    def sample(self, H_graph, compl_H_graph, Ext_fields, key):
        H_graph, spin_sites = self._make_H_graph_input(H_graph, Ext_fields)

        values, log_probs, logits = self.forward_padded(H_graph, spin_sites)

        key, subkey = jax.random.split(key)

        sampled_classes = jax.random.categorical(subkey, log_probs, axis=-1)

        one_hot_spins = jax.nn.one_hot(sampled_classes, num_classes=self.n_classes)

        spin_log_prob = jnp.sum(log_probs * one_hot_spins, axis=-1)

        return values[:-1], spin_log_prob[:-1], sampled_classes[:-1], log_probs[:-1], logits[:-1], key

    def eval_padded(self, H_graph, masks, Ext_fields, actions):
        ### TODO add embedding for next spin which has to be generated
        observed_class = actions[:,0]

        H_graph, spin_sites = self._make_H_graph_input(H_graph, Ext_fields)

        values, _, logits = self.forward_padded(H_graph, spin_sites)

        one_hot_spins = jax.nn.one_hot(observed_class, num_classes=self.n_classes)

        log_probs = softmax_utils.log_softmax(logits, masks)
        log_probs = jnp.where(one_hot_spins == 1, log_probs[:-1], jnp.zeros_like(log_probs[:-1]))
        spin_log_prob = jnp.sum(log_probs, axis=-1)

        return values[:-1], spin_log_prob

    def eval(self, H_graph, compl_H_graph, Ext_fields, actions):
        print("unpadded")
        print(actions.shape)
        observed_class = actions[:, 0]

        H_graph, spin_sites = self._make_H_graph_input(H_graph, Ext_fields)

        values, log_probs, logits = self.forward(H_graph, spin_sites)

        one_hot_spins = jax.nn.one_hot(observed_class, num_classes=self.n_classes)

        spin_log_prob = jnp.sum(log_probs * one_hot_spins, axis=-1)
        print("datatype", log_probs.dtype)
        return values, spin_log_prob


if(__name__ == "__main__"):

    import jax.lax


    n_classes = 16
    n_sampled_sites = int(jnp.log2(n_classes))
    n_node = jnp.array([8,10,15,18, 7])

    first_spin_indexes = jutils.get_first_node_idxs(n_node)

    ### TODO check this line of code
    spin_site_index_mat = first_spin_indexes[:, jnp.newaxis] + jnp.arange(0, n_sampled_sites, dtype=jnp.int32)[jnp.newaxis,:]  # jnp.cumsum(graph_idx * n_node)
    spin_sites_index = jnp.ravel(spin_site_index_mat)
    spin_input_token = jnp.ravel(spin_site_index_mat - first_spin_indexes[:, jnp.newaxis] +1)

    rand_nodes = jnp.sum(n_node)
    num_nodes = rand_nodes
    Spin_identifier = jnp.zeros(num_nodes)
    Spin_identifier = Spin_identifier.at[spin_sites_index].set(spin_input_token)

    one_hot_identifier = jax.nn.one_hot(Spin_identifier, num_classes=n_sampled_sites + 1)

    GNN_embeddings = jnp.concatenate([jnp.arange(0, num_nodes)[:, jnp.newaxis] , one_hot_identifier], axis = -1)

    sampled_spin_embeddings = GNN_embeddings[spin_sites_index]
    concat_node_embeddings = jnp.reshape(sampled_spin_embeddings, (n_node.shape[0], n_sampled_sites* sampled_spin_embeddings.shape[-1]))

    pass