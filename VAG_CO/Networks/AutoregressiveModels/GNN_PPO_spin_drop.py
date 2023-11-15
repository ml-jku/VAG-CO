from ..BuildingBlocks.EncodeProcessDecodeGNNs import EncodeProcess
from ..BuildingBlocks.GNNetworks import ProbMLP, ValueMLP
import flax.linen as nn
import jax.numpy as jnp
import jax
import jax.tree_util as tree
from ...jraph_utils import utils as jutils

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
        self.encode_MLP_features = self.cfg.Network_params.GNNs.encode_MLP_features
        self.n_GNN_layers = self.cfg.Network_params.GNNs.n_GNN_layers
        self.policy_MLP_features = self.cfg.Network_params.policy_MLP_features
        self.value_MLP_features = self.cfg.Network_params.value_MLP_features

        self.policy_global_features = self.cfg.Network_params.GNNs.policy_global_features
        self.value_global_features = self.cfg.Network_params.GNNs.value_global_features

        self.EnergyFunction = self.cfg.Ising_params.EnergyFunction

        self.layer_norm = self.cfg.Network_params.layer_norm
        self.edge_updates = self.cfg.Network_params.GNNs.edge_updates

        self.lam = self.cfg.Train_params.PPO.lam
        self.GNN_mode = self.cfg.Network_params.GNNs.mode
        #RNN = lambda x: GRNN_inside_loop(SampleMode = self.SampleMode,out_features = self.out_features, hidden_RNN_features = self.hidden_RNN_features, n_RNN_layers = self.n_RNN_layers, training = self.training)
        self.cell = nn.scan(
            nn.jit(GNN_actor_critic),
            variable_broadcast="params",
            split_rngs={"params": False})

        self.apply_cell = self.cell(self.spiral_transform, self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features, self.encode_MLP_features, self.n_GNN_layers,
                                    self.policy_MLP_features, self.value_MLP_features, self.EnergyFunction, self.policy_global_features, self.value_global_features,
                                    layer_norm = self.layer_norm, edge_updates = self.edge_updates, SampleMode = self.SampleMode, training = self.training)

        self.gamma = 1.


    @nn.compact
    def __call__(self, H_graph, ones, key, T):
        if (self.SampleMode == "sample"):
            return self.sample(H_graph, key)
        elif(self.SampleMode == "sample_unpadded"):
            actions = ones
            return self.sample_unpadded(H_graph, key) # T serves here as an idx
        elif(self.SampleMode == "eval_spin"):
            actions = ones
            return self.eval_spin(H_graph, actions)
        elif(self.SampleMode == "sample_sparse"):
            return self.sample(H_graph, key)
        elif(self.SampleMode == "eval_spin_sparse"):
            actions = ones
            return self.eval_spin_padded(H_graph, actions)#

    def sample(self, H_graphs, key):
        return self.apply_cell.sample(H_graphs, key)

    def sample_unpadded(self, H_graphs, key):
        return self.apply_cell.sample_unpadded(H_graphs, key)

    def eval_spin(self, H_graph, actions):
        values, log_probs = self.apply_cell.eval(H_graph, actions)

        return values, log_probs

    def eval_spin_padded(self, H_graph, actions):
        values, log_probs = self.apply_cell.eval_padded(H_graph, actions)

        return values, log_probs


class GNN_actor_critic(nn.Module):
    spiral_transform: list
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    encode_MLP_features: list
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

    def setup(self):
        # self.value_GNN = EncodeProcessDecode(self.GNN_MLP_features, n_layers = self.n_GNN_layers,
        #                                message_passing = self.GNN_mode, training = self.training, weight_tied = False)
        self.prob_GNN = EncodeProcess(self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features, self.encode_MLP_features, layer_norm = self.layer_norm,
                                       edge_updates = self.edge_updates,  n_layers = self.n_GNN_layers,
                                       message_passing = self.GNN_mode, training = self.training, weight_tied = False)
        self.probMLP = ProbMLP(features= self.policy_MLP_features, training = self.training)
        self.valueMLP = ValueMLP(features= self.value_MLP_features, training = self.training)

    def __call__(self, carry, xs):
        if(self.SampleMode == "sample"):
            return self.sample(carry, xs)
        elif (self.SampleMode == "sample_sparse"):
            return self.sample_batched_graphs(carry, xs)

    def forward_padded(self, H_graph, spin_sites):
        GNN_embeddings = self.prob_GNN(H_graph)
        node_embedding = GNN_embeddings[spin_sites]

        ### TODO add global features
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
            print("node embedding shape", node_embedding.shape)
            print(GNN_embeddings.shape)
            print("sum embedding shape", sum_embedding.shape)
            concat_embedding = jnp.concatenate([sum_embedding, node_embedding], axis = -1)

        if(self.value_global_features):
            values = self.valueMLP(concat_embedding)
        else:
            values = self.valueMLP(node_embedding)

        if(self.policy_global_features):
            log_prob = self.probMLP(concat_embedding)
        else:
            log_prob = self.probMLP(node_embedding)

        return values, log_prob

    def forward(self, H_graph, spin_sites):
        GNN_embeddings = self.prob_GNN(H_graph)
        node_embedding = jnp.squeeze(GNN_embeddings[spin_sites], axis = -2)

        ### TODO add global features
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
            print("node embedding shape", node_embedding.shape)
            print(GNN_embeddings.shape)
            print("sum embedding shape", sum_embedding.shape)
            concat_embedding = jnp.concatenate([sum_embedding, node_embedding], axis = -1)

        if(self.value_global_features):
            values = self.valueMLP(concat_embedding)
        else:
            values = self.valueMLP(node_embedding)

        if(self.policy_global_features):
            log_prob = self.probMLP(concat_embedding)
        else:
            log_prob = self.probMLP(node_embedding)

        return values, log_prob


    def sample_unpadded(self, H_graph, key):
        n_node = H_graph.n_node
        spin_site = jutils.get_first_node_idxs(n_node)#jnp.cumsum(graph_idx * n_node)

        External_field = H_graph.nodes
        Spin_identifier = jnp.zeros(External_field.shape[0])
        Spin_identifier = Spin_identifier.at[spin_site].set(jnp.ones_like(spin_site))
        one_hot_identifier = jax.nn.one_hot(Spin_identifier, num_classes=2)
        H_embedding = jnp.concatenate([External_field, one_hot_identifier], axis = -1)

        print(H_embedding)

        H_graph = H_graph._replace(nodes=H_embedding)

        values, log_probs = self.forward(H_graph, spin_site)

        key, subkey = jax.random.split(key)

        sampled_bin_values = jax.random.categorical(subkey, log_probs, axis=-1)

        one_hot_spins = jax.nn.one_hot(sampled_bin_values, num_classes=2)

        spin_log_prob = jnp.sum(log_probs * one_hot_spins, axis=-1)

        return values, spin_log_prob, jnp.zeros_like(spin_site), sampled_bin_values, key

    def sample(self, H_graph, key):
        n_node = H_graph.n_node
        spin_site = jutils.get_first_node_idxs(n_node)

        External_field = H_graph.nodes
        Spin_identifier = jnp.zeros(External_field.shape[0])
        Spin_identifier = Spin_identifier.at[spin_site].set(jnp.ones_like(spin_site))
        one_hot_identifier = jax.nn.one_hot(Spin_identifier, num_classes=2)
        H_embedding = jnp.concatenate([External_field, one_hot_identifier], axis = -1)

        H_graph = H_graph._replace(nodes=H_embedding)
        values, log_probs = self.forward_padded(H_graph, spin_site)

        key, subkey = jax.random.split(key)

        sampled_bin_values = jax.random.categorical(subkey, log_probs, axis=-1)

        one_hot_spins = jax.nn.one_hot(sampled_bin_values, num_classes=2)

        spin_log_prob = jnp.sum(log_probs * one_hot_spins, axis=-1)

        return values[:-1], spin_log_prob[:-1], jnp.zeros_like(spin_site[:-1]), sampled_bin_values[:-1], key

    def eval_padded(self, H_graph, actions):
        ### TODO add embedding for next spin which has to be generated
        spin_values = actions[:,1]

        n_node = H_graph.n_node
        n_graph = n_node.shape[0]
        graph_idx = jnp.arange(n_graph)

        spin_site = jutils.get_first_node_idxs(n_node)

        External_field = H_graph.nodes
        Spin_identifier = jnp.zeros(External_field.shape[0])
        Spin_identifier = Spin_identifier.at[spin_site].set(jnp.ones_like(spin_site))
        one_hot_identifier = jax.nn.one_hot(Spin_identifier, num_classes=2)
        H_embedding = jnp.concatenate([External_field, one_hot_identifier], axis = -1)

        H_graph = H_graph._replace(nodes=H_embedding)

        values, log_probs = self.forward_padded(H_graph, spin_site)

        one_hot_spins = jax.nn.one_hot(spin_values, num_classes=2)

        spin_log_prob = jnp.sum(log_probs[:-1] * one_hot_spins, axis=-1)

        return values[:-1], spin_log_prob

    def eval(self, H_graph, actions):
        print("unpadded")
        print(actions.shape)
        spin_values = actions[:, 1]

        n_node = H_graph.n_node

        spin_site = jutils.get_first_node_idxs(n_node)

        External_field = H_graph.nodes
        Spin_identifier = jnp.zeros(External_field.shape[0])
        Spin_identifier = Spin_identifier.at[spin_site].set(jnp.ones_like(spin_site))
        one_hot_identifier = jax.nn.one_hot(Spin_identifier, num_classes=2)
        H_embedding = jnp.concatenate([External_field, one_hot_identifier], axis = -1)
        print("spin site", spin_site)
        print("H_embedding", H_embedding)
        H_graph = H_graph._replace(nodes=H_embedding)

        values, log_probs = self.forward(H_graph, spin_site)

        one_hot_spins = jax.nn.one_hot(spin_values, num_classes=2)

        spin_log_prob = jnp.sum(log_probs * one_hot_spins, axis=-1)
        return values, spin_log_prob
