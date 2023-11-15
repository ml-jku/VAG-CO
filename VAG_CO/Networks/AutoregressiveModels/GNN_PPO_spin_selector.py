from ..BuildingBlocks.EncodeProcessDecodeGNNs import EncodeProcess
from ..BuildingBlocks.GNNetworks import ProbMLP, ValueMLP
import flax.linen as nn
import jax.numpy as jnp
import jax
import jraph
import jax.tree_util as tree

### TODO add observations
class GNN_PPO_SpinSelector(nn.Module):
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
        self.lam = self.cfg.Train_params.PPO.lam
        self.GNN_mode = self.cfg.Network_params.GNNs.mode
        #RNN = lambda x: GRNN_inside_loop(SampleMode = self.SampleMode,out_features = self.out_features, hidden_RNN_features = self.hidden_RNN_features, n_RNN_layers = self.n_RNN_layers, training = self.training)
        self.cell = nn.scan(
            nn.jit(GNN_actor_critic),
            variable_broadcast="params",
            split_rngs={"params": False})

        self.apply_cell = self.cell(self.spiral_transform, self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features, self.encode_MLP_features, self.n_GNN_layers,
                                    self.policy_MLP_features, self.value_MLP_features, SampleMode = self.SampleMode, training = self.training)

        self.gamma = 1.


    @nn.compact
    def __call__(self, H_graph, ones, key, T):
        if (self.SampleMode == "sample"):
            return self.sample(H_graph, key)
        elif(self.SampleMode == "eval_spin"):
            actions = ones
            return self.eval_spin(H_graph, actions) # T serves here as an idx
        elif(self.SampleMode == "sample_sparse"):
            log_probs_empty = T
            return self.sample(H_graph, key, log_probs_empty)
        elif(self.SampleMode == "eval_spin_sparse"):
            actions = ones
            return self.eval_spin_padded(H_graph, actions)#

    def sample(self, H_graphs, key, n_spins):
        return self.apply_cell.sample(H_graphs, key, n_spins)


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
    SampleMode: str = "sample"
    training: bool = False
    GNN_mode: str = "non_linear"

    def setup(self):
        # self.value_GNN = EncodeProcessDecode(self.GNN_MLP_features, n_layers = self.n_GNN_layers,
        #                                message_passing = self.GNN_mode, training = self.training, weight_tied = False)
        self.prob_GNN = EncodeProcess(self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features, self.encode_MLP_features, n_layers = self.n_GNN_layers,
                                       message_passing = self.GNN_mode, training = self.training, weight_tied = False)
        self.probEmbedMLP = ValueMLP(features= self.policy_MLP_features, training = self.training)
        self.valueMLP = ValueMLP(features= self.value_MLP_features, training = self.training)

    def __call__(self, carry, xs):
        if(self.SampleMode == "sample"):
            return self.sample(carry, xs)
        elif (self.SampleMode == "sample_sparse"):
            return self.sample_batched_graphs(carry, xs)

    def sample_unpadded(self, H_graph, key):
        nodes = H_graph.nodes
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        n_node = H_graph.n_node
        n_edge = H_graph.n_edge
        n_graph = n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        sum_n_edge = tree.tree_leaves(H_graph.edges)[0].shape[0]
        n_actions = 2 * n_node
        sum_n_nodes = tree.tree_leaves(nodes)[0].shape[0]
        sum_n_actions = 2*sum_n_nodes
        node_gr_idx_t_n_node = jnp.repeat(graph_idx * n_actions, n_actions, axis=0, total_repeat_length=sum_n_actions)
        node_action_idx = jnp.repeat(graph_idx, n_actions, axis=0, total_repeat_length=sum_n_actions)
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=sum_n_nodes)
        log_probs_empty = jnp.zeros((n_graph, sum_n_actions)) ### TODO replace this or use this as an argument

        nodes_Embeddings = self.prob_GNN(H_graph)
        prob_embeddings = self.probEmbedMLP(nodes_Embeddings)
        aggregated_node_Embeddings = jax.ops.segment_sum(nodes_Embeddings, node_gr_idx, n_graph)
        values = self.valueMLP(aggregated_node_Embeddings)

        prob_embeddings = jnp.reshape(prob_embeddings, (sum_n_actions,1))
        softmax = jraph.partition_softmax(prob_embeddings, n_actions, sum_n_actions)

        key, subkey = jax.random.split(key)

        resh_ideces = jnp.arange(0, sum_n_actions) - node_gr_idx_t_n_node  # H_graphs.senders - node_gr_idx_t_n_node
        log_probs = log_probs_empty.at[node_action_idx, resh_ideces].set(jnp.squeeze(softmax, axis=-1))  # jnp.squeeze(softmax, axis = -1)

        sampled_indeces = jax.random.categorical(key, log_probs, axis=-1)
        sampled_site = sampled_indeces % n_node
        sampled_bin_values = jnp.array(sampled_indeces / n_node, dtype=jnp.int64)

        sampled_node_indeces = sampled_site + graph_idx * n_node

        sampled_edge_indeces = jnp.repeat(sampled_node_indeces, n_edge, axis=0, total_repeat_length=sum_n_edge)

        return values, softmax[sampled_indeces], sampled_node_indeces, sampled_edge_indeces, sampled_site, sampled_bin_values, key

    def sample(self, H_graph, key, log_probs_empty):
        nodes = H_graph.nodes
        n_node = H_graph.n_node
        n_edge = H_graph.n_edge
        n_graph = n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        sum_n_edge = tree.tree_leaves(H_graph.edges)[0].shape[0]
        n_actions = 2 * n_node
        sum_n_nodes = tree.tree_leaves(nodes)[0].shape[0]
        sum_n_actions = 2*sum_n_nodes
        node_gr_idx_t_n_node = jnp.repeat(graph_idx * n_actions, n_actions, axis=0, total_repeat_length=sum_n_actions)
        node_action_idx = jnp.repeat(graph_idx, n_actions, axis=0, total_repeat_length=sum_n_actions)
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=sum_n_nodes)
        #log_probs_empty = jnp.zeros((n_graph, sum_n_actions)) ### TODO replace this

        nodes_Embeddings = self.prob_GNN(H_graph)
        prob_embeddings = self.probEmbedMLP(nodes_Embeddings)
        aggregated_node_Embeddings = jax.ops.segment_sum(nodes_Embeddings, node_gr_idx, n_graph)
        values = self.valueMLP(aggregated_node_Embeddings)

        prob_embeddings = jnp.reshape(prob_embeddings, (sum_n_actions,1))
        softmax = jraph.partition_softmax(prob_embeddings, n_actions, sum_n_actions)

        key, subkey = jax.random.split(key)

        resh_ideces = jnp.arange(0, sum_n_actions) - node_gr_idx_t_n_node  # H_graphs.senders - node_gr_idx_t_n_node
        log_probs = log_probs_empty.at[node_action_idx, resh_ideces].set(jnp.squeeze(softmax, axis=-1))  # jnp.squeeze(softmax, axis = -1)

        sampled_indeces = jax.random.categorical(key, log_probs, axis=-1)
        unpadded_n_node = n_node[:-1]
        sampled_site = sampled_indeces % unpadded_n_node
        sampled_bin_values = jnp.array(sampled_indeces / unpadded_n_node, dtype=jnp.int64)

        return values[:-1], softmax[sampled_indeces], sampled_site, sampled_bin_values, key

    def eval_padded(self, H_graph, actions):
        ### TODO maybe ignore padded graphs
        spin_sites = actions[:,0]
        spin_values = actions[:,1]

        nodes = H_graph.nodes
        n_node = H_graph.n_node
        n_graph = n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        n_actions = 2 * n_node
        sum_n_nodes = tree.tree_leaves(nodes)[0].shape[0]
        sum_n_actions = 2*sum_n_nodes
        nodes_per_graph = jax.lax.cumsum(graph_idx * n_node)
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=sum_n_nodes)

        nodes_Embeddings = self.prob_GNN(H_graph)
        prob_embeddings = self.probEmbedMLP(nodes_Embeddings)

        aggregated_node_Embeddings = jax.ops.segment_sum(nodes_Embeddings, node_gr_idx, n_graph)
        values = self.valueMLP(aggregated_node_Embeddings)

        prob_embeddings = jnp.reshape(prob_embeddings, (sum_n_actions,1))
        softmax = jraph.partition_softmax(prob_embeddings, n_actions, sum_n_actions)

        ### TODO calculate spin_sites, spin_values -> graph_idxs
        graph_idxs = spin_sites + nodes_per_graph[:-1]
        softmax_idx = jnp.array(graph_idxs + spin_values, dtype = jnp.int64)

        log_probs = softmax[softmax_idx]
        ### TODO if padded to neares power of two, add log_probs and values to gloabals and unpadd afterwards
        return values[:-1], log_probs

    def eval(self, H_graph, actions):
        ### TODO maybe ignore padded graphs
        spin_sites = actions[:,0]
        spin_values = actions[:,1]

        n_node = H_graph.n_node
        n_graph = n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        n_actions = 2 * n_node
        sum_n_actions = jnp.sum(n_actions)
        sum_n_nodes = jnp.sum(n_node)
        nodes_per_graph = jnp.repeat(graph_idx * n_node, n_node, axis=0, total_repeat_length=sum_n_nodes)
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=sum_n_nodes)

        nodes_Embeddings = self.prob_GNN(H_graph)
        prob_embeddings = self.probEmbedMLP(nodes_Embeddings)

        aggregated_node_Embeddings = jax.ops.segment_sum(nodes_Embeddings, node_gr_idx, n_graph)
        values = self.valueMLP(aggregated_node_Embeddings)

        prob_embeddings = jnp.reshape(prob_embeddings, (sum_n_actions,1))
        softmax = jraph.partition_softmax(prob_embeddings, n_actions, sum_n_actions)

        ### TODO calculate spin_sites, spin_values -> graph_idxs
        graph_idxs = spin_sites + nodes_per_graph
        softmax_idx = graph_idxs + spin_values

        log_probs = softmax[softmax_idx]
        return values, log_probs

