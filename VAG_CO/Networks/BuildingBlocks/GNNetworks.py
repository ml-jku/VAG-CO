import jax
import flax.linen as nn
import jraph
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Optional, Tuple
import jax.tree_util as tree

# lambda x: hk.Linear(3, dtype = jnp.complex128, param_dtype = jnp.complex128, kernel_init=nn.initializers.normal(stddev=0.01),
#                        bias_init=nn.initializers.normal(stddev=0.01))(x)
### TODO set dtype globally
dtype_list = [jnp.complex64, jnp.float32]
datatype = dtype_list[1]

#### TODO add mean message passing

def calculate_graph_magnetisation(up_down_spins,graph):
    nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
    nodes = up_down_spins

    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)

    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)

    magnetisation = jax.ops.segment_sum(nodes, node_gr_idx, n_graph)

    return magnetisation

def calculate_graph_Nup_Ndown(up_down_spins,graph):
    nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
    nodes = up_down_spins

    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)

    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)

    N_up = jax.ops.segment_sum(jnp.where(nodes==1, 1, 0), node_gr_idx, n_graph)
    N_down = jax.ops.segment_sum(jnp.where(nodes==-1, 1, 0), node_gr_idx, n_graph)

    return N_up, N_down

def add_self_edges_fn(receivers: jnp.ndarray, senders: jnp.ndarray, edges: jnp.array,
                      total_num_nodes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Adds self edges. Assumes self edges are not in the graph yet."""
  #print(receivers.dtype, jnp.arange(total_num_nodes).dtype)
  receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)), axis=0)
  senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
  #print(edges.shape, jnp.zeros([total_num_nodes, 1]).shape)
  edges = jnp.concatenate((edges, jnp.zeros([total_num_nodes, edges.shape[1]])), axis=0)
  # if(edges.shape[0] == 0):
  #   edges = jnp.zeros([total_num_nodes, 1])
  # else:
  #   edges = jnp.concatenate((edges, jnp.zeros([total_num_nodes, 1])), axis=0)
  return receivers, senders, edges

class FlaxDropoutMLP(nn.Module):
  features: jnp.ndarray
  training: bool = False
  dropout: float = 0.2

  def setup(self):
      layers = []
      for feat in self.features[:-1]:
          layers.append(nn.Dense(feat, dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros))
          layers.append(jax.nn.relu)
          layers.append(nn.Dropout(rate=self.dropout, deterministic=not self.training))
          layers.append(nn.LayerNorm(dtype=datatype, param_dtype=datatype))
      layers.append(
          nn.Dense(self.features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                   bias_init=nn.initializers.zeros))
      layers.append(nn.Dropout(rate=self.dropout, deterministic=not self.training))
      layers.append(jax.nn.relu)
      layers.append(nn.LayerNorm(dtype=datatype, param_dtype=datatype))

      self.mlp = nn.Sequential(layers)

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.mlp(x)

class ValueMLP(nn.Module):
  features: jnp.ndarray
  training: bool = False
  dropout_rate: float = 0.2
  selu: bool = False

  def setup(self):
      layers = []
      for feat in self.features[:-1]:
          layers.append(nn.Dense(feat, dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros))
          if(self.selu):
              layers.append(jax.nn.selu)
          else:
              layers.append(jax.nn.relu)
              layers.append(nn.LayerNorm(dtype=datatype, param_dtype=datatype))
      layers.append(
          nn.Dense(self.features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                   bias_init=nn.initializers.zeros))
      #layers.append(nn.Dropout(rate=self.dropout_rate, broadcast_dims = (0,), deterministic=not self.training))

      self.mlp = nn.Sequential(layers)

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.mlp(x)

class FlaxMLP(nn.Module):
  features: jnp.ndarray
  training: bool = False
  dropout_rate: float = 0.2
  layer_norm: bool = True

  def setup(self):

      layers = []
      for feat in self.features[:-1]:
          layers.append(nn.Dense(feat, dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros))

          layers.append(jax.nn.relu)

          if(self.layer_norm):
            layers.append(nn.LayerNorm(dtype=datatype, param_dtype=datatype))

      layers.append(
          nn.Dense(self.features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                   bias_init=nn.initializers.zeros))
      #layers.append(nn.Dropout(rate=self.dropout_rate, broadcast_dims = (0,), deterministic=not self.training))

      layers.append(jax.nn.relu)

      if (self.layer_norm):
          layers.append(nn.LayerNorm(dtype=datatype, param_dtype=datatype))

      self.mlp = nn.Sequential(layers)

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.mlp(x)

class ProbMLP(nn.Module):
    features: jnp.ndarray
    training: bool = False
    dropout_rate: float = 0.2
    selu: bool = False

    def setup(self):
        layers = []
        for feat in self.features[:-1]:
            layers.append(nn.Dense(feat, dtype = datatype, param_dtype = datatype, kernel_init=nn.initializers.he_normal(),
                           bias_init=nn.initializers.zeros))
            if (self.selu):
                layers.append(jax.nn.selu)
            else:
                layers.append(jax.nn.relu)
                layers.append(nn.LayerNorm(dtype=datatype, param_dtype=datatype))
        layers.append(nn.Dense(self.features[-1], dtype = datatype, param_dtype = datatype, kernel_init=nn.initializers.xavier_normal(),
                       bias_init=nn.initializers.zeros))


        self.mlp = nn.Sequential(layers)
        self.log_softm = lambda x: jax.nn.log_softmax(x, axis = -1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        logits = self.mlp(x)
        return self.log_softm(logits), logits




class ResidualLinearMessageLayer(nn.Module):
    #print("WARNING: this linear network is made for undirected graphs!")
    GNN_MLP_features: list

    def setup(self):
        self.W_edge = nn.Dense(self.GNN_MLP_features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros, use_bias = False)
        self.W_mess = nn.Dense(self.GNN_MLP_features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros, use_bias = False)

        if(False):
            self.W_self = nn.Dense(self.GNN_MLP_features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                                     bias_init=nn.initializers.zeros)
        else:
            self.W_self = self.W_mess

        self.W_skip = nn.Dense(self.GNN_MLP_features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros)

        self.LayerNorm = nn.LayerNorm( dtype=datatype, param_dtype=datatype)

        self.activ = nn.elu

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies a Graph Attention layer."""
        nodes, edges, receivers, senders, _, n_node, n_edges = graph

        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        edge_embed = self.W_edge(edges)
        ### TODO check sending and stuff here
        sent_messages = self.W_mess(nodes[senders])
        messages = edge_embed + sent_messages

        num_neighbours = jax.ops.segment_sum(jnp.ones_like(messages), receivers, total_num_nodes) + jax.ops.segment_sum(jnp.ones_like(messages), senders, total_num_nodes)
        aggr_messages = jax.ops.segment_sum(messages, receivers, total_num_nodes) + jax.ops.segment_sum(messages, senders, total_num_nodes)

        ### take the mean
        aggr_messages = aggr_messages/num_neighbours

        out = self.activ(self.LayerNorm(aggr_messages + self.W_self(nodes)))

        return graph._replace(nodes =  self.W_skip(nodes) + out, edges = edges)

class MPNN(nn.Module):
    #print("WARNING: this nonlinear network has been changed to work with directed graphs!")
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    edgeMLP: bool = True
    training: bool = False
    layer_norm: bool = True


    def setup(self):
        self.W_node = nn.Dense(self.node_MLP_features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros, use_bias = False)

        self.W_message = nn.Dense(self.message_MLP_features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros, use_bias = False)

        self.NodeMLP = FlaxMLP(features=self.node_MLP_features, training = self.training, layer_norm = self.layer_norm)

        self.ln_nodes = nn.LayerNorm(dtype=datatype, param_dtype=datatype)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies a Graph Attention layer."""
        nodes, edges_sparse, receivers_sparse, senders_sparse, _, n_node, n_edges = graph
        senders = jnp.concatenate([senders_sparse, receivers_sparse], axis = 0)
        receivers = jnp.concatenate([receivers_sparse, senders_sparse], axis = 0)
        edges = jnp.concatenate([edges_sparse, edges_sparse], axis = 0)

        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        #print("total_num_nodes", total_num_nodes, "num_nodes", nodes.shape[0])


        sent_attributes = nodes[senders]
        concated_message_input = jnp.concatenate([sent_attributes, edges], axis = -1)
        messages = self.W_message(concated_message_input)
        #print("messages", jnp.any(jnp.isnan(messages)))

        aggr_messages = jax.ops.segment_sum(messages, receivers, total_num_nodes)

        concated_nodeMLP_input = jnp.concatenate([nodes, aggr_messages], axis = -1)
        #print("concated_nodeMLP_input", jnp.any(jnp.isnan(concated_nodeMLP_input)))
        out = self.NodeMLP(concated_nodeMLP_input)

        #print(nodes.shape, self.W(out).shape)
        new_nodes = self.ln_nodes(self.W_node(nodes) + out)
        return graph._replace(nodes = new_nodes)

class MPNN_simple(nn.Module):
    #print("WARNING: this nonlinear network has been changed to work with directed graphs!")
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    edgeMLP: bool = True
    training: bool = False
    layer_norm: bool = True


    def setup(self):
        self.W_message = nn.Dense(self.message_MLP_features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros, use_bias = False)

        self.NodeMLP = FlaxMLP(features=self.node_MLP_features, training = self.training, layer_norm = self.layer_norm)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies a Graph Attention layer."""
        nodes, edges_sparse, receivers_sparse, senders_sparse, _, n_node, n_edges = graph
        senders = jnp.concatenate([senders_sparse, receivers_sparse], axis = 0)
        receivers = jnp.concatenate([receivers_sparse, senders_sparse], axis = 0)
        edges = jnp.concatenate([edges_sparse, edges_sparse], axis = 0)

        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        sent_attributes = nodes[senders]
        concated_message_input = jnp.concatenate([sent_attributes, edges], axis = -1)
        messages = self.W_message(concated_message_input)

        aggr_messages = jax.ops.segment_sum(messages, receivers, total_num_nodes)

        concated_nodeMLP_input = jnp.concatenate([nodes, aggr_messages], axis = -1)
        out = self.NodeMLP(concated_nodeMLP_input)

        return graph._replace(nodes = out)

class LearnableConstant(nn.Module):
    constant_size: int

    def setup(self):
        self.constant = self.param('constant', nn.Embedding, self.constant_size)

    def __call__(self, x):
        constant_value = self.constant(jnp.zeros((1,)))  # Access the learnable constant value
        return x * constant_value

class GIN(nn.Module):
    #print("WARNING: this nonlinear network has been changed to work with directed graphs!")
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    edgeMLP: bool = True
    training: bool = False
    layer_norm: bool = True


    def setup(self):
        self.eps = LearnableConstant(1)

        self.MessageMLP = FlaxMLP(features=self.message_MLP_features, training=self.training, layer_norm=self.layer_norm)
        self.selfMLP = FlaxMLP(features=self.message_MLP_features, training=self.training, layer_norm=self.layer_norm)

        self.NodeMLP = FlaxMLP(features=self.node_MLP_features, training = self.training, layer_norm = self.layer_norm)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies a Graph Attention layer."""
        nodes, edges_sparse, receivers_sparse, senders_sparse, _, n_node, n_edges = graph
        senders = jnp.concatenate([senders_sparse, receivers_sparse], axis = 0)
        receivers = jnp.concatenate([receivers_sparse, senders_sparse], axis = 0)
        edges = jnp.concatenate([edges_sparse, edges_sparse], axis = 0)

        h_nodes = self.selfMLP(nodes)

        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        sent_attributes = nodes[senders]
        concated_message_input = jnp.concatenate([sent_attributes, edges], axis = -1)
        messages = self.MessageMLP(concated_message_input)

        aggr_messages = jax.ops.segment_sum(messages, receivers, total_num_nodes)
        out = self.NodeMLP(1 *h_nodes + self.eps(h_nodes) + aggr_messages)

        return graph._replace(nodes = out)

class GIN_skip():
    def __init__(self):
        raise ValueError("not implemented")



class ResidualNonlinearMessageLayer(nn.Module):
    #print("WARNING: this nonlinear network has been changed to work with directed graphs!")
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    edgeMLP: bool = False
    training: bool = False
    layer_norm: bool = True


    def setup(self):
        self.W_node = nn.Dense(self.node_MLP_features[-1], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros, use_bias = False)

        self.MessageMLP = FlaxMLP(features=self.message_MLP_features, training = self.training, layer_norm = self.layer_norm )
        self.NodeMLP = FlaxMLP(features=self.node_MLP_features, training = self.training, layer_norm = self.layer_norm)
        if(self.edgeMLP):
            self.EdgeMLP = FlaxMLP(features=self.edge_MLP_features, training = self.training, layer_norm = self.layer_norm)
            self.W_edge = nn.Dense(self.edge_MLP_features[-1], dtype=datatype, param_dtype=datatype,
                                   kernel_init=nn.initializers.xavier_normal(),
                                   bias_init=nn.initializers.zeros, use_bias=False)

        self.ln_edges = nn.LayerNorm(dtype=datatype, param_dtype=datatype)
        self.ln_nodes = nn.LayerNorm(dtype=datatype, param_dtype=datatype)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies a Graph Attention layer."""
        nodes, edges_sparse, receivers_sparse, senders_sparse, _, n_node, n_edges = graph
        senders = jnp.concatenate([senders_sparse, receivers_sparse], axis = 0)
        receivers = jnp.concatenate([receivers_sparse, senders_sparse], axis = 0)
        edges = jnp.concatenate([edges_sparse, edges_sparse], axis = 0)


        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        concated_message_input = jnp.concatenate([sent_attributes, received_attributes, edges], axis = -1)

        messages = self.MessageMLP(concated_message_input)

        aggr_messages = jax.ops.segment_sum(messages, receivers, total_num_nodes)

        concated_nodeMLP_input = jnp.concatenate([nodes, aggr_messages], axis = -1)
        #print("concated_nodeMLP_input", jnp.any(jnp.isnan(concated_nodeMLP_input)))
        out = self.NodeMLP(concated_nodeMLP_input)

        if(self.edgeMLP):
            raise ValueError("This code should be updated -> edge updates are stale")
            out_edges = self.EdgeMLP(concated_message_input)
            new_edges = self.ln_edges(out_edges + self.W_edge(edges))

        #print(nodes.shape, self.W(out).shape)
        new_nodes = self.ln_nodes(self.W_node(nodes) + out)
        return graph._replace(nodes = new_nodes)


class GAT(nn.Module): ### TODO this has been changed and should be tested before use
    print("WARNING: this nonlinear network has been changed to work with directed graphs!")
    features: list
    heads: int = 4
    w_init: Callable = nn.initializers.he_normal()
    a_init: Callable = nn.initializers.xavier_normal()

    def setup(self):
        self.att_features = int(self.features[0]/self.heads)
        # self.W_self = nn.Dense(self.features[0], dtype=datatype, param_dtype=datatype, kernel_init=nn.initializers.he_normal(),
        #                          bias_init=nn.initializers.zeros, use_bias = False)
        self.ln1 = nn.LayerNorm()
        # self.ln2 = nn.LayerNorm()
        # self.MLP = FlaxMLP(features=self.features)

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies a Graph Attention layer."""
        nodes, edges, receivers, senders, _, n_node, n_edges = graph

        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        receivers, senders, edges = add_self_edges_fn(receivers, senders, edges, total_num_nodes)

        W = self.param('W', self.w_init, (self.heads, self.att_features, nodes.shape[-1]))
        W_edge_par = self.param('W_edge', self.w_init, (self.heads, self.att_features, edges.shape[-1]))
        a = self.param('a', self.a_init, (self.heads, 3*self.att_features,))

        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        #print("asd", W, sent_attributes.shape)
        # print(jax.lax.dot_general(W, sent_attributes, (((-1, ), (-1)), ((0,), ()))),)
        # W_senders = jax.lax.dot_general(W, sent_attributes, (((-1, ), (-1)), ((0,), ())),)
        W_senders = jnp.einsum("hfi, nj -> hnf", W, sent_attributes)
        W_receivers = jnp.einsum("hfi, nj -> hnf", W, received_attributes)
        W_edges = jnp.einsum("hfi, nj -> hnf", W_edge_par, edges)

        #print(nodes.shape, senders.shape, sent_attributes.shape, W_senders.shape, W_edges.shape)
        concat_message = jnp.concatenate([W_senders, W_receivers, W_edges], axis = -1)
        message = jnp.einsum("hf, hnf -> hn", a, concat_message)
        leaky_message = jnp.nn.leaky_relu(message)

        z = leaky_message - jax.ops.segment_max(leaky_message, receivers, total_num_nodes)
        exp_z = nn.exp(z)
        soft_norm = jax.ops.segment_sum(exp_z, receivers, total_num_nodes)
        alpha = jax.ops.segment_sum(exp_z/soft_norm[receivers], receivers, total_num_nodes)

        att_message = jnp.expand_dims(alpha, axis=2) *  W_senders

        att_message = jnp.swapaxes(att_message, 0,1)
        aggr_messages = jax.ops.segment_sum(att_message, receivers, total_num_nodes)

        concat_aggr_messages = jnp.reshape(aggr_messages, (aggr_messages.shape[0], aggr_messages.shape[1]*aggr_messages.shape[2]))

        new_nodes = nn.elu(concat_aggr_messages)
        # new_nodes = self.W_self(nodes) + nn.elu(concat_aggr_messages)
        new_nodes = self.ln1(new_nodes)
        # new_nodes = new_nodes + self.MLP(new_nodes)
        # new_nodes = self.ln2(new_nodes)

        return graph._replace(nodes = new_nodes)



def test_networks():
    pass

if __name__ == "__main__":

    node_features = [[1.],[2.],[3.],[4.],[5.]]
    senders = [0,1,2,3,4]
    receivers = [1,2,3,4,0]
    edges = [[1.], [1.], [1.], [1.], [1.]]


    node_features = jnp.array(node_features)

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)

    edges = jnp.array(edges)

    n_node = jnp.array([len(node_features)])
    n_edge = jnp.array([len(edges)])

    global_context = jnp.array([[1]])  ### Global context is not needed
    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context
    )

    layer = ResidualNonlinearMessageLayer(1,1)

    key = jax.random.PRNGKey(0)
    params = layer.init(key,graph)
    #print(params)

    out = layer.apply(params, graph)
    print(out)

