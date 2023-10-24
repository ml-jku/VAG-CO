import jraph
import jax.numpy as jnp
import numpy as np

def __nearest_bigger_power_of_k(x: int, k: float) -> int:
    """Computes the nearest power of two greater than x for padding."""
    if x == 0:
        x = 1
    exponent = np.log(x) / np.log(k)
    return int(k**(int(exponent) + 1))

def pad_graph_to_nearest_power_of_k(graphs_tuple: jraph.GraphsTuple, k = 1.4, np_ = jnp) -> jraph.GraphsTuple:
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = __nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_node), k) + 1
    pad_edges_to = __nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_edge), k)
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)


def add_random_node_features(jraph_graph, n_random_node_features, seed):
    np.random.seed(seed)
    external_fields = jraph_graph.nodes
    random_bin_state = np.random.randint(0, 2, size=(len(external_fields), n_random_node_features))
    #random_bin_state = np.expand_dims(random_bin_state, axis=-1)

    jraph_nodes = np.concatenate((external_fields, random_bin_state), axis=1)
    return jraph_graph._replace(nodes=jraph_nodes)
