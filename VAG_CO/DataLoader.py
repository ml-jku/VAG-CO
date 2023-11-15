import jraph

from torch.utils.data import Dataset
import numpy as np

### TODO add inheritage
class General1DGridDataset(Dataset):
    def __init__(self, O_graphs, nx_G, H_graph, n_nodes):
        super().__init__()
        self.x = n_nodes
        self.y = 1
        self.O_graphs = O_graphs
        (sigma_x_graph, sigma_y_graph, sigma_z_graph) = self.O_graphs
        self.H_graph = H_graph

        self.n_nodes = len(sigma_x_graph.nodes)
        self.num_spins = self.n_nodes

        # self.nx_graph_list = build_up_graph_snake(nx_graph)
        #
        # #self.message_graph_list = [from_networkx_to_jraph(message_graph)  for message_graph in self.message_graph_list]
        self.back_transform = np.arange(0, n_nodes)
        self.spiral_transform = np.arange(0, n_nodes)
        #
        # self.jraph_graph_list = [ from_networkx_to_jraph(nx_graph) for nx_graph in self.nx_graph_list]
        #
        # self.argwhere_graph = [make_index_graph(jraph_g) for jraph_g in self.jraph_graph_list]


    def len(self):
        #return len(self.jraph_graph_list)
        return 0

    def __getitem__(self, idx):
        #return self.jraph_graph_list[idx]
        return None

class GeneralPlaceholder(Dataset):
    def __init__(self, cfg):
        super().__init__()
        ### TODO add spin identifier
        n_rand_nodes = cfg["Ising_params"]["n_rand_nodes"]
        edge_embedding_type = cfg["Ising_params"]["edge_embedding_type"]
        self.x = 100
        self.y = 1
        n_edges = 10

        senders = np.arange(0,n_edges)
        receivers = np.arange(0,n_edges)

        nodes = np.ones((self.x, n_rand_nodes))

        if(edge_embedding_type == False):
            edges = np.ones((n_edges,1))
        else:
            edges = np.ones((n_edges,1+2*n_rand_nodes))
        n_node = np.array([self.x])
        n_edge = np.array([[n_edges]])
        globals = np.array([0])

        H_graph = jraph.GraphsTuple(senders = senders, receivers = receivers, nodes = nodes, edges = edges, n_node = n_node, n_edge = n_edge, globals = globals)

        self.H_graph = H_graph

        self.n_nodes = self.x
        self.num_spins = self.n_nodes

        # self.nx_graph_list = build_up_graph_snake(nx_graph)
        #
        # #self.message_graph_list = [from_networkx_to_jraph(message_graph)  for message_graph in self.message_graph_list]
        self.back_transform = np.arange(0, self.n_nodes)
        self.spiral_transform = np.arange(0, self.n_nodes)
        #
        # self.jraph_graph_list = [ from_networkx_to_jraph(nx_graph) for nx_graph in self.nx_graph_list]
        #
        # self.argwhere_graph = [make_index_graph(jraph_g) for jraph_g in self.jraph_graph_list]


    def len(self):
        #return len(self.jraph_graph_list)
        return 0

    def __getitem__(self, idx):
        #return self.jraph_graph_list[idx]
        return None



