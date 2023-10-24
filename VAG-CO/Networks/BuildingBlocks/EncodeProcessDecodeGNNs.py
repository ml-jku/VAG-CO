import jax.numpy as jnp
import flax.linen as nn
import jraph
from Networks.BuildingBlocks.GNNetworks import ResidualNonlinearMessageLayer, FlaxMLP, MPNN_simple, MPNN, GIN, GIN_skip, ResidualNonlinearMessageLayer
import jax

dtype_list = [jnp.complex64, jnp.float32]
datatype = dtype_list[1]


class EncodeProcessDecode(nn.Module):
    GNN_MLP_features: list
    n_layers: int = 5
    message_passing: str = "non_linear"
    training: bool = False
    weight_tied: bool = False

    def setup(self):

        module_list = []
        self.encoder_edge_MLP = FlaxMLP(features = [2*self.GNN_MLP_features[0], self.GNN_MLP_features[-1]], training = self.training)
        self.encoder_node_MLP = FlaxMLP(features = [2*self.GNN_MLP_features[0],  self.GNN_MLP_features[-1]], training = self.training)
        self.decoder_node_MLP = FlaxMLP(features = [self.GNN_MLP_features[0], 2*self.GNN_MLP_features[-1]], training = self.training)

        if(not self.weight_tied):
            print("not weight tied GNN is used")
            for i in range(self.n_layers):
                if(i != self.n_layers - 1):
                    layer = ResidualNonlinearMessageLayer(self.GNN_MLP_features, True, training = self.training)
                else:
                    layer = ResidualNonlinearMessageLayer(self.GNN_MLP_features, False, training = self.training)

                module_list.append(layer)
        else:
            print("weight tied GNN is used")
            layer = ResidualNonlinearMessageLayer(self.GNN_MLP_features, True, training=self.training)
            for i in range(self.n_layers):
                module_list.append(layer)

        self.module_list = module_list


    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

        nodes = graph.nodes
        edges = graph.edges

        enc_nodes = self.encoder_node_MLP(nodes)
        enc_edges = self.encoder_edge_MLP(edges)

        graph = graph._replace(nodes = enc_nodes, edges = enc_edges)

        for layer in self.module_list:
            graph = layer(graph)

        nodes = graph.nodes
        decode_nodes = self.decoder_node_MLP(nodes)

        return decode_nodes


class EncodeProcess_node_embedding(nn.Module):
    ### TODO maybe only encode hamiltonian graph in first layer
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    encode_node_features: list
    encode_edge_features: list
    n_layers: int = 2
    message_passing: str = "non_linear"
    layer_norm: bool = True
    edge_updates: bool = False
    training: bool = False
    weight_tied: bool = False

    def setup(self):

        module_list = []

        if(self.message_passing == "MPNN_simple"):
            Graph_func = MPNN_simple
        elif(self.message_passing == "MPNN"):
            Graph_func =  MPNN
        elif(self.message_passing == "MPNN_nonlinear"):
            Graph_func =  ResidualNonlinearMessageLayer
        elif (self.message_passing == "GIN"):
            Graph_func = GIN
        elif (self.message_passing == "GIN_skip"):
            Graph_func = GIN_skip
        else:
            ValueError("This type of GNN Network is not implemented")

        print("edge updates",self.edge_updates)
        print("num_layers", self.n_layers)

        self.encoder_node_MLP = FlaxMLP(features = self.encode_node_features, training = self.training, layer_norm = self.layer_norm)

        for i in range(self.n_layers):
            layer = Graph_func(self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features
                               , layer_norm=self.layer_norm, edgeMLP=self.edge_updates, training=self.training)

            module_list.append(layer)

        self.module_list = module_list

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

        nodes = graph.nodes

        enc_nodes = self.encoder_node_MLP(nodes)

        graph = graph._replace(nodes=enc_nodes)
        for layer in self.module_list:
            # print("before", jnp.any(jnp.isnan(graph.nodes)))
            # print("edges", jnp.any(jnp.isnan(graph.edges)))
            graph = layer(graph)
            # print("after", jnp.any(jnp.isnan(graph.nodes)))
            # print("edges", jnp.any(jnp.isnan(graph.edges)))

        nodes = graph.nodes
        return nodes

class EncodeProcessNew(nn.Module):
    ### TODO maybe only encode hamiltonian graph in first layer
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    encode_node_features: list
    encode_edge_features: list
    n_layers: int = 2
    message_passing: str = "non_linear"
    layer_norm: bool = True
    edge_updates: bool = False
    training: bool = False
    weight_tied: bool = False

    def setup(self):

        print(self.message_passing)
        raise ValueError("stopped")
        if(self.message_passing == "MPNN_non_linear"):
            Graph_func =  ResidualNonlinearMessageLayer
        elif(self.message_passing == "MPNN_simple"):
            Graph_func = MPNN_simple
        elif(self.message_passing == "MPNN"):
            Graph_func =  MPNN
        elif (self.message_passing == "GIN"):
            Graph_func = GIN
        elif (self.message_passing == "GIN_skip"):
            Graph_func = GIN_skip
        else:
            ValueError("This type of GNN Network is not implemented")


        self.embedding_GNN_layer = Graph_func(self.encode_edge_features, self.encode_node_features, self.encode_edge_features
                                                      , layer_norm = self.layer_norm, edgeMLP = self.edge_updates, training = self.training)

        module_list = []
        for i in range(self.n_layers - 1):
            layer = Graph_func(self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features
                                                      , layer_norm = self.layer_norm, edgeMLP = self.edge_updates, training = self.training)

            module_list.append(layer)


        self.module_list = module_list


    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

        graph = self.embedding_GNN_layer(graph)

        compl_graph = graph._replace(nodes = graph.nodes)
        for layer in self.module_list:
            # print("before", jnp.any(jnp.isnan(graph.nodes)))
            # print("edges", jnp.any(jnp.isnan(graph.edges)))
            compl_graph = layer(compl_graph)
            # print("after", jnp.any(jnp.isnan(graph.nodes)))
            # print("edges", jnp.any(jnp.isnan(graph.edges)))

        nodes = compl_graph.nodes
        return nodes

class EncodeProcess(nn.Module):
    message_MLP_features: list
    node_MLP_features: list
    edge_MLP_features: list
    encode_node_features: list
    encode_edge_features: list
    n_layers: int = 2
    message_passing: str = "non_linear"
    layer_norm: bool = True
    edge_updates: bool = False
    training: bool = False
    weight_tied: bool = False

    def setup(self):

        module_list = []
        self.encoder_edge_MLP = FlaxMLP(features = self.encode_edge_features, training = self.training, layer_norm = self.layer_norm)
        self.encoder_node_MLP = FlaxMLP(features = self.encode_node_features, training = self.training, layer_norm = self.layer_norm)

        if(self.message_passing == "non_linear"):
            Graph_func =  ResidualNonlinearMessageLayer
        else:
            Graph_func =  MPNN_simple


        for i in range(self.n_layers):
            layer = Graph_func(self.message_MLP_features, self.node_MLP_features, self.edge_MLP_features
                                                      , layer_norm = self.layer_norm, edgeMLP = self.edge_updates, training = self.training)

            module_list.append(layer)


        self.module_list = module_list


    def __call__(self, graph: jraph.GraphsTuple, compl_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

        nodes = graph.nodes
        edges = graph.edges

        enc_nodes = self.encoder_node_MLP(nodes)
        enc_edges = self.encoder_edge_MLP(edges)

        graph = graph._replace(nodes = enc_nodes, edges = enc_edges)

        for layer in self.module_list:
            # print("before", jnp.any(jnp.isnan(graph.nodes)))
            # print("edges", jnp.any(jnp.isnan(graph.edges)))
            graph =  layer(graph)
            # print("after", jnp.any(jnp.isnan(graph.nodes)))
            # print("edges", jnp.any(jnp.isnan(graph.edges)))

        nodes = graph.nodes
        return nodes




