import numpy as np
from jraph_utils import utils as jutils
import copy

def solveMIS(j_graph):

    i_graph = jutils.from_jgraph_to_igraph(j_graph)
    orig_i_graph = copy.deepcopy(i_graph)
    ### TODO adapt code so that vertex position can be determined can also be set
    vertex_set = []
    finished = False
    if(j_graph.edges.shape[0] == 0):
        vertex_set = [1 for i in range(j_graph.nodes.shape[0])]
    else:
        while(not finished):
            degrees = i_graph.degree()
            num_vertices = i_graph.vcount()
            vertices = np.arange(0, num_vertices)
            sorted_vertices = sorted(zip(vertices, degrees), key = lambda x: x[1])

            min_degree_node = sorted_vertices[0][0]

            vertex_set.append(min_degree_node)

            neighbours = i_graph.neighbors(i_graph.vs[min_degree_node])

            delete_nodes = [min_degree_node] + neighbours

            i_graph.delete_vertices(delete_nodes)

            if(i_graph.vcount() == 0):
                finished = True

    pred_Energy = -len(vertex_set)
    return pred_Energy




if(__name__ == "__main__"):

    import igraph

    ig = igraph.Graph()
    ig.add_vertices(5)
    ig.add_edges([(0,1),(1,2), (3,4), (2,4), (4,1)])

    num_vertices = ig.vcount()
    vertices = np.arange(0,num_vertices)

