import copy

import igraph
import jraph

from GlobalProjectVariables import MVC_A, MVC_B
import sympy as sp
from sympy import Sum, symbols, Indexed, lambdify, simplify, expand, IndexedBase, Idx, poly
from utils.sympy_utils import get_two_body_corr, get_one_body_corr, get_constant, replace_bins_by_spins
import numpy as np
from EnergyFunctions import jraphEnergy
from collections import Counter
from jraph_utils import utils as jutils

def MVC(H_graph, B = MVC_B, A = MVC_A):
    '''
    input is an undirected jraph
    :param H_graph:
    :param B:
    :param A:
    :return:
    '''

    senders = H_graph.senders
    receivers = H_graph.receivers
    n_nodes = H_graph.nodes.shape[0]

    X = IndexedBase('X')
    Hb = [ 0.5* B*(1-X[s]) *(1-X[r]) for s,r in zip(senders,receivers)]
    Ha = [ A*X[n] for n in range(n_nodes)]
    E = Hb + Ha
    return expand(sum(E)), X

def MVC_sparse(j_graph, B = MVC_B, A = MVC_A):
    ### todo save factor of two by using directed graph
    ### TODO save factor of two by using map
    senders = j_graph.senders
    receivers = j_graph.receivers
    n_nodes = j_graph.nodes.shape[0]
    n_edges = j_graph.edges.shape[0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    n = symbols("n")
    expression = 0.5 * B * (1 - X[i]) * (1 - X[j])

    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = 2*get_two_body_corr(spin_expression, S, i,j)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    couplings = np.zeros((n_edges,1))
    constant = 0
    self_senders = np.arange(0, n_nodes)
    self_receivers = self_senders

    for idx, (s, r) in enumerate(zip(senders, receivers)):
        J_sr = J_ij

        couplings[idx] += float(J_sr)
        external_fields[s] += float(external_field_on_i)
        external_fields[r] += float(external_field_on_j)
        constant += float(self_connection)

    expression = A * X[n]
    spin_expression = replace_bins_by_spins(expression, X, S, n)
    spin_expression = expand(spin_expression)
    ext_field = get_one_body_corr(spin_expression, S, n)
    constant_per_spin = get_constant(spin_expression, S)

    for n in range(n_nodes):
        external_fields[n] += float(ext_field)
        constant += float(constant_per_spin)

    self_connections = constant/n_nodes*np.ones((2*n_nodes, 1))### TODO check if this factor of two is correct here
    new_nodes = external_fields
    new_edges = np.concatenate([ couplings, self_connections ], axis = 0)
    n_edge = np.array([new_edges.shape[0]])
    new_senders = np.concatenate([senders, self_senders, self_senders], axis = -1)
    new_receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = -1)
    H_graph = j_graph._replace(nodes = new_nodes, edges = new_edges, n_edge = n_edge, senders = new_senders, receivers = new_receivers)
    return H_graph

def MaxCut(j_graph):
    n_nodes = j_graph.nodes.shape[0]
    edges = j_graph.edges
    senders = j_graph.senders
    receivers = j_graph.receivers

    self_receivers = np.arange(0, n_nodes)
    self_senders = np.arange(0, n_nodes)
    self_edges = np.zeros((n_nodes, 1))

    senders = np.concatenate([senders, self_senders, self_senders], axis = 0)
    receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = 0)
    full_edges = np.concatenate([edges, self_edges, self_edges], axis = 0)
    n_edge = np.array([full_edges.shape[0]])

    H_graph = j_graph._replace(edges = full_edges, senders = senders, receivers = receivers, n_edge = n_edge)
    return H_graph


def WMIS_sparse(j_graph, B = MVC_B, A = MVC_A):
    ### todo save factor of two by using directed graph
    ### TODO save factor of two by using map
    senders = j_graph.senders
    receivers = j_graph.receivers
    n_nodes = j_graph.nodes.shape[0]
    n_edges = j_graph.edges.shape[0]
    weight = j_graph.nodes[:,0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    n = symbols("n")
    expression = 0.5 * B * X[i] * X[j]

    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = 2*get_two_body_corr(spin_expression, S, i,j)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    couplings = np.zeros((n_edges,1))
    constant = 0
    self_senders = np.arange(0, n_nodes)
    self_receivers = self_senders

    for idx, (s, r) in enumerate(zip(senders, receivers)):
        J_sr = J_ij

        couplings[idx] += float(J_sr)
        external_fields[s] += float(external_field_on_i)
        external_fields[r] += float(external_field_on_j)
        constant += float(self_connection)

    expression = - A * X[n]
    spin_expression = replace_bins_by_spins(expression, X, S, n)
    spin_expression = expand(spin_expression)
    ext_field = get_one_body_corr(spin_expression, S, n)
    constant_per_spin = get_constant(spin_expression, S)

    for n in range(n_nodes):
        external_fields[n] += weight[n]*float(ext_field)
        constant += weight[n]*float(constant_per_spin)

    self_connections = constant/n_nodes*np.ones((2*n_nodes, 1))### TODO check if this factor of two is correct here
    new_nodes = external_fields
    new_edges = np.concatenate([ couplings, self_connections ], axis = 0)
    n_edge = np.array([new_edges.shape[0]])
    new_senders = np.concatenate([senders, self_senders, self_senders], axis = -1)
    new_receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = -1)
    H_graph = j_graph._replace(nodes = new_nodes, edges = new_edges, n_edge = n_edge, senders = new_senders, receivers = new_receivers)
    return H_graph

def MIS_sparse(j_graph, B = MVC_B, A = MVC_A):
    ### todo save factor of two by using directed graph
    ### TODO save factor of two by using map
    senders = j_graph.senders
    receivers = j_graph.receivers
    n_nodes = j_graph.nodes.shape[0]
    n_edges = j_graph.edges.shape[0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    n = symbols("n")
    expression = 0.5 * B * X[i] * X[j]

    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = 2*get_two_body_corr(spin_expression, S, i,j)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    couplings = np.zeros((n_edges,1))
    constant = 0
    self_senders = np.arange(0, n_nodes)
    self_receivers = self_senders

    for idx, (s, r) in enumerate(zip(senders, receivers)):
        J_sr = J_ij

        couplings[idx] += float(J_sr)
        external_fields[s] += float(external_field_on_i)
        external_fields[r] += float(external_field_on_j)
        constant += float(self_connection)

    expression = - A * X[n]
    spin_expression = replace_bins_by_spins(expression, X, S, n)
    spin_expression = expand(spin_expression)
    ext_field = get_one_body_corr(spin_expression, S, n)
    constant_per_spin = get_constant(spin_expression, S)

    for n in range(n_nodes):
        external_fields[n] += float(ext_field)
        constant += float(constant_per_spin)

    self_connections = constant/n_nodes*np.ones((2*n_nodes, 1))### TODO check if this factor of two is correct here
    new_nodes = external_fields
    new_edges = np.concatenate([ couplings, self_connections ], axis = 0)
    n_edge = np.array([new_edges.shape[0]])
    new_senders = np.concatenate([senders, self_senders, self_senders], axis = -1)
    new_receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = -1)
    H_graph = j_graph._replace(nodes = new_nodes, edges = new_edges, n_edge = n_edge, senders = new_senders, receivers = new_receivers)
    return H_graph

def add_to_graph(ig, senders, receivers, couplings):

    for (s,r,e) in zip(senders, receivers, couplings):
        if(ig.are_connected(s,r)):
            existing_edge  = ig.es.select(_source=s, _target=r)[0]
            existing_edge["weight"] += e
        else:
            ig.add_edge(s,r,weight = e)
    return ig


def MaxCl(H_graph, B = 1.):

    num_X_nodes = H_graph.nodes.shape[0]
    external_fields_YX, senders_YX, receivers_YX, couplings_YX, constant_YX, num_Y_nodes, HA_graph = build_YX(H_graph, B)
    external_fields_XX, senders_XX, receivers_XX, couplings_XX, constant_XX = build_XX(H_graph, B)
    external_fields_YY, senders_YY, receivers_YY, couplings_YY, constant_YY = build_YY(H_graph, num_Y_nodes, B)
    C_external_fields, constant_C = build_C_term(H_graph, B)

    ### TODO add HB + HC
    HB1_graph = construct_graph(external_fields_YY, senders_YY, receivers_YY, couplings_YY, constant_YY, external_fields_YY.shape[0])
    HB2_graph = construct_graph(external_fields_XX, senders_XX, receivers_XX, couplings_XX, constant_XX, external_fields_XX.shape[0])
    HC_graph = construct_graph(C_external_fields, np.zeros((0,), dtype = np.int32), np.zeros((0,), dtype = np.int32), np.zeros((0,1)), constant_C, C_external_fields.shape[0])
    ### TODO add HB
    ### TODO add HC

    overall_external_fields = external_fields_YX
    overall_external_fields[0:num_Y_nodes] += external_fields_YY
    overall_external_fields[num_Y_nodes:] += external_fields_XX + C_external_fields

    overall_constant = constant_YY + constant_YX + constant_XX + constant_C
    self_senders = np.arange(0, num_X_nodes+num_Y_nodes)
    self_receivers = np.arange(0, num_X_nodes+num_Y_nodes)
    self_edges = overall_constant/(num_Y_nodes+ num_X_nodes)*np.ones((num_X_nodes+num_Y_nodes,1))

    ig = igraph.Graph(edges = [(s,r) for (s,r) in zip(senders_YX, receivers_YX)],edge_attrs={'weight': [e for e in couplings_YX]} )

    ig = add_to_graph(ig, senders_YY, receivers_YY, couplings_YY)
    ig = add_to_graph(ig, num_Y_nodes+senders_XX, num_Y_nodes+receivers_XX, couplings_XX)

    edge_arr = np.array(ig.get_edgelist())
    overall_senders = edge_arr[:,0]
    overall_receivers = edge_arr[:,1]
    overall_edges = np.array(ig.es["weight"])

    overall_senders = np.concatenate([overall_senders, self_senders, self_receivers], axis = 0)
    overall_receivers = np.concatenate([overall_receivers, self_receivers, self_senders], axis = 0)
    overall_edges = np.concatenate([overall_edges, self_edges, self_edges], axis = 0)

    n_node = np.array([num_Y_nodes+num_X_nodes])
    n_edge = np.array([overall_edges.shape[0]])
    glob = n_node

    new_H_graph = jraph.GraphsTuple(nodes = overall_external_fields, edges = overall_edges, senders = overall_senders, receivers = overall_receivers,
                                    globals = glob, n_node = n_node, n_edge = n_edge)

    # print("old graph size")
    # print(H_graph.nodes.shape, H_graph.edges.shape)
    # print("new graph_size")
    # print(new_H_graph.nodes.shape, new_H_graph.edges.shape)

    return new_H_graph, HA_graph, HB1_graph, HB2_graph, HC_graph

def build_C_term(H_graph, B):
    C = B

    n_nodes = H_graph.nodes.shape[0]
    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    expression = - C* X[i]
    spin_expression = replace_bins_by_spins(expression, X, S, i)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    constant = 0
    for i in range(n_nodes):
        external_fields[i] += float(external_field_on_i)
        constant += float(self_connection)

    return external_fields, constant

def crosscheck_YX(num_additional_nodes, num_nodes, A= 1):
    X = IndexedBase('X')
    Y = IndexedBase('Y')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    expression_Y = sum(2**i*X[i] for i in range(num_additional_nodes))
    expression_X = sum(X[i + num_additional_nodes] for i in range(num_nodes))

    full_expression = A*(expression_Y-expression_X)**2
    spin_expression = full_expression
    for i in range(num_additional_nodes + num_nodes):
        spin_expression = replace_bins_by_spins(spin_expression, X, S, i)

    spin_expression = expand(spin_expression)

    constant = float(get_constant(spin_expression, S))
    edges = []
    senders = []
    receivers = []
    overall_num_nodes = num_additional_nodes + num_nodes
    ext_fields = np.zeros((overall_num_nodes,1))

    for i in range(overall_num_nodes):
        external_field_on_i = get_one_body_corr(spin_expression, S, i)
        ext_fields[i] += float(external_field_on_i)

    for s in range(overall_num_nodes):
        for r in range(overall_num_nodes):
            if(s == r):
                constant += float(get_two_body_corr(spin_expression, S, s,r))
            else:
                J_ij = float(get_two_body_corr(spin_expression, S, s, r))
                senders.extend([s])
                receivers.extend([r])
                edges.extend([J_ij])

    edges = np.array(edges)
    edges = np.expand_dims(edges, axis = -1)
    senders = np.array(senders)
    receivers = np.array(receivers)

    #HA_graph = jraph.GraphsTuple(nodes = ext_fields, edges = edges)
    print("Symbolic")
    print(expand(full_expression))
    print(spin_expression)
    print(np.ravel(ext_fields))
    print(spin_expression)
    print([el for el in zip(senders, receivers, edges)])
    print(constant)

    def test(idx_y, x_idxs):
        y_spins = -1*np.ones((num_additional_nodes))
        y_spins[idx_y] = 1
        x_spins = -1*np.ones((num_nodes))
        for x_idx in x_idxs:
            x_spins[x_idx] = 1

        spins = np.concatenate([y_spins, x_spins])

        evaluated_expr = spin_expression.subs({S[i]: spins[i] for i in range(overall_num_nodes) })
        print(evaluated_expr)

        HA_graph = construct_graph(ext_fields, senders, receivers, edges, constant, overall_num_nodes)

        energy = jraphEnergy.compute_Energy_full_graph(HA_graph, np.expand_dims(spins, axis = -1))
        print(energy)

    HA_graph = construct_graph(ext_fields, senders, receivers, edges, constant, overall_num_nodes)
    return ext_fields, senders, receivers, edges, constant, HA_graph


def multinomial_theorem(vec_1, vec_2, A=1.):
    vec_concat = np.concatenate([vec_1, -vec_2], axis=0)

    res = np.tensordot(vec_concat[:, np.newaxis], vec_concat[np.newaxis, :], axes=[[1], [0]])
    overall_nodes = vec_concat.shape[0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    expression = 0.5 * A * X[i] * X[j]
    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = float(2 * get_two_body_corr(spin_expression, S, i, j))
    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((overall_nodes, 1))
    senders = []
    receivers = []
    edges = []
    constant = 0
    for i in range(overall_nodes):
        for j in range(overall_nodes):
            constant += float(self_connection) * (res[i, j] + res[j, i])
            external_fields[i, 0] += 2 * float(external_field_on_i) * res[i, j]
            external_fields[j, 0] += 2 * float(external_field_on_j) * res[j, i]
            if(i != j):
                senders.extend([j])
                receivers.extend([i])
                edges.extend([2*res[i,j]*J_ij])
            else:
                constant += res[j, i]*J_ij

    edges = np.array(edges)
    edges = np.expand_dims(edges, axis = -1)
    senders = np.array(senders)
    receivers = np.array(receivers)
    return external_fields, senders, receivers, edges, constant


def build_YX(H_graph, B):
    senders = H_graph.senders
    receivers = H_graph.receivers
    n_nodes = H_graph.nodes.shape[0]

    degree = np.zeros((H_graph.nodes.shape[0],1))
    ones_edges = np.ones_like(H_graph.edges)
    np.add.at(degree, H_graph.receivers, ones_edges)
    degree = np.squeeze(degree)
    degree_list = sorted(list(degree), key=lambda x: -x)

    DegreeContainer = Counter(degree_list).keys()
    DegreeOccurances = Counter(degree_list).values()

    max_degree = 0
    for degree, occ in zip(DegreeContainer, DegreeOccurances):
        if(degree <= occ):
            max_degree = degree
            break
        else:
            pass

    print(DegreeContainer)
    print(DegreeOccurances)
    print(max_degree)

    #max_degree = int(np.max(degree)) + 1
    num_additional_nodes = int(np.log2(max_degree)) + 1

    A = (max_degree + 2)*B
    C = B

    # A = 1
    # num_additional_nodes = 2
    # n_nodes = 2
    overall_num_nodes = n_nodes + num_additional_nodes

    Y = 2**np.arange(0,num_additional_nodes)
    X = np.ones((n_nodes))
    ### todo remove self connections from here
    ext_fields, senders, receivers, edges, constant = multinomial_theorem(Y, X, A = A)

    if(False):
        check_ext_fields, check_senders, check_receivers, check_edges, check_constant, check_HA_graph = crosscheck_YX(num_additional_nodes, n_nodes, A = A)

        print("constant", constant, check_constant)

        print("ext Fields")
        print(np.ravel(check_ext_fields))
        print(np.ravel(ext_fields))

        for s, cs, e, ce in zip(senders, check_senders, edges, check_edges):
            print("senders")
            print(s, cs)
            print("edges")
            print(e, ce)

    HA_graph = copy.deepcopy(construct_graph(ext_fields, senders, receivers, edges, constant, overall_num_nodes))

    return ext_fields, senders, receivers, edges, constant, num_additional_nodes, HA_graph

def construct_graph(external_fields, senders, receivers, edges, constant, num_nodes):
    self_senders = np.arange(0, num_nodes)
    self_receivers = np.arange(0, num_nodes)
    all_senders = np.concatenate([senders, self_senders, self_receivers])
    all_receivers = np.concatenate([receivers, self_receivers, self_senders])
    self_loops = constant / num_nodes * np.ones((num_nodes, 1))

    edges = np.concatenate([edges, self_loops, self_loops], axis=0)
    n_node = np.array([num_nodes])
    n_edge = np.array([all_senders.shape[0]])
    nodes = external_fields

    jgraph = jraph.GraphsTuple(nodes=nodes, edges=edges, senders=all_senders, receivers=all_receivers, n_node=n_node,
                                 n_edge=n_edge, globals=n_edge)
    return jgraph

def build_XX(H_graph, B):
    senders = H_graph.senders
    receivers = H_graph.receivers
    n_nodes = H_graph.nodes.shape[0]

    degree = np.zeros((H_graph.nodes.shape[0],1))
    ones_edges = np.ones_like(H_graph.edges)
    np.add.at(degree, H_graph.receivers, ones_edges)
    max_degree = int(np.max(degree)) + 1
    num_additional_nodes = int(np.log2(max_degree)) + 1

    A = (max_degree + 2)*B
    C = B
    n_edges = H_graph.edges.shape[0]
    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    n = symbols("n")
    expression = -0.5 * B * X[i] * X[j]

    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = 2*get_two_body_corr(spin_expression, S, i,j)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    couplings = np.zeros((n_edges,1))
    constant = 0
    self_senders = np.arange(0, n_nodes)
    self_receivers = self_senders

    for idx, (s, r) in enumerate(zip(senders, receivers)):
        J_sr = J_ij

        couplings[idx] += float(J_sr)
        external_fields[s] += float(external_field_on_i)
        external_fields[r] += float(external_field_on_j)
        constant += float(self_connection)

    # expression = -C * X[n]
    # spin_expression = replace_bins_by_spins(expression, X, S, n)
    # spin_expression = expand(spin_expression)
    # ext_field = get_one_body_corr(spin_expression, S, n)
    # constant_per_spin = get_constant(spin_expression, S)
    #
    # for n in range(n_nodes):
    #     external_fields[n] += float(ext_field)
    #     constant += float(constant_per_spin)

    new_nodes = external_fields
    return new_nodes, senders, receivers, couplings, constant

def build_YY(H_graph, num_additional_nodes, B):
    S_y = IndexedBase('S_y')

    sum_S_y_i =  sum([2 ** idx * (S_y[idx] + 1) / 2 for idx in range(num_additional_nodes)])
    HB1 = 0.5*B*expand((sum_S_y_i) * (sum_S_y_i - 1))

    senders_y = []
    receivers_y = []
    edges_y = []
    external_fields_y = np.zeros((num_additional_nodes, 1))
    self_connection_y = float(get_constant(HB1, S_y))

    for i in range(num_additional_nodes):
        external_field_on_i = get_one_body_corr(HB1, S_y, i)
        external_fields_y[i] += float(external_field_on_i)

    for i in range(num_additional_nodes):
        for j in range(num_additional_nodes):
            if(i != j):
                J_ij = float(get_two_body_corr(HB1, S_y, i, j))

                senders_y.extend([i])
                receivers_y.extend([j])
                edges_y.extend([J_ij])
            else:
                self_connection_y += float(get_two_body_corr(HB1, S_y, i, j))


    senders_y = np.array(senders_y)
    receivers_y = np.array(receivers_y)
    edges_y = np.array(edges_y)
    edges_y = np.expand_dims(edges_y, axis = -1)

    return external_fields_y, senders_y, receivers_y, edges_y, self_connection_y


def MaxCl_EGN(j_graph, beta = 4.):
    ### todo save factor of two by using directed graph
    ### TODO save factor of two by using map
    senders = j_graph.senders
    receivers = j_graph.receivers
    n_nodes = j_graph.nodes.shape[0]
    n_edges = j_graph.edges.shape[0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    n = symbols("n")
    expression = - 0.5*(beta + 1)* X[i] * X[j]

    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = 2*get_two_body_corr(spin_expression, S, i,j)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    expression_compl = beta/4 * X[i] * X[j]
    expression_compl = replace_bins_by_spins(expression_compl, X, S, i)
    expression_compl = replace_bins_by_spins(expression_compl, X, S, j)
    spin_expression_compl = expand(expression_compl)
    J_ij_compl = 2*get_two_body_corr(spin_expression_compl, S, i,j)

    external_field_on_i_compl = get_one_body_corr(spin_expression_compl, S, i)
    external_field_on_j_compl = get_one_body_corr(spin_expression_compl, S, j)
    self_connection_compl = get_constant(spin_expression_compl, S)

    external_fields = np.zeros((n_nodes,1))
    couplings = np.zeros(((n_nodes-1)*n_nodes,1))
    constant = 0
    self_senders = np.arange(0, n_nodes)
    self_receivers = self_senders

    edge_list = zip(senders, receivers)

    full_senders = []
    full_receivers = []

    index = 0
    for n_i in range(n_nodes):
        for n_j in range(n_nodes):
            edge = (n_i, n_j)
            #print(edge, index,(n_nodes-1)*n_nodes)
            if(edge in edge_list):
                ### tode get index
                couplings[index] += float(J_ij)
                external_fields[n_i] += float(external_field_on_i)
                external_fields[n_j] += float(external_field_on_j)
                constant += float(self_connection)
            if(n_i != n_j):
                couplings[index] += float(J_ij_compl)
                external_fields[n_i] += float(external_field_on_i_compl)
                external_fields[n_j] += float(external_field_on_j_compl)
                constant += float(self_connection_compl)
                index += 1
                full_senders.append(n_i)
                full_receivers.append(n_j)

    senders = np.array(full_senders)
    receivers = np.array(full_receivers)


    self_connections = constant/n_nodes*np.ones((2*n_nodes, 1))### TODO check if this factor of two is correct here
    new_nodes = external_fields
    new_edges = np.concatenate([ couplings, self_connections ], axis = 0)
    n_edge = np.array([new_edges.shape[0]])
    new_senders = np.concatenate([senders, self_senders, self_senders], axis = -1)
    new_receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = -1)
    H_graph = j_graph._replace(nodes = new_nodes, edges = new_edges, n_edge = n_edge, senders = new_senders, receivers = new_receivers)
    return H_graph

def MaxCl_check(j_graph, beta = 4.):
    senders = j_graph.senders
    receivers = j_graph.receivers
    edge_list = zip(senders, receivers)
    n_nodes = j_graph.nodes.shape[0]

    edges = []
    edge_weights = []
    for n_i in range(n_nodes):
        for n_j in range(n_nodes):
            edge = (n_i, n_j)
            edge_weight = 0
            if(edge in edge_list):
                edge_weight += -(1+beta)/2
            if(n_i != n_j):

                edge_weight += beta/2
                edges.append((n_i, n_j))

                edge_weights.append(edge_weight)

    new_senders = np.array([edge[0] for edge in edges])
    new_receivers = np.array([edge[1] for edge in edges])
    couplings = np.expand_dims(np.array(edge_weights), axis = -1)
    n_edge = np.array([couplings.shape[0]])
    new_j_graph = j_graph._replace(senders = new_senders, receivers = new_receivers, edges = couplings, n_edge = n_edge)

    return new_j_graph
### TODO test this with MaxCl bin formulation



if(__name__ == "__main__"):
    from utils import SympyHamiltonians
    from jraph_utils import utils as jutils
    import igraph as ig
    import time

    pass

