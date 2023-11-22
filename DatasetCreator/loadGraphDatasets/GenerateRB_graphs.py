import numpy as np
import itertools
import random
import igraph as ig
from collections import Counter
from .RB_graphs import generate_xu_instances
from tqdm import tqdm

def generate_instance(n, k, r, p):
    '''
    n: number of cliques
    k: number of nodes in each clique
    a: log(k)/log(n)
    s: in each sampling iteration, the number of edges to be added
    iterations: how many iteration to sample
    return: the single-directed edges in numpy array form
    '''
    a = np.log(k) / np.log(n)
    v = k * n
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)
    parts = np.reshape(np.int64(range(v)), (n, k))
    nand_clauses = []

    for i in parts:
        nand_clauses += itertools.combinations(i, 2)
    edges = set()
    for _ in range(iterations):
        i, j = np.random.choice(n, 2, replace=False)
        all = set(itertools.product(parts[i, :], parts[j, :]))
        all -= edges
        edges |= set(random.sample(tuple(all), k=min(s, len(all))))

    nand_clauses += list(edges)
    clauses = np.array(nand_clauses)

    ordered_edge_list =[ (min([edge[0], edge[1]]), max([edge[0], edge[1]])) for edge in nand_clauses]

    # edges = nand_clauses
    # print(Counter(edges).keys())
    # print(Counter(edges).values())
    # print(len(Counter(edges)), len(edges))
    #
    # edges = ordered_edge_list
    # print(Counter(edges).keys())
    # print(Counter(edges).values())
    # print(len(Counter(edges)), len(edges))
    return Counter(ordered_edge_list)


def combinations(z):
    result = []
    for x in range(2, int(z ** 0.5) + 1):
        y = z // x
        if x * y == z:
            result.append((x, y))
            result.append((y,x))
    return result

def generateRB(mode = "val", n_train_graphs = 2000,seed = 123, RB_size = "200", parent = True, EnergyFunction = "MVC", solve = True, take_p = None, n_val_graphs = 500, val_time_limit = float("inf")):
    import pickle
    from unipath import Path
    import os
    from Gurobi import GurobiSolver
    from jraph_utils import utils as jutils

    if(take_p == None):
        dataset_name = f"RB_iid_{RB_size}"
    else:
        dataset_name = f"RB_iid_{RB_size}_p_{take_p}"

    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    if(mode == "val"):
        seed_int = 5
    elif(mode == "test"):
        seed_int = 4
    else:
        seed_int = 0

    np.random.seed(seed + seed_int)

    if(mode == "val"):
        n_graphs = n_val_graphs
    elif(mode == "train"):
        n_graphs = n_train_graphs
    else:
        if(take_p == None):
            n_graphs = 500
        else:
            n_graphs = 100


    solutions = {}
    solutions["Energies"] = []
    solutions["H_graphs"] = []
    solutions["gs_bins"] = []
    solutions["graph_sizes"] = []
    solutions["densities"] = []
    solutions["runtimes"] = []
    solutions["upperBoundEnergies"] = []
    solutions["p"] = []

    print(dataset_name, "is currently solved with gurobi")
    for i in tqdm(range(n_graphs)):
        while(True):
            if(RB_size == "200"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(20, 25)
                k = np.random.randint(9, 10)
            elif(RB_size == "500"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(30, 35)
                k = np.random.randint(15, 20)
            elif(RB_size == "100"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(9, 15)
                k = np.random.randint(8,11)
            elif(RB_size == "very_small"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(5,10)
                k = np.random.randint(5,10)
            elif RB_size == "small":
                min_n, max_n = 200, 300
                n = np.random.randint(20, 25)
                k = np.random.randint(5, 12)
            elif RB_size == "large":
                solve = False
                min_n, max_n = 800, 1200
                n = np.random.randint(40, 55)
                k = np.random.randint(20, 25)
            if(RB_size == "small" or RB_size == "large"):
                if(take_p == None):
                    p = np.random.uniform(0.3, 1.0)
                else:
                    p = take_p
            else:
                if (take_p == None):
                    p = np.random.uniform(0.25, 1.0)
                else:
                    p = take_p

            if(mode == "train"):
                time_limit = 0.1
            elif(RB_size != "small" or RB_size != "large"):
                time_limit = val_time_limit

            edges = generate_xu_instances.get_random_instance(n, k, p)

            g = ig.Graph([(edge[0], edge[1]) for edge in edges])
            isolated_nodes = [v.index for v in g.vs if v.degree() == 0]
            g.delete_vertices(isolated_nodes)
            num_nodes = g.vcount()
            if min_n <= num_nodes <= max_n:
                break

        H_graph = jutils.from_igraph_to_jgraph(g)

        if(solve):
            if(EnergyFunction == "MVC"):
                _, Energy, solution, runtime = GurobiSolver.solveMVC_as_MIP(H_graph, time_limit=time_limit)

            elif(EnergyFunction == "MIS"):
                _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph, time_limit=time_limit)
                print("solution", p, "Model ", -n, "Gurobi" , Energy)

                # import networkx as nx
                # import matplotlib.pyplot as plt

                # Create a graph
                # plt.title(f"p {p}")
                # G = nx.Graph()
                #
                # # Add nodes with attributes (0 or 1)
                # node_attributes = {idx: num for idx, num in enumerate(solution)}
                # G.add_nodes_from(node_attributes.keys())
                #
                # edges = [(s,r) for s,r in zip(H_graph.senders, H_graph.receivers)]
                # G.add_edges_from(edges)
                #
                # # Define colors based on node attributes
                # node_colors = ['red' if node_attributes[node] == 1 else 'blue' for node in G.nodes]
                #
                # # Draw the graph
                # pos = nx.spring_layout(G)  # You can use other layout algorithms as well
                # nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10,
                #         font_color='white')
                # # Show the plot
                # plt.show()


            elif(EnergyFunction == "MaxCl"):
                H_graph = jutils.from_igraph_to_jgraph(g)
                H_graph_compl = jutils.from_igraph_to_jgraph(g.complementer(loops=False))
                _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph_compl, time_limit=time_limit)

            elif(EnergyFunction == "MaxCut"):

                _, Energy, boundEnergy, solution, runtime = GurobiSolver.solveMaxCut(H_graph, time_limit=time_limit, bnb = False, verbose=False)

                solutions["upperBoundEnergies"].append(boundEnergy)
        else:
            if(EnergyFunction == "MIS"):
               Energy = -n
               solution = None
               runtime = None
            else:
                ValueError("Other Energy Functions that are not solved with gurobi are not implmented yet")




        solutions["Energies"].append(Energy)
        solutions["gs_bins"].append(solution)
        solutions["H_graphs"].append(H_graph)
        solutions["graph_sizes"].append(g.vcount())
        solutions["densities"].append(2*g.ecount()/(g.vcount()*(g.vcount()-1)))
        solutions["runtimes"].append(runtime)
        solutions["p"].append(p)


    newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
    pickle.dump(solutions, open(save_path, "wb"))

    return solutions["densities"], solutions["graph_sizes"]



def plot_graphs():
    ### TODO update this if needed
    from matplotlib import pyplot as plt
    modes = [ "val", "test", "train"]
    sizes = ["200", "500"]
    size2 = "very_small"
    plt.figure()
    for size in sizes:
        plt.figure()
        plt.title(size)
        for mode in modes:
            density, graph_sizes = generateRB(RB_size = size, mode = mode, solve = False)

            plt.plot(density, graph_sizes, "x",label = mode)
            # generateRB(RB_size = "200", mode = mode)
            # generateRB(RB_size = "500", mode = mode)
        plt.xlabel("density")
        plt.ylabel("num nodes")
        plt.legend()
        plt.show()

def create_and_solve_graphs_MIS(dataset_name , parent = True, EnergyFunction = "MIS", modes = [ "test", "train","val"], seeds = [123]):
    if("small" in dataset_name):
        ### Settings from EGN-ANneal
        sizes = ["small"]
        n_train_graphs = 2000
    elif ("large" in dataset_name):
        sizes = ["large"]
        n_train_graphs = 2000
    else:
        sizes = ["200"]
        n_train_graphs = 3000

    for seed in seeds:
        for size in sizes:
            for mode in modes:
                if(mode != "test"):
                    curr_p_list = [None]
                else:
                    curr_p_list = np.linspace(0.25, 1, num=10)

                for curr_p in curr_p_list:
                    generateRB(RB_size = size, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, n_train_graphs=n_train_graphs, take_p = curr_p)

def create_and_solve_graphs_MVC(parent = True, seeds = [123], sizes = ["200"], modes = ["test", "train", "val"]):
    EnergyFunction = "MVC"
    sizes = sizes
    for seed in seeds:
        for size in sizes:
            for mode in modes:

                if(mode != "test"):
                    curr_p_list = [None]
                else:
                    curr_p_list = np.linspace(0.25, 1, num=10)

                for curr_p in curr_p_list:

                    generateRB(RB_size = size, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, take_p = curr_p)

def create_and_solve_graphs_MaxCut(parent = True, mode = "val"):
    EnergyFunction = "MaxCut"
    modes = [ mode]
    sizes = ["100"]
    seeds = [123]

    if(mode != "test"):
        curr_p_list = [None]
    else:
        curr_p_list = np.linspace(0.25, 1, num=10)

    print("curr_p_list", curr_p_list)

    for curr_p in reversed(curr_p_list):
        for seed in seeds:
            for size in sizes:
                for mode in modes:
                    print("solve", curr_p, "in", mode)
                    if(mode == "test"):
                        val_time_limit = 20*60
                    elif(mode == "val"):
                        val_time_limit = 3*60
                    generateRB(RB_size = size, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, take_p = curr_p, n_val_graphs = 100, val_time_limit = val_time_limit)

if(__name__ == "__main__"):
    create_and_solve_graphs_MVC(modes = ["test"])
    #create_and_solve_graphs_MaxCut()
    pass
    #plot_graphs()
    #create_and_solve_graphs_MVC()

