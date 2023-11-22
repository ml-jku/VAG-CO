import numpy as np
import igraph as ig
from tqdm import tqdm


def generateRB(mode = "val", n_train_graphs = 4000,seed = 123, m = 4, RB_size = "small", parent = True, EnergyFunction = "MaxCut", solve = True, n_val_graphs = 500, val_time_limit = float("inf")):
    import pickle
    from unipath import Path
    import os
    from DatasetCreator.Gurobi import GurobiSolver
    from ..jraph_utils import utils as jutils

    np.random.seed(seed)
    dataset_name = f"BA_{RB_size}"

    if(RB_size == "small"):
        n_min = 200
        n_max = 300
    else:
        raise ValueError("THis RB dataset is not implemented")

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
    elif(mode == "test"):
        n_graphs = n_val_graphs

    if (mode == "train"):
        time_limit = 0.01
    elif (mode == "val"):
        time_limit = 0.1
    elif(mode == "test"):
        time_limit = 0.1


    solutions = {}
    solutions["Energies"] = []
    solutions["H_graphs"] = []
    solutions["gs_bins"] = []
    solutions["graph_sizes"] = []
    solutions["densities"] = []
    solutions["runtimes"] = []
    solutions["upperBoundEnergies"] = []

    MC_value_list = []
    print(dataset_name, "is currently solved with gurobi")
    for i in tqdm(range(n_graphs)):

        n = np.random.randint(n_min, high = n_max)
        g = ig.Graph.Barabasi(n, m)
        H_graph = jutils.from_igraph_to_jgraph(g)

        _, Energy, boundEnergy, solution, runtime, MC_value = GurobiSolver.solveMaxCut(H_graph, time_limit=time_limit, bnb = False, verbose=False, thread_fraction = 0.35)

        solutions["upperBoundEnergies"].append(boundEnergy)

        solutions["Energies"].append(Energy)
        solutions["gs_bins"].append(solution)
        solutions["H_graphs"].append(H_graph)
        solutions["graph_sizes"].append(g.vcount())
        solutions["densities"].append(2*g.ecount()/(g.vcount()*(g.vcount()-1)))
        solutions["runtimes"].append(runtime)
        MC_value_list.append(MC_value)

    print("mean_E + 1", np.mean(np.array(MC_value_list)))

    if(solve):
        newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
        pickle.dump(solutions, open(save_path, "wb"))

    return solutions["densities"], solutions["graph_sizes"]



def create_and_solve_graphs_MaxCut(parent = True, seeds = [123, 124, 125]):
    EnergyFunction = "MaxCut"
    modes = [ "val","train", "test"]
    sizes = ["small"]

    for seed in seeds:
        for size in sizes:
            for mode in modes:
                generateRB(RB_size = size, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, n_val_graphs = 500)

def load_solutions(parent = True, dataset_name = "BA_small", mode = "test", EnergyFunction = "MaxCut", seed = 123):
    import os
    from unipath import Path
    import pickle

    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"

    with open(save_path, "rb") as f:
        res = pickle.load(f)

    Energies = np.array(res["Energies"])
    upper_Bound_Energies = res["upperBoundEnergies"]
    H_graphs = res["H_graphs"]
    gs_bins_per_graph = res["gs_bins"]

    n_nodes = np.array([H_graph.n_node[0] for H_graph in H_graphs])
    n_edges = np.array([H_graph.n_edge[0] for H_graph in H_graphs])

    MC_results_list = []
    for  H_graph, gs_bins in zip(H_graphs, gs_bins_per_graph):
        gs_spins = 2 * gs_bins - 1
        receivers = H_graph.receivers
        senders = H_graph.senders
        MC_result = 0
        for s, r in zip(senders, receivers):
            if (s != r):
                MC_result += (1-gs_spins[ s]*gs_spins[ r])/4
        MC_results_list.append(MC_result)

    MC_results_arr = np.array(MC_results_list)
    print("MC_results arr", np.mean(MC_results_arr))

    MC_value = n_edges/4 - Energies/2
    print("MC_value", np.mean(MC_value))
    print("finished")

if(__name__ == "__main__"):
    import os
    import socket
    hostname = socket.gethostname()
    print(os.environ["GRB_LICENSE_FILE"])
    os.environ["GRB_LICENSE_FILE"] = f"/system/user/sanokows/gurobi_{hostname}.lic"
    create_and_solve_graphs_MaxCut(seeds=[123])
    load_solutions()
    #create_and_solve_graphs_MaxCut()
    pass
    #plot_graphs()
    #create_and_solve_graphs_MVC()
