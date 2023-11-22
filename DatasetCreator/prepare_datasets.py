from loadGraphDatasets import solveTUDatasets, saveHamiltonianGraphs, GenerateRB_graphs, GenerateRRGs, GenerateBAGraphs

import argparse

### TODO add arguments to sepcify whether normalized graphs should be generated
TU_datasets = ["MUTAG", "TWITTER", "ENZYMES", "PROTEINS", "IMDB-BINARY", "COLLAB"]
RB_datasets = ["RB_iid_200", "RB_iid_100", "RB_iid_small", "RB_iid_large"]
BA_datasets = ["BA_small"]
RRG_datasets = ['RRG_100_k_=all']
dataset_choices = TU_datasets + RB_datasets + RRG_datasets + BA_datasets
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='RB_iid_large', choices = dataset_choices, help='Define the dataset')
parser.add_argument('--problem', default='MIS', choices = ["MIS", "MVC", "MaxCl", "MaxCut"], help='Define the CO problem')
parser.add_argument('--seed', default=[123], type = int, help='Define dataset seed', nargs = "+")
parser.add_argument('--modes', default=["val", "train", "test"], type = str, help='Define dataset split', nargs = "+")
parser.add_argument('--licence', default="yes", type = str, help='use licence or not')
parser.add_argument('--IsingFormulation', default="No", type = str, help='')
args = parser.parse_args()

### TODO implement BA MaxCl, TODO implement BA large

if(__name__ == "__main__"):
    ### TODO first set up your gurobi licence before running this code. Otherwise large CO Problem Instances cannot be solved by gurobi!
    import os
    import socket
    if(args.licence == "yes"):
        hostname = socket.gethostname()
        print(os.environ["GRB_LICENSE_FILE"])
        os.environ["GRB_LICENSE_FILE"] = f"/system/user/sanokows/gurobi_{hostname}.lic"
    ### TODO first set up your gurobi licence before running this code. Otherwise large CO Problem Instances cannot be solved by gurobi!
    else:
        pass

    print("Solving dataset with gurobi")
    if(args.dataset in TU_datasets):
        solveTUDatasets.solve_datasets(args.dataset, args.problem, parent = False, seeds = args.seed)
    elif(args.dataset in RB_datasets):
        if(args.problem == "MIS"):
            GenerateRB_graphs.create_and_solve_graphs_MIS(args.dataset, parent = False, EnergyFunction= args.problem, modes = args.modes, seeds = args.seed)
        else:
            if(args.dataset == "RB_iid_100"):
                sizes = ["100"]
            else:
                sizes = ["200"]
            GenerateRB_graphs.create_and_solve_graphs_MVC(parent = False, sizes = sizes, seeds = args.seed)
    elif(args.dataset in RRG_datasets):
        GenerateRRGs.make_dataset(args.dataset,parent = False, seeds = args.seed, modes = args.modes)
    elif(args.dataset in BA_datasets):
        GenerateBAGraphs.create_and_solve_graphs_MaxCut(seeds=args.seed,  parent = False)
    else:
        raise ValueError("Dataset is not defined")

    if(args.IsingFormulation == "yes"):
        print("Translating dataet into spin formulation")
        saveHamiltonianGraphs.solve(args.dataset, args.problem, seeds=args.seed)
