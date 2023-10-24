import os
import ray
from ray import tune
from trainPPO import trainPPO_configuration
from omegaconf import  OmegaConf
import argparse
import ast
import json
import numpy as np
import sys
import socket

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Switch ray into local mode for debugging')
parser.add_argument('--EnergyFunction', default='MIS', choices = ["MaxCut", "MIS", "MVC", "MaxCl", "WMIS", "MaxCl_compl", "MaxCl_compl_no_loops"], help='Define the EnergyFunction of the IsingModel')
parser.add_argument('--IsingMode', default='RB_iid_200', choices = ["BA_small","ExtToyMIS_[50, 100]_[2, 10]_0.7_0.8","ToyMIS_13_5_10","MUTAG", "ENZYMES", "PROTEINS", "TWITTER", "COLLAB", "IMDB-BINARY", "RRG_100_k_=all", "RRG_150_k_=all","RRG_200_k_=all", "RB_iid_small" , "RB_iid_200", "RB_iid_100", "ToyMIS_13_3_5", "ToyMIS_13_3_3", "ToyMIS_13_3_10"], help='Define the Training dataset')
parser.add_argument('--AnnealSchedule', default=['cosine_frac'], choices = ["linear", "cosine", 'cosine_frac', "frac"], help='Define the Annealing Schedule',  nargs = "+")
parser.add_argument('--num_hidden_neurons', default=64, type = int, help='Define number of hidden neurons')
parser.add_argument('--encode_edge_features', default=20, type = int, help='Define number of hidden neurons')
parser.add_argument('--encode_node_features', default=40, type = int, help='Define number of hidden neurons')
parser.add_argument('--prob_MLP', default=["[120,120]"], help='Define number of hidden neurons', nargs = "+")
parser.add_argument('--value_MLP', default=["[120,120,1]"], help='Define number of hidden neurons',  nargs = "+")
parser.add_argument('--message_nh', default=20, type = int, help='Define number of hidden neurons')
parser.add_argument('--n_val_workers', default=5, type = int, help='Define number of val workers')
parser.add_argument('--time_horizon', default=10, type = int, help='Define the time_horizon')
parser.add_argument('--temps', default=[0.1], type = float, help='Define gridsearch over Temperature', nargs = "+")
parser.add_argument('--N_warmup', default=400, type = int, help='Define gridsearch over Number of Annealing steps')
parser.add_argument('--N_anneal', default=[1000], type = int, help='Define gridsearch over Number of Annealing steps', nargs = "+")
parser.add_argument('--N_equil', default = 0, type = int, help='Define gridsearch over Number of Equil steps')
parser.add_argument('--T_min', default = 0., type = float, help='Define over Tmin')
parser.add_argument('--lrs', default=[0.5*10**-3], type = float, help='Define gridsearch over learning rate', nargs = "+")
parser.add_argument('--mini_Sb', default=15, type = int, help='Define miniSb')
parser.add_argument('--mini_Hb', default=15, type = int, help='Define miniHb')
parser.add_argument('--mini_Nb', default=10, type = int, help='Define miniNb')
parser.add_argument('--Nb', default=30, type = int, help='Define Sb')
parser.add_argument('--Hb', default=30, type = int, help='Define Hb')
parser.add_argument('--batch_epochs', default=2, type = int, help='Define batch epochs')
parser.add_argument('--seed', default=[123], type = int, help='Define dataset seed', nargs = "+")
parser.add_argument('--GPUs', '--names-list', nargs='+', default=["0"])
#parser.add_argument('--GPUs', default=[0], type = int, help='Define Nb', nargs = "+")
parser.add_argument('--n_rand_nodes', default=1, type = int, help='define node embedding size')
parser.add_argument('--node_embeddings', default="None", choices = ["None", "random", "eigenvectors",], help='Define node embedding type')
parser.add_argument('--save_mode', default="best", choices = ["best", "temperature"], help='Define save mode')
parser.add_argument('--padding_factor', default=1.2, type = float, help='Define node embedding type')
parser.add_argument('--lam', default=0.98, type = float, help='Define lambda')
parser.add_argument('--mov_avrg', default=-1, type = float, help='mov average alpha; -1 switches off the moving average, otherwise it should be between zero and one')
parser.add_argument('--n_GNN_layers', default=[3], type = int, help='num of GNN Layers', nargs = "+")
parser.add_argument('--n_sample_spins', default=5, type = int, help='number of spins that are sampled in each forward pass')
parser.add_argument('--project_name', default="", type = str, help='define project name for WANDB')
parser.add_argument('--ordering', default="BFS", type = str, help='define ordering of nodes')
parser.add_argument('--load_path', default="None", type = str, help='define checkpoint')
parser.add_argument('--masking', action='store_true')
parser.add_argument('--no-masking', dest='masking', action='store_false')
parser.set_defaults(masking=False)
parser.add_argument('--pruning', action='store_true')
parser.add_argument('--no-pruning', dest='pruning', action='store_false')
parser.set_defaults(pruning=True)
parser.add_argument('--self_loops', action='store_true', help = "use or not to use normalization with self loops")
parser.add_argument('--no-self_loops', dest='self_loops', action='store_false')
parser.set_defaults(self_loops=False)
parser.add_argument('--centrality', action='store_true', help = "use or not to use centrality for bfs")
parser.add_argument('--no-centrality', dest='centrality', action='store_false')
parser.set_defaults(centrality=False)
parser.add_argument('--GNN_backbone', default=["MPNN"], choices=["MPNN_simple", "MPNN", "MPNN_nonlinear", "GIN", "GIN_skip"], type = str, help='define GNN backbone', nargs = "+")
### TODO add sperate encode features for edges and nodes
args = parser.parse_args()

from unipath import Path
orig_path = os.getcwd()
path_dir = str(Path(orig_path).parent)

### TODO save argparse in a txt file!
def PPO_runs(policy_global_features, value_global_features):
    lam = float(np.round((1-1/args.time_horizon), decimals = 3))
    resources_per_trial = 1.
    devices = args.GPUs
    n_workers = int(len(devices)/resources_per_trial)

    nh = args.num_hidden_neurons

    if(len(args.GPUs) > 1):
        device_str = ""
        for idx, device in enumerate(devices):
            if (idx != len(devices) - 1):
                device_str += str(devices[idx]) + ","
            else:
                device_str += str(devices[idx])

        print(device_str)
    else:
        device_str = str(args.GPUs[0])

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"

    local_mode = args.debug
    ray.init(num_cpus=n_workers + 1, local_mode=local_mode)
    if local_mode:
        print("Init ray in local_mode!")

    if(args.prob_MLP):
        prob_MLP_list = ast.literal_eval(args.prob_MLP[0])
    if(args.value_MLP):
        value_MLP_list = ast.literal_eval(args.value_MLP[0])
    n_sample_spins = args.n_sample_spins
    prob_MLP_list.extend([2 ** n_sample_spins])

    flexible_config = {
                        "project": "deeper_PPO",
                        "Paths.load_path": None,
                        ### saving

                        ### learning params
                        "Train_params.seed": tune.grid_search(args.seed),
                        "Train_params.lr": tune.grid_search(args.lrs),
                        "Anneal_params.N_warmup": args.N_warmup,
                        "Anneal_params.N_anneal": tune.grid_search(args.N_anneal),
                        "Anneal_params.N_equil": args.N_equil,
                        "Anneal_params.Temperature": tune.grid_search(args.temps),
                        "Network_params.GNNs.mode": tune.grid_search(args.GNN_backbone),

                        ### Network params
                        "Network_params.GNNs.n_GNN_layers": tune.grid_search(args.n_GNN_layers),
                        "Network_params.GNNs.message_MLP_features": tune.grid_search([[args.message_nh, args.message_nh]]),
                        "Network_params.GNNs.node_MLP_features": tune.grid_search([[nh,nh]]),
                        "Network_params.GNNs.encode_node_features": tune.grid_search([[args.encode_node_features, args.encode_node_features]]),
                        "Network_params.GNNs.encode_edge_features": tune.grid_search([[args.encode_edge_features]]),
                        "Network_params.policy_MLP_features": tune.grid_search([prob_MLP_list]),
                        "Ising_params.sampled_sites": args.n_sample_spins,
                        "Network_params.value_MLP_features": tune.grid_search([value_MLP_list]),

                        "Network_params.GNNs.policy_global_features": tune.grid_search(policy_global_features),
                        "Network_params.GNNs.value_global_features": tune.grid_search(value_global_features),
                        "Anneal_params.T_min": args.T_min,
                        "Train_params.PPO.lam": lam,
                        "Ising_params.ordering": args.ordering,
                        "Anneal_params.schedule": tune.grid_search(args.AnnealSchedule)
                       }
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    #run_PPO_experiment_func = lambda flex_conf: run_PPO_experiment_hydra()
    #tune.run(run_PPO_experiment_hydra, config=flexible_config, resources_per_trial={"gpu": resources_per_trial}, log_to_file=False, loggers=None  )
    trainable_with_gpu = tune.with_resources(run_PPO_experiment_hydra, {"gpu": 1})
    tuner = tune.Tuner( trainable_with_gpu, param_space = flexible_config, run_config=ray.air.RunConfig(storage_path=f"{os.getcwd()}/ray_results", name=f"test_experiment_{dt_string}"))

    results = tuner.fit()

### TODO use different configs for different Datasets
def PPO_standard_settings(run_config):
    ### TODO add different standart settings dependent of the dataset
    OmegaConf.update(run_config, "Train_params.n_basis_states", args.Nb, merge=True)
    OmegaConf.update(run_config, "Train_params.H_batch_size", args.Hb, merge=True)
    if(args.debug):
        OmegaConf.update(run_config, "Test_params.n_test_graphs", 2, merge=True)
        OmegaConf.update(run_config, "Train_params.n_basis_states", 20, merge=True)
        OmegaConf.update(run_config, "Train_params.H_batch_size", 10, merge=True)
    else:
        OmegaConf.update(run_config, "Test_params.n_test_graphs", args.n_val_workers, merge=True)

    OmegaConf.update(run_config, "Train_params.batch_epochs", args.batch_epochs, merge=True)

    OmegaConf.update(run_config, "Train_params.PPO.mini_Nb", args.mini_Nb, merge=True)
    OmegaConf.update(run_config, "Train_params.PPO.mini_Hb", args.mini_Hb, merge=True)
    OmegaConf.update(run_config, "Train_params.PPO.mini_Sb", args.mini_Sb, merge=True)
    OmegaConf.update(run_config, "Train_params.PPO.time_horizon", args.time_horizon, merge=True)
    OmegaConf.update(run_config, "Train_params.PPO.mov_avrg", args.mov_avrg, merge=True)

    OmegaConf.update(run_config, "Ising_params.IsingMode", args.IsingMode, merge=True)
    OmegaConf.update(run_config, "Ising_params.n_rand_nodes", args.n_rand_nodes, merge=True) ## This has to be dividable by 2
    OmegaConf.update(run_config, "Ising_params.node_embedding_type", args.node_embeddings, merge=True)

    OmegaConf.update(run_config, "Train_params.lr_schedule", "cosine_restart", merge=True)

    OmegaConf.update(run_config, "Network_params.network_type", "GNN_sample_configuration", merge=True) ### GNN_reduce

    OmegaConf.update(run_config, "Ising_params.EnergyFunction", args.EnergyFunction, merge=True)

    OmegaConf.update(run_config, "Network_params.layer_norm", True, merge=True) ### GNN_reduce
    OmegaConf.update(run_config, "Network_params.edge_updates", False, merge=True) ### GNN_reduce ### GNN_reduce
    OmegaConf.update(run_config, "Save_settings.save_mode", args.save_mode, merge=True) ### GNN_reduce
    OmegaConf.update(run_config, "Ising_params.graph_padding_factor", args.padding_factor, merge=True) ### GNN_reduce
    OmegaConf.update(run_config, "Ising_params.self_loops", args.self_loops, merge=True)
    OmegaConf.update(run_config, "Ising_params.centrality", args.centrality, merge=True)
    OmegaConf.update(run_config, "Train_params.masking", args.masking, merge=True)
    OmegaConf.update(run_config, "Train_params.pruning", args.pruning, merge=True)

#@hydra.main(version_base=None, config_path= path + "/Experiment_configs/", config_name="HydraBaseConfig")
def run_PPO_experiment_hydra( flexible_config):
    run_config = OmegaConf.load(orig_path + "/Experiment_configs/HydraBaseConfig.yaml")
    PPO_standard_settings(run_config)

    ### TODO overwrite keys correctly
    OmegaConf.update(run_config, "Paths.path", orig_path + "/model_checkpoints", merge=True)
    OmegaConf.update(run_config, "Paths.work_path", path_dir + "/DatasetCreator", merge=True)
    OmegaConf.update(run_config, "Train_params.device", None, merge=True)
    OmegaConf.update(run_config, "Save_settings.save_params", True, merge=True)

    n_GNN_layers = flexible_config["Network_params.GNNs.n_GNN_layers"]
    global_policy = flexible_config["Network_params.GNNs.policy_global_features"]
    global_value = flexible_config["Network_params.GNNs.value_global_features"]
    nh = flexible_config["Network_params.GNNs.message_MLP_features"][0]
    lr = flexible_config["Train_params.lr"]
    T = flexible_config["Anneal_params.Temperature"]
    lam = flexible_config["Train_params.PPO.lam"]
    time_horizon = run_config["Train_params"]["PPO"]["time_horizon"]
    ordering = run_config["Ising_params"]["ordering"]
    embedding = run_config["Ising_params"]["node_embedding_type"]
    seed = flexible_config["Train_params.seed"]
    anneal_schedule = flexible_config["Anneal_params.schedule"]
    GNN_backbone = flexible_config["Network_params.GNNs.mode"]
    OmegaConf.update(run_config, "Ising_params.shuffle_seed", seed, merge=True)


    Experiments = f"seed_{seed}_ord_{ordering}_n_layers={n_GNN_layers}_tau_{time_horizon}_schedule_{args.AnnealSchedule}_embedding_{embedding}"
    for key in flexible_config:
        OmegaConf.update(run_config, key, flexible_config[key], merge = True)


    run_config["project"] = run_config["Ising_params"]["IsingMode"] + "_" + run_config["Ising_params"]["EnergyFunction"] + "_" + run_config["project"] +f"_self_loops_{args.self_loops}"+ args.project_name
    run_config["group"] = Experiments

    run_config["job_type"] = f"T_{T}_lr_{lr}_masking_{args.masking} = " + str(run_config["Anneal_params"]["N_anneal"])

    print("start run now")
    print(Experiments)

    print("overwritten cfg")
    print(run_config)

    IsingMode = run_config["Ising_params"]["IsingMode"]

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"


    run_config["group"] = Experiments + f"_GNN_backbone_{GNN_backbone}"
    PPORunner = trainPPO_configuration.HiVNAPPo()
    PPORunner.train(run_config, args.load_path)


def start_annealing_run():

    value_global_features = [True]
    policy_global_features = [True]

    PPO_runs(value_global_features= value_global_features,
             policy_global_features = policy_global_features)



if __name__ == "__main__":
    start_annealing_run()
