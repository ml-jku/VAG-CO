import os
import ray
from ray import tune
import argparse
from train import TrainMeanField
from unipath import Path

### TODO test relaxed code
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Switch ray into local mode for debugging')
parser.add_argument('--EnergyFunction', default='MIS', choices = ["MaxCut", "MIS", "MVC", "MaxCl", "WMIS"], help='Define the EnergyFunction of the IsingModel')
parser.add_argument('--IsingMode', default='RB_iid_200', choices = ["RRG_200_k_=all", "ToyMIS_13_5_10","MUTAG", "ENZYMES", "PROTEINS", "TWITTER", "COLLAB", "IMDB-BINARY", "RRG_100_k_=all","RRG_1000_k_=all" , "RB_iid_200", "RB_iid_100", "ToyMIS_13_3_5", "ToyMIS_13_3_3", "ToyMIS_13_3_10"], help='Define the Training dataset')
parser.add_argument('--AnnealSchedule', default='linear', choices = ["linear", "cosine"], help='Define the Annealing Schedule')
parser.add_argument('--temps', default=[0.], type = float, help='Define gridsearch over Temperature', nargs = "+")
parser.add_argument('--N_warmup', default=0, type = int, help='Define gridsearch over Number of Annealing steps')
parser.add_argument('--N_anneal', default=[2000], type = int, help='Define gridsearch over Number of Annealing steps', nargs = "+")
parser.add_argument('--N_equil', default = 0, type = int, help='Define gridsearch over Number of Equil steps')
parser.add_argument('--lrs', default=[5e-5], type = float, help='Define gridsearch over learning rate', nargs = "+")
parser.add_argument('--seed', default=[123], type = int, help='Define dataset seed', nargs = "+")
parser.add_argument('--GPUs', default=["0"], type = str, help='Define Nb', nargs = "+")
parser.add_argument('--n_rand_nodes', default=1, type = int, help='define node embedding size')
parser.add_argument('--n_GNN_layers', default=[8], type = int, help='num of GNN Layers', nargs = "+")
parser.add_argument('--relaxed', action='store_true')
parser.add_argument('--no-relaxed', dest='relaxed', action='store_false')
parser.set_defaults(relaxed=False)
args = parser.parse_args()

def meanfield_run():
    resources_per_trial = 1.
    devices = args.GPUs
    n_workers = int(len(devices)/resources_per_trial)

    device_str = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            device_str += str(devices[idx]) + ","
        else:
            device_str += str(devices[idx])

    print(device_str)

    if(len(args.GPUs) > 1):
        device_str = ""
        for idx, device in enumerate(devices):
            if (idx != len(devices) - 1):
                device_str += str(devices[idx]) + ","
            else:
                device_str += str(devices[idx])

        print(device_str, type(device_str))
    else:
        device_str = str(args.GPUs[0])

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str


    local_mode = args.debug
    ray.init(num_cpus=n_workers + 1, local_mode=local_mode)
    if local_mode:
        print("Init ray in local_mode!")


    p = Path(os.getcwd())
    working_directory_path = str(p.parent)


    flexible_config = {
                        "dataset_name": args.IsingMode,
                        "problem_name": args.EnergyFunction,
                        "jit": True,
                        "wandb": True,

                        "seed": tune.grid_search(args.seed),
                        "lr": tune.grid_search(args.lrs),

                        "random_node_features": True,
                        "n_random_node_features": 6,
                        "relaxed": args.relaxed,
                        "T_max": tune.grid_search(args.temps),
                        "N_warmup": args.N_warmup,
                        "N_anneal": tune.grid_search(args.N_anneal),
                        "N_equil": args.N_equil,

                        "n_features_list_prob": [64, 64, 2],
                        "n_features_list_nodes": [64, 64],
                        "n_features_list_edges": [64, 64],
                        "n_features_list_messages": [64, 64],
                        "n_features_list_encode": [64],
                        "n_features_list_decode": [64],
                        "n_message_passes": tune.grid_search(args.n_GNN_layers),
                        "message_passing_weight_tied": False,
                        "linear_message_passing": True,
                        "working_directory_path": working_directory_path

                       }



    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    trainable_with_gpu = tune.with_resources(run, {"gpu": 1})
    tuner = tune.Tuner( trainable_with_gpu, param_space = flexible_config, run_config=ray.air.RunConfig(storage_path=f"{os.getcwd()}/ray_results", name=f"test_experiment_{dt_string}"))

    results = tuner.fit()


def run( flexible_config):

    config = {
        "dataset_name": "RRG_100_k_=all",
        "problem_name": "MIS",
        "jit": True,
        "wandb": True,

        "seed": 123,
        "lr": 1e-4,
        "batch_size": 32, # H
        "N_basis_states": 30, # n_s

        "random_node_features": True,
        "n_random_node_features": 6,
        "relaxed": True,

        "T_max": 0.05,
        "N_warmup": 0,
        "N_anneal": 2000,
        "N_equil": 0,
        "stop_epochs": 400,

        "n_features_list_prob": [64, 64, 2],
        "n_features_list_nodes": [64, 64],
        "n_features_list_edges": [64, 64],
        "n_features_list_messages": [64, 64],
        "n_features_list_encode": [64],
        "n_features_list_decode": [64],
        "n_message_passes": 8,
        "message_passing_weight_tied": False,
        "linear_message_passing": True,
        "working_directory_path": None
    }


    for key in flexible_config:
        if(key in config.keys()):
            config[key] = flexible_config[key]
        else:
            raise ValueError("key does not exist")

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".92"

    wandb_project = f"MeanField__{config['dataset_name']}_{config['problem_name']}_relaxed_{config['relaxed']}"
    if config['T_max'] > 0.:
        wandb_group = f"{config['seed']}_LMP_T_{config['T_max']}_warmup_{config['N_warmup']}_anneal_{config['N_anneal']}_MPasses_{config['n_message_passes']}_n_rnf_{config['n_random_node_features']}"
    else:
        wandb_group = f"{config['seed']}_LMP_T_{config['T_max']}_anneal_{config['N_anneal']}_MPasses_{config['n_message_passes']}"

    wandb_run = f"lr_{config['lr']}_batchsize_{config['batch_size']}_basisstates_{config['N_basis_states']}_rnf_{config['random_node_features']}"

    train = TrainMeanField(config, wandb_project=wandb_project, wandb_group=wandb_group, wandb_run=wandb_run)

    train.train()





if __name__ == "__main__":
    meanfield_run()