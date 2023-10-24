
from torch.utils.data import DataLoader
from torch_geometric.datasets import TUDataset
import numpy as np
from jraph_utils import utils as jutils
from loadGraphDatasets.jraph_Dataloader import JraphSolutionDataset, JraphSolutionDataset_fromMemory

from loadGraphDatasets.loadTwitterGraph import TWITTER


def get_num_nodes_v1(pyg_graph):
    num_nodes = pyg_graph.x.shape[0]
    return num_nodes

def get_num_nodes_v2(pyg_graph):

    num_nodes = pyg_graph.num_nodes
    return num_nodes

def loader_func(global_reset, finished, loader, collator, dataset, shuffle = True):
    if(global_reset):
        loader = iter(DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collator))
        igraph, orig_igraph, gs = next(loader)
        finished = False
    else:
        try:
            finished = finished
            igraph, orig_igraph, gs = next(loader)
        except:
            finished = True
            loader = iter(DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collator))
            igraph, orig_igraph, gs = next(loader)

    return igraph, orig_igraph, gs, finished, loader

def get_Dataset(cfg, mode, shuffle_seed, dataset_name, indices = None):
    #if(dataset_name == "RRG_1000_k_=all" or dataset_name == "COLLAB"):
    if(True):
        return JraphSolutionDataset(cfg, mode=mode, seed=shuffle_seed, indices = indices)
    else:
        return JraphSolutionDataset_fromMemory(cfg, mode=mode, seed=shuffle_seed)



def init_Dataset(cfg, num_workers = 0, H_idx = 0, mode = "train"):
    path = cfg["Paths"]["work_path"]
    dataset_name = cfg["Ising_params"]["IsingMode"]
    shuffle_seed = cfg["Ising_params"]["shuffle_seed"]
    test_batch_size = cfg["Test_params"]["n_test_graphs"]
    random_node_features = cfg["Ising_params"]["n_rand_nodes"]

    np.random.seed(H_idx)

    Dataset_dict = {}

    if(mode == "train"):
        collate_data = lambda data: jutils.collate_igraph_normed(data)
        Dataset = get_Dataset(cfg, mode, shuffle_seed, dataset_name)
        jraph_loader = iter(DataLoader(Dataset, batch_size=1, shuffle = True,
                                           collate_fn=collate_data, num_workers=num_workers))
        Dataset_dict["loader_func"] = lambda x, y, loader: loader_func(x, y, loader, collate_data, Dataset)
        Dataset_dict["loader"] = jraph_loader
        Dataset_dict["num_graphs"] = len(Dataset)

    else:
        collate_data = lambda data: jutils.collate_igraph_normed(data)
        Dataset = JraphSolutionDataset_fromMemory(cfg, mode=mode, seed=shuffle_seed)
        print("overall dataset size", len(Dataset))
        H_idx_val_dataset_idxs = np.array_split(np.arange(0, len(Dataset)), test_batch_size)[H_idx]
        #if(dataset_name != "RRG_1000_k_=all" or dataset_name != "COLLAB"):
        if(True):
            H_idx_val_dataset = list(map(lambda x: Dataset[x], H_idx_val_dataset_idxs))
        else:
            H_idx_val_dataset = get_Dataset(cfg, mode, shuffle_seed, dataset_name, indices=H_idx_val_dataset_idxs)

        jraph_val_loader = iter(DataLoader(H_idx_val_dataset, batch_size=1, shuffle = False,
                                            collate_fn=collate_data, num_workers=num_workers))
        Dataset_dict["loader_func"] = lambda x,y, loader: loader_func(x,y, loader, collate_data, H_idx_val_dataset, shuffle = False)
        Dataset_dict["loader"] = jraph_val_loader
        Dataset_dict["num_graphs"] = len(H_idx_val_dataset)

    Dataset_dict["mean_energy"] = Dataset.mean_energy
    Dataset_dict["std_energy"] = Dataset.std_energy

    return Dataset_dict

def init_TUDataset(cfg, num_workers = 0, H_idx = 0, mode = "train"):
    path = cfg["Paths"]["work_path"]
    dataset_name = cfg["Ising_params"]["IsingMode"]
    shuffle_seed = cfg["Ising_params"]["shuffle_seed"]
    test_batch_size = cfg["Test_params"]["n_test_graphs"]
    random_node_features = cfg["Ising_params"]["n_rand_nodes"]

    dataset = TUDataset(root=f'{path}/loadGraphDatasets/tmp/{dataset_name}', name=dataset_name)

    ### TODO add flag for MaxCut
    if (dataset_name != "COLLAB" or cfg["Ising_params"]["EnergyFunction"] == "MaxCut"):
        full_dataset_len = len(dataset)
    else:
        full_dataset_len = 1000

    if(dataset_name == "COLLAB" or dataset_name == "IMDB-BINARY"):
        get_num_nodes_fuc = get_num_nodes_v2
    else:
        get_num_nodes_fuc = get_num_nodes_v1

    full_dataset_arganged = np.arange(0, full_dataset_len)

    np.random.seed(shuffle_seed)
    np.random.shuffle(full_dataset_arganged)
    np.random.seed(H_idx)

    if (cfg["Ising_params"]["EnergyFunction"] == "MIS"):
        ts = 0.6
        vs = 0.3
    else:
        ts = 0.8
        vs = 0.1

    train_dataset_len = int(ts*full_dataset_len)
    val_dataset_len = int(vs*full_dataset_len)

    train_dataset_idxs = full_dataset_arganged[0:train_dataset_len]
    pyg_train_dataset = dataset[train_dataset_idxs]

    pyg_loader = iter(DataLoader(pyg_train_dataset, batch_size=1, shuffle=True, collate_fn=jutils.collate_from_pyg_to_igraph))

    Dataset_dict = {}
    if (mode == "train"):
        return_pyg_loader_func = lambda x,y, loader: loader_func(x,y, loader, jutils.collate_from_pyg_to_igraph, pyg_train_dataset)
        Dataset_dict["loader_func"] = return_pyg_loader_func
        Dataset_dict["loader"] = pyg_loader
        Dataset_dict["num_graphs"] = len(pyg_train_dataset)
    elif(mode == "val"):
        collate_data = lambda data: jutils.collate_from_jgraph_to_igraph(data)
        val_dataset = JraphSolutionDataset(cfg, mode="val", seed=shuffle_seed)
        H_idx_val_dataset_idxs = np.array_split(np.arange(0, len(val_dataset)), test_batch_size)[H_idx]
        H_idx_val_dataset = list(map(lambda x: val_dataset[x], H_idx_val_dataset_idxs))
        jraph_val_loader = iter(DataLoader(H_idx_val_dataset, batch_size=1,
                                            collate_fn=collate_data, num_workers=num_workers))
        Dataset_dict["loader_func"] = lambda x,y, loader: loader_func(x,y, loader, collate_data, H_idx_val_dataset)
        Dataset_dict["loader"] = jraph_val_loader
        Dataset_dict["num_graphs"] = len(val_dataset)
    elif(mode == "test"):
        collate_data = lambda data: jutils.collate_from_jgraph_to_igraph(data)
        test_dataset = JraphSolutionDataset(cfg, mode="test", seed=shuffle_seed)
        H_idx_test_dataset_idxs = np.array_split(np.arange(0, len(test_dataset)), test_batch_size)[H_idx]
        H_idx_test_dataset = list(map(lambda x: test_dataset[x], H_idx_test_dataset_idxs))
        jraph_test_loader = iter(DataLoader(H_idx_test_dataset, batch_size=1,
                                            collate_fn=collate_data, num_workers=num_workers))
        Dataset_dict["loader_func"] = lambda x,y, loader: loader_func(x,y,jraph_test_loader, collate_data, H_idx_test_dataset)
        Dataset_dict["num_graphs"] = len(test_dataset)
        Dataset_dict["loader"] = jraph_test_loader

    val_dataset = JraphSolutionDataset(cfg, mode="val", seed=shuffle_seed)
    Dataset_dict["val_loader"] = val_dataset
    return Dataset_dict


### TODO add function here that takes a pyg_dataset and returns a igraph


def init_RRGDataset(self):
    ### TODO implement this
    pass


def init_TWITTERDataset(cfg, num_workers = 0, H_idx = 0, mode = "train"):
    collate_data = lambda data: jutils.collate_from_jgraph_to_igraph(data)

    shuffle_seed = cfg["Ising_params"]["shuffle_seed"]
    if(mode == "train"):
        dataset = TWITTER(cfg, mode=mode)
        num_graphs = len(dataset)
    else:
        dataset = JraphSolutionDataset(cfg, mode=mode, seed=shuffle_seed)
        num_graphs = len(dataset)
        test_batch_size = cfg["Test_params"]["n_test_graphs"]
        H_idx_test_dataset_idxs = np.array_split(np.arange(0, len(dataset)), test_batch_size)[H_idx]
        dataset = list(map(lambda x: dataset[x], H_idx_test_dataset_idxs))

    np.random.seed(H_idx)
    jraph_dataloader = iter(DataLoader(dataset, batch_size=1, shuffle=True,
                                       collate_fn=collate_data, num_workers=0))

    Dataset_dict = {}
    Dataset_dict["loader_func"] = lambda x,y, loader: loader_func(x,y, loader, collate_data, dataset)
    Dataset_dict["num_graphs"] = num_graphs
    Dataset_dict["loader"] = jraph_dataloader

    val_dataset = JraphSolutionDataset(cfg, mode="val", seed=shuffle_seed)
    Dataset_dict["val_loader"] = val_dataset
    return Dataset_dict


if(__name__ == "__main__"):
    list = [1,2,3,4]
    iterator = iter(list)

    pass