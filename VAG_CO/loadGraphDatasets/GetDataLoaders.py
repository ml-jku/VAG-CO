from torch.utils.data import DataLoader
import numpy as np
from jraph_utils import utils as jutils
from .jraph_Dataloader import JraphSolutionDataset, JraphSolutionDataset_fromMemory



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




if(__name__ == "__main__"):
    list = [1,2,3,4]
    iterator = iter(list)

    pass