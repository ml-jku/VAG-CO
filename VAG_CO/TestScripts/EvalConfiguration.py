
import pickle

import matplotlib.pyplot as plt
import numpy as np
from . import EvalUtils


def evaluate_dataset(paths, mode = "OG", n_perm = 8, Ns = 1, n_test_graphs = 2):
    eval_mode_dict = {}
    eval_mode_dict["mode"] = mode
    eval_mode_dict["n_perm"] = n_perm
    eval_mode_dict["Ns"] = Ns
    eval_mode_dict["n_test_graphs"] = n_test_graphs

    EvalUtils.iterate_over_seeds(paths, eval_mode_dict)


if(__name__ == "__main__"):
    ### TWITTER runs
    import os

    #os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"#str(args.GPUs[0])

    #MUTAG_MIS()
    load_IMDB_MVC()
    load_TWITTER_MVC()
    load_COLLAB_MVC()
    load_ENZYMES_MIS()
    load_IMDB_MIS()
    load_MUTAG_MIS()
    load_COLLAB_MIS()
    load_PROTEINS_MIS()
    #plot_RRG_figure()
    #
    #load_RRG_AR()
    #load_and_plot_over_basis_states()
    #RRG_MIS()
    #COLLAB_MaxCl()
    #plot_RRG_figure()
    #Collab_MVC()
