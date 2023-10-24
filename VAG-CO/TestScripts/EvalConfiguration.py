
import pickle

import matplotlib.pyplot as plt
import numpy as np
from TestScripts import EvalUtils


def evaluate_dataset(paths, mode = "OG", n_perm = 8, Ns = 1, n_test_graphs = 2):
    eval_mode_dict = {}
    eval_mode_dict["mode"] = mode
    eval_mode_dict["n_perm"] = n_perm
    eval_mode_dict["Ns"] = Ns
    eval_mode_dict["n_test_graphs"] = n_test_graphs

    EvalUtils.iterate_over_seeds(paths, eval_mode_dict)


def load_COLLAB_MVC():
    ### T == 0.05
    print("COLLAB Results")
    Nb = 30
    path1 = "/system/user/publicdata/sanokows/CombOpt/PPO/COLLAB_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_15000/406wlxqu"
    path2 = "/system/user/publicdata/sanokows/CombOpt/PPO/COLLAB_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_15000/7miys49r"
    path3 = "/system/user/publicdata/sanokows/CombOpt/PPO/COLLAB_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_15000/1o6vmn5y"

    EvalUtils.load_different_seeds_old([path1, path2, path3])


def load_TWITTER_MVC():
    print("TWITTER RESULTS")
    Nb = 30
    ### TODO replace paths
    path1 = "/system/user/publicdata/sanokows/CombOpt/PPO/TWITTER_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/930yzlna"
    path2 = "/system/user/publicdata/sanokows/CombOpt/PPO/TWITTER_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/9zi0hc7p"
    path3 = "/system/user/publicdata/sanokows/CombOpt/PPO/TWITTER_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/eldi6y97"
    EvalUtils.load_different_seeds_old([path1, path2, path3])


def load_IMDB_MVC():
    path1 = "/system/user/publicdata/sanokows/CombOpt/PPO/IMDB-BINARY_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_2000/5ov7yerz"
    path2 = "/system/user/publicdata/sanokows/CombOpt/PPO/IMDB-BINARY_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_2000/4lg1ylph"
    path3 = "/system/user/publicdata/sanokows/CombOpt/PPO/IMDB-BINARY_MVC_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_2000/oe1zthur"
    ps = [8]
    for p in ps:
        print("IMDB MVC p = ", p)
        EvalUtils.load_different_seeds_old([path1, path2, path3])



def load_MUTAG_MIS():
    path1 = "/system/user/publicdata/sanokows/CombOpt/PPO/MUTAG_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/438gwqn9"
    path2 = "/system/user/publicdata/sanokows/CombOpt/PPO/MUTAG_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/40n7g4yn"
    path3 = "/system/user/publicdata/sanokows/CombOpt/PPO/MUTAG_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/lf0u7iz6"

    ps = [8]
    for p in ps:
        print("MUTAG MIS p = ", p)
        EvalUtils.load_different_seeds_old([path1, path2, path3], n_perms=p)


def load_ENZYMES_MIS():
    print("ENZYMES MIS")
    path1 = "/system/user/publicdata/sanokows/CombOpt/PPO/ENZYMES_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_6000/s9cwuhq5"
    path2 = "/system/user/publicdata/sanokows/CombOpt/PPO/ENZYMES_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_6000/0t0zolzq"
    path3 = "/system/user/publicdata/sanokows/CombOpt/PPO/ENZYMES_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_6000/sjonepp9"

    ps = [8]
    for p in ps:
        print("ENZMES MIS p = ", p)
        EvalUtils.load_different_seeds_old([path1, path2, path3], n_perms=p)


def load_PROTEINS_MIS():
    path1 = "/system/user/publicdata/sanokows/CombOpt/PPO/PROTEINS_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_10000/foy2dj7d"
    path2 = "/system/user/publicdata/sanokows/CombOpt/PPO/PROTEINS_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_10000/9f6884xy"
    path3 = "/system/user/publicdata/sanokows/CombOpt/PPO/PROTEINS_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_10000/wv93vtfc"

    ps = [8]
    for p in ps:
        print("PROTEINS MIS p = ", p)
        EvalUtils.load_different_seeds_old([path1, path2, path3], n_perms=p)

def load_COLLAB_MIS():
    print("COLLAB MIS")
    path1 = "/system/user/publicdata/sanokows/CombOpt/PPO/COLLAB_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/mk2525ub"
    path2 = "/system/user/publicdata/sanokows/CombOpt/PPO/COLLAB_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/3wy7ser6"
    path3 = "/system/user/publicdata/sanokows/CombOpt/PPO/COLLAB_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/7biv1u2u"

    ps = [8]
    for p in ps:
        print("COLLAB MIS p = ", p)
        EvalUtils.load_different_seeds_old([path1, path2, path3], n_perms=p)

def load_IMDB_MIS():
    print("IMDB MIS")
    path1 = "/system/user/publicdata/sanokows/CombOpt/PPO/IMDB-BINARY_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_2000/0yq5nhi6"
    path2 = "/system/user/publicdata/sanokows/CombOpt/PPO/IMDB-BINARY_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_2000/07erg2mq"
    path3 = "/system/user/publicdata/sanokows/CombOpt/PPO/IMDB-BINARY_MIS_deeper_PPOGNN_sample_configuration_configur/N_anneal_=_4000/84fd17tj"

    ps = [8]
    for p in ps:
        print("IMDB MIS p = ", p)
        EvalUtils.load_different_seeds_old([path1, path2, path3], n_perms=p)


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
