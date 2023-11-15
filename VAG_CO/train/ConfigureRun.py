import jax

from Samplers.PPO_sampler_vectorised import PPOSampler
from Samplers.PPO_sampler_ReplayBuffer import PPOSampler as PPOSampler_ReplayBuffer
import os
from DataLoader import General1DGridDataset, GeneralPlaceholder
from train.AnnealingSchedules import Schedules

def configure_run(cfg, sparse_graphs = False):

    if(cfg.Train_params.device != None):
        print("device set to " + cfg.Train_params.device)
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.Train_params.device
        print("Device is supposed to be ", cfg.Train_params.device)

    else:
        print("Device is None")
    print("jax devices", jax.devices())
    ### Flags
    (x, y) = (cfg.Ising_params["x"], cfg.Ising_params["y"])

    IsingMode = cfg.Ising_params["IsingMode"]
    if(IsingMode == "MaxCutSparse"):
        sparse_graphs = True

    N_warmup = cfg.Anneal_params["N_warmup"]
    N_anneal = cfg.Anneal_params["N_anneal"]
    N_equil = cfg.Anneal_params["N_equil"]
    batch_epochs = cfg.Train_params["batch_epochs"]
    anneal_schedule = cfg.Anneal_params["schedule"]
    EnergyFunction = cfg.Ising_params.EnergyFunction

    ICGenerator = None
    GraphDataloader = GeneralPlaceholder(cfg)


    epoch_dict = {"N_warmup": N_warmup, "N_anneal": N_anneal, "N_equil": N_equil, "batch_epochs": batch_epochs}

    if(cfg.TrainMode == "PPO"):
        RNN = PPOSampler(GraphDataloader, epoch_dict, cfg, sparse_graphs = sparse_graphs)
    else:
        ValueError("This TrainMode is not implemented yet")

    if(anneal_schedule == "linear"):
        anneal_scheduler = Schedules.linear_decrease
    elif(anneal_schedule == "triangular"):
        anneal_scheduler = Schedules.triangular_schedule
    elif(anneal_schedule == "hyperbel"):
        anneal_scheduler = Schedules.hyperbel_schedule
    elif(anneal_schedule == "cosine"):
        anneal_scheduler = Schedules.cosine
    else:
        ValueError("Annealing Schudule is not valid.")

    return RNN, ICGenerator, anneal_scheduler

def configure_run_ReplayBuffer(cfg, sparse_graphs = False):

    if(cfg.Train_params.device != None):
        print("device set to " + cfg.Train_params.device)
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.Train_params.device
    else:
        print(jax.devices())
        print("Device is None")
    ### Flags

    IsingMode = cfg.Ising_params["IsingMode"]
    if(IsingMode == "MaxCutSparse"):
        sparse_graphs = True

    N_warmup = cfg.Anneal_params["N_warmup"]
    N_anneal = cfg.Anneal_params["N_anneal"]
    N_equil = cfg.Anneal_params["N_equil"]
    batch_epochs = cfg.Train_params["batch_epochs"]
    anneal_schedule = cfg.Anneal_params["schedule"]
    EnergyFunction = cfg.Ising_params.EnergyFunction

    ICGenerator = None
    GraphDataloader = GeneralPlaceholder(cfg)


    epoch_dict = {"N_warmup": N_warmup, "N_anneal": N_anneal, "N_equil": N_equil, "batch_epochs": batch_epochs}

    if(cfg.TrainMode == "PPO"):
        RNN = PPOSampler_ReplayBuffer(GraphDataloader, epoch_dict, cfg, sparse_graphs = sparse_graphs)
    else:
        ValueError("This TrainMode is not implemented yet")

    if(anneal_schedule == "linear"):
        anneal_scheduler = Schedules.linear_decrease
    elif(anneal_schedule == "triangular"):
        anneal_scheduler = Schedules.triangular_schedule
    elif(anneal_schedule == "hyperbel"):
        anneal_scheduler = Schedules.hyperbel_schedule
    elif(anneal_schedule == "cosine"):
        anneal_scheduler = Schedules.cosine
    elif(anneal_schedule == "cosine_frac"):
        anneal_scheduler = Schedules.cosine_frac
    elif (anneal_schedule == "frac"):
        anneal_scheduler = Schedules.fractional_schedule
    else:
        ValueError("Annealing Schudule is not valid.")

    return RNN, ICGenerator, anneal_scheduler
