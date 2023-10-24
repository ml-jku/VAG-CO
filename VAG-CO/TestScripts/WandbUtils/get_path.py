import os
import os.path
import wandb
import time
from Experiment_configs.BaseConfigs import base_config
base_path = base_config["path"] + "/"

def get_wandb_hash(entity: str, project: str, filter =  None):

    api = wandb.Api(timeout=30)
    runs = api.runs(f"{entity}/{project}")

    run_dict = {}
    for run in runs:
        run_state = run.state
        if(run_state == "finished" or filter != None):
            if(filter == None):
                N_anneal = run.config["N_anneal"]
                run_dict[str(run.id)] = N_anneal
            else:
                for key in filter:
                    if(filter[key] == run.config[key]):
                        N_anneal = run.config["N_anneal"]
                        run_dict[str(run.id)] = N_anneal


    return run_dict

def return_path_to_model(project_name ,base_path = base_path, filter = None):
    run_dicts = {}

    run_id_dict = get_wandb_hash(entity='lattice', project=project_name)

    for run_id in run_id_dict:

        for root, dir, filelist in os.walk(base_path + project_name):
            if run_id in root:
                if(filter == None):
                    run_dicts[run_id] = root
                elif(run_id_dict[run_id] == filter):
                    run_dicts[run_id] = root

    return run_dicts

def return_warmup_run(project_name, base_path = base_path, filter = None):
    run_dicts = {}

    run_id_dict = get_wandb_hash(entity='lattice', project=project_name, filter = filter)
    print(run_id_dict)
    for run_id in run_id_dict:

        for root, dir, filelist in os.walk(base_path + project_name):
            if run_id in root:
                run_dicts[run_id] = root

    if(len(run_dicts.keys()) == 1):
        for key in run_dicts:
            return "/"+os.path.relpath(run_dicts[key], base_path)
    elif(len(run_dicts.keys()) >= 1):
        ValueError("More than one warmup run was found")
    else:
        ValueError("No warmup run was found")

def return_final_run(project_name, N_anneal, base_path = base_path):
    run_dicts = {}

    run_id_dict = get_wandb_hash(entity='lattice', project=project_name)

    for run_id in run_id_dict:
        if(run_id_dict[run_id] == N_anneal):
            for root, dir, filelist in os.walk(base_path + project_name):
                if run_id in root:
                    run_dicts[run_id] = root


    if(len(run_dicts.keys()) == 1):
        for key in run_dicts:
            return "/"+os.path.relpath(run_dicts[key], base_path)
    elif(len(run_dicts.keys()) >= 1):
        ValueError("More than one warmup run was found")
    else:
        ValueError("No warmup run was found")


def return_path_to_id(project_name, run_id, base_path = "/system/user/publicdata/sanokows/Quantum/models/debug_mode/"):

    for root, dir, filelist in os.walk(base_path + project_name):
        if run_id in root:
            run_path = root
            break

    return run_path


if(__name__ == "__main__"):
    pass
