import pickle
import os
import json
import sys

def create_save_path(project_name, Experiment, N_anneal, run, path, config):
    path_list = ["", project_name,Experiment, "N_anneal_=_{n}".format(n=N_anneal), run.name]
    run_path = path

    for path_el in path_list:
        run_path = os.path.join(run_path, path_el)
        if not os.path.exists(run_path):
            os.mkdir(run_path)

    with open(os.path.join(run_path, "config.pickle"), "wb") as file:
        pickle.dump(config, file)

    # Get the command-line arguments passed
    command_line_args = ' '.join(sys.argv[1:])

    # Define the filename for saving the arguments
    output_filename = os.path.join(run_path, 'command_line_args.txt')

    # Open the file in write mode
    with open(output_filename, 'w') as file:
        file.write(command_line_args)

    print(f"Command-line arguments saved to '{output_filename}'")

    return run_path

def save_best_model_params(best_params, opt_states, best_metric, run_path, add_string = ""):
    with open(os.path.join(run_path, f"best_{best_metric}_weights{add_string}.pickle"), "wb") as file:
        pickle.dump(best_params, file)

    with open(os.path.join(run_path, f"opt_state_{best_metric}{add_string}.pickle"), "wb") as file:
        pickle.dump(opt_states, file)

def save_best_model(best_params, best_metrics, best_energy_dict, run_path):
    pickle.dump(best_params, open(os.path.join(run_path, "best_model_weights.pickle"), "wb"))
    pickle.dump(best_metrics, open(os.path.join(run_path, "best_metrics.pickle"), "wb"))
    with open(os.path.join(run_path, "best_E_dict.json"), "w") as outfile:
        json.dump(best_energy_dict, outfile)

def save_model_at_T(params, opt_states, run_path, T):
    pickle.dump(params, open(os.path.join(run_path, "model_weights_at_{T:.3f}.pickle".format(T=T)), "wb"))
    pickle.dump(opt_states, open(os.path.join(run_path, "opt_state_{T:.3f}.pickle".format(T=T)), "wb"))

def save_curr_model(params, opt_states, run_path, epoch, T, config):
    with open(os.path.join(run_path, "curr_model_weights.pickle"), "wb") as f:
        pickle.dump(params, f)

    with open(os.path.join(run_path, "curr_state.pickle"), "wb") as f:
        pickle.dump(opt_states, f)

    end_model_dict = {"epoch": epoch, "T": T, "config": config}
    with open(os.path.join(run_path, "end_model_dict.pickle"), "wb") as outfile:
        pickle.dump(end_model_dict, outfile)


if("__main__" == __name__):
    pass