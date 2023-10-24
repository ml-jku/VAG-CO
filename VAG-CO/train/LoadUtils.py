import pickle



def load_model(path):
    base_path = path#"/publicdata/sanokows/CombOpt/PPO/SK_deeper_PPOGNN/N_anneal_=_20000/23cjk58i"

    param_path = base_path + "/best_val_rel_error_weights.pickle"
    opt_state_path = base_path + "/opt_state_val_rel_error.pickle"

    file = open(param_path, 'rb')
    params = pickle.load(file)

    file = open(opt_state_path, 'rb')
    opt_state = pickle.load(file)
    return params, opt_state

def checkpoint(path):
    base_path = path#"/publicdata/sanokows/CombOpt/PPO/SK_deeper_PPOGNN/N_anneal_=_20000/23cjk58i"

    param_path = base_path + "/curr_model_weights.pickle"
    opt_state_path = base_path + "/curr_state.pickle"
    dict_path = base_path + "/end_model_dict.pickle"

    with open(param_path, 'rb') as file:
        params = pickle.load(file)

    with open(opt_state_path, 'rb') as file:
        opt_state = pickle.load(file)

    with open(dict_path, "rb") as f:
        dict = pickle.load(f)

    return params, opt_state, dict["epoch"]