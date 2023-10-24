import wandb
import numpy as np
from matplotlib import pyplot as plt
import jax.numpy as jnp


def wandb_metric_init():
    wandb.define_metric("train/epoch")
    wandb.define_metric("best_test/epoch")
    wandb.define_metric("mean_test/epoch")
    wandb.define_metric("prob/epoch")
    wandb.define_metric("TestScripts/Temperature")
    # define which metrics will be plotted against it
    wandb.define_metric("train/*", step_metric="train/epoch")

    wandb.define_metric("best_test/*", step_metric="best_test/epoch")
    wandb.define_metric("mean_test/*", step_metric="mean_test/epoch")
    wandb.define_metric("TestScripts/*", step_metric="TestScripts/Temperature")
    wandb.define_metric("prob/*", step_metric="prob/epoch")

def log_train_metrics(RNN, metrics, epoch, batch_epochs):
    batched_mean_E, batched_var_E, batched_mean_Entropy, batched_cost_E = metrics

    train_metric_dict = {"train/epoch": epoch, "train/Mean_E": batched_mean_E, "train/var_E": batched_var_E,
                         "train/FreeEnergy": batched_mean_E - RNN.T * batched_mean_Entropy,
                         "train/mean_Entropy": batched_mean_Entropy, "train/cost_E": batched_cost_E,
                         "train/Temperature": RNN.T}
    wandb.log(train_metric_dict)

    wandb.log({"lr": RNN.lr_scheduler(epoch * batch_epochs)})


def log_end_metrics(eval_model, RNN, params, RNN_test_key, gt_data, train_stuff):
    (test_O_graphs, test_H_graphs, gt_Energies, ground_states) = gt_data
    (epoch, N_warmup, N_anneal, config) = train_stuff
    wandb.define_metric("result/N_anneal")
    # define which metrics will be plotted against it
    wandb.define_metric("result/E_res_over_N_anneal", step_metric="result/N_anneal")
    wandb.define_metric("result/E_rel_over_N_anneal", step_metric="result/N_anneal")

    eval_metrics, batched_mean_Entropy, batched_mean_E, RNN_test_key = eval_model(RNN.eval_model, RNN, params, test_O_graphs,
                                                                                  test_H_graphs, gt_Energies, epoch, RNN_test_key)

    E_rel = eval_metrics["mean_test/mean_E_rel"]
    E_res = eval_metrics["mean_test/mean_E_res"]

    wandb.log({"result/N_anneal": N_anneal, "result/E_res_over_N_anneal": E_res, "result/E_rel_over_N_anneal": E_rel})

def log_eval_metrics(eval_model, RNN, params, RNN_test_key, gt_data, train_stuff):
    (test_O_graphs, test_H_graphs, gt_Energies, ground_states) = gt_data
    (epoch, N_warmup, N_anneal, config) = train_stuff
    eval_metrics, batched_mean_Entropy, batched_mean_E, RNN_test_key = eval_model(RNN.eval_model, RNN, params,
                                                                                  test_O_graphs, test_H_graphs,
                                                                                  gt_Energies, epoch, RNN_test_key)

    batched_E_rel = eval_metrics["best_test/batched_E_rel"]
    test_mean_E = eval_metrics["best_test/mean_E_rel"]

    batched_log_probs1 = jnp.expand_dims(RNN.batched_state_probability(params, ground_states, test_H_graphs), axis=1)
    batched_log_probs2 = jnp.expand_dims(RNN.batched_state_probability(params, -ground_states + 1, test_H_graphs),
                                         axis=1)

    batched_log_probs = jnp.concatenate([batched_log_probs1, batched_log_probs2], axis=-1)
    batched_log_probs = jnp.sum(jnp.exp(batched_log_probs), axis=-1)

    log_dict = {"prob/epoch": epoch, "prob/mean_probs": jnp.mean(batched_log_probs),
                "prob/min_prob": jnp.min(batched_log_probs),
                "prob/max_prob": jnp.max(batched_log_probs), "prob/median_prob": np.median(np.array(batched_log_probs))}

    if config['IsingMode'] == 'MaxCut':
        n_edges = test_O_graphs[2].edges.squeeze(axis=2).sum(1)

        gt_max_cuts = n_edges / 2 - 1 / 2 * gt_Energies
        max_cuts = n_edges / 2 - 1 / 2 * batched_mean_E

        best_E = eval_metrics["best_test/H_batched_best_E"]

        print(n_edges.shape, best_E.shape)
        best_max_cuts = n_edges / 2 - 1 / 2 * best_E

        log_dict["TestScripts/max_cut_ratio"] = max_cuts / gt_max_cuts
        log_dict["mean_test/best_max_cut_ratio"] = 1 - jnp.mean(best_max_cuts / gt_max_cuts)
        log_dict["mean_test/mean_max_cut_ratio"] = 1 - jnp.mean(max_cuts / gt_max_cuts)

    wandb.log(log_dict)

    if (epoch > N_warmup):
        log_dict = {"TestScripts/Temperature": RNN.T, "TestScripts/mean_E_rel": test_mean_E}
        wandb.log(log_dict)

    return test_mean_E, batched_log_probs, batched_E_rel, eval_metrics

def log_prob_metrics(batched_log_probs, batched_E_rel, gs_hardness, config):
    batched_log_probs = np.array(batched_log_probs)
    batched_E_rel = np.array(batched_E_rel)

    sorted_log_probs, sorted_E_rel, sorted_batched_Energy_gaps, sorted_gs_hardness = zip(*sorted(zip(batched_log_probs, batched_E_rel, gs_hardness)))

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(np.arange(0, len(sorted_log_probs)), (1 / (2 ** config["x"])) * np.ones((len(sorted_log_probs))), label="baseline")
    plt.plot(np.arange(0, len(sorted_log_probs)), sorted_log_probs, label="probs")
    plt.plot(np.arange(0, len(sorted_log_probs)), sorted_gs_hardness, label="hardness")
    plt.xlabel("sorted Hamiltonian id")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    # plt.yscale("log")
    plt.ylabel("probs")

    plt.subplot(212)
    plt.plot(np.arange(0, len(sorted_log_probs)), sorted_E_rel)
    plt.yscale("log")
    plt.xlabel("sorted Hamiltonian id")
    plt.ylabel("E_rel")

    plt.tight_layout()

    wandb.log({"log_probs_histo": wandb.Image(fig)})
    plt.close("all")