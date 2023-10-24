import os.path
import os
import copy
import numpy as np
from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
import pickle
import optax
import jraph
from tqdm import tqdm
import wandb
import igraph

from Networks.policy import Policy
from jraph_utils import pad_graph_to_nearest_power_of_k, add_random_node_features
from utils.lr_schedule import cosine_linear
from Data.RandomGraphs import ErdosRenyiGraphs
from Data.LoadGraphDataset import SolutionDatasetLoader


class TrainMeanField:
	def __init__(self, config, wandb_project, wandb_group, wandb_run):

		jax.config.update('jax_disable_jit', not config["jit"])

		self.config = config

		self.path_to_models = config["working_directory_path"] + "/MeanField/Checkpoints"


		self.seed = self.config["seed"]
		self.key = jax.random.PRNGKey(self.seed)

		# if epoch % save_modulo == 0 the params will be saved
		self.save_modulo = 50

		self.dataset_name = self.config["dataset_name"]
		self.problem_name = self.config["problem_name"]

		self.epochs = self.config["N_warmup"] + self.config["N_anneal"] + self.config["N_equil"]
		self.config["epochs"] = self.epochs

		self.lr = self.config["lr"]
		self.N_basis_states = self.config["N_basis_states"]
		self.batch_size = self.config["batch_size"]
		self.random_node_features = self.config["random_node_features"]
		self.n_random_node_features = self.config["n_random_node_features"]
		self.relaxed = self.config["relaxed"]

		self.T_max = self.config["T_max"]
		self.T = self.T_max
		self.N_warmup = self.config["N_warmup"]
		self.N_anneal = self.config["N_anneal"]
		self.N_equil = self.config["N_equil"]

		# Network
		self.n_features_list_prob = self.config["n_features_list_prob"]
		self.n_features_list_nodes = self.config["n_features_list_nodes"]
		self.n_features_list_edges = self.config["n_features_list_edges"]
		self.n_features_list_messages = self.config["n_features_list_messages"]
		self.n_features_list_encode = self.config["n_features_list_encode"]
		self.n_features_list_decode = self.config["n_features_list_decode"]
		self.n_message_passes = self.config["n_message_passes"]
		self.message_passing_weight_tied = self.config["message_passing_weight_tied"]
		self.linear_message_passing = self.config["linear_message_passing"]

		if self.config["wandb"]:
			self.wandb_mode = "online"
		else:
			self.wandb_mode = "disabled"

		self.wandb_project = wandb_project

		self.wandb_run_id = wandb.util.generate_id()
		self.wandb_group = wandb_group
		self.wandb_run = f"{self.wandb_run_id}_{wandb_run}"

		self.best_rel_error = float('inf')

		# Loss function
		self.loss = self.loss_free_energy_relaxed if self.relaxed else self.loss_free_energy
		if self.relaxed:
			if self.problem_name == 'MVC':
				self.relaxed_energy = self.MVC_Energy
			elif self.problem_name == 'MIS':
				self.relaxed_energy = self.MIS_Energy
			else:
				raise ValueError(f'For {self.problem_name} exists no energy function that can be used with relaxed states!')

		self.stop_epochs = self.config["stop_epochs"]
		self.epochs_since_best = 0

		self.__init_dataset()
		self.__init_network()
		self.__init_functions()
		self.__init_wandb(self.config)

	def __init_network(self):
		"""
		initialize network and optimizer
		"""
		self.model = Policy(n_features_list_prob=self.n_features_list_prob,
							n_features_list_nodes=self.n_features_list_nodes,
							n_features_list_edges=self.n_features_list_edges,
							n_features_list_messages=self.n_features_list_messages,
							n_features_list_encode=self.n_features_list_encode,
							n_features_list_decode=self.n_features_list_decode,
							n_message_passes=self.n_message_passes,
							message_passing_weight_tied=self.message_passing_weight_tied,
							linear_message_passing=self.linear_message_passing)

		self.__init_params()

		self.optimizer = optax.adam(learning_rate=self.lr)
		# self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(), optax.scale(-self.lr))
		self.opt_state = self.optimizer.init(self.params)

	def __init_functions(self):
		"""
		initialize functions (for jitting or vmapping)
		"""
		self.__vmap_get_energy = jax.vmap(self.__get_energy, in_axes=(None, 0), out_axes=(1))
		self.__vmap_MVC_energy = jax.vmap(self.MVC_Energy, in_axes=(None, 0), out_axes=(1))
		self.cosine_schedule_func = lambda epoch: cosine_linear(epoch, self.epochs, amplitude=self.lr/2, mean_lr=self.lr, factor=0.1)
		self.__init_loss_grad()

	def __init_params(self):
		"""
		initialize network parameters
		"""
		self.key, subkey = jax.random.split(self.key)

		jraph_graph, _, _ = next(iter(self.dataloader_val))
		if self.random_node_features:
			jraph_graph = add_random_node_features(jraph_graph, n_random_node_features=self.n_random_node_features, seed=0)
		jraph_graph = pad_graph_to_nearest_power_of_k(jraph_graph)
		jraph_graph = self.__cast_jraph_to(jraph_graph, np_=jnp)

		self.params = self.model.init({"params": subkey},
									  jraph_graph=jraph_graph,
									  N_basis_states=self.N_basis_states,
									  key=self.key)

	def __init_dataset(self):
		data_generator = SolutionDatasetLoader(config = self.config,dataset=self.dataset_name,
											   problem=self.problem_name,
											   batch_size=self.batch_size,
											   relaxed=self.relaxed,
											   seed=self.seed)
		self.dataloader_train, self.dataloader_test, self.dataloader_val, (
			self.mean_energy, self.std_energy) = data_generator.dataloaders()

	def __init_wandb(self, config):
		"""
		initialize weights and biases

		@param project: project name
		"""
		wandb.init(project=self.wandb_project, name=self.wandb_run, group=self.wandb_group, id=self.wandb_run_id,
				   config=config, mode=self.wandb_mode)

	def __init_loss_grad(self):
		self.loss_grad = jax.jit(jax.value_and_grad(self.loss, has_aux=True))

	@partial(jax.jit, static_argnums=(0,))
	def __update_params(self, params, grads, opt_state):
		grad_update, opt_state = self.optimizer.update(grads, opt_state)
		params = optax.apply_updates(params, grad_update)
		return params, opt_state

	@partial(jax.jit, static_argnums=(0,))
	def loss_backward(self, params, opt_state, graphs, T, key):
		(loss, (energies, free_energies, entropies, spin_log_probs, key)), grad = self.loss_grad(params, graphs, T, key)
		params, opt_state = self.__update_params(params, grad, opt_state)
		return params, opt_state, loss, (energies, free_energies, entropies, spin_log_probs, key)

	@partial(jax.jit, static_argnums=(0,))
	def __get_energy(self, jraph_graph, state):
		"""
		get energy of batched jraph_graph

		@params jraph_graph: batched jraph graph
		@params state: one basis state for batched graph
		@returns: energy (batch_size, 1)
		"""
		state = jnp.expand_dims(state, axis=-1)
		nodes = jraph_graph.nodes
		n_node = jraph_graph.n_node
		n_graph = jraph_graph.n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		sum_n_node = jraph_graph.nodes.shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

		energy_messages = jraph_graph.edges * state[jraph_graph.senders] * state[jraph_graph.receivers]
		energy_per_node = 0.5 * jax.ops.segment_sum(energy_messages, jraph_graph.receivers,
													sum_n_node) + state * jnp.expand_dims(jraph_graph.nodes[:, 0],
																						  axis=-1)
		energy = jax.ops.segment_sum(energy_per_node, node_graph_idx, n_graph)
		return energy

	def loss_energy(self, params, graphs, T, key):
		"""
		loss function

		mean_{over graphs}( mean_{over basis states}( (E(s) - E_mean_basis_states) * log_prob(s) ) )

		NOTE on dimensions:
		states: (n_basis_states, total_n_nodes)
		log_probs: (n_basis_states, batch_size)
		energies: (batch_size, n_basis_states, 1)
		"""
		states, _log_probs, spin_log_probs, spin_logits, _, key = self.model.apply(params, graphs, self.N_basis_states, key)
		log_probs = _log_probs[:, :-1]
		# log_probs.shape: (Nb, Hb)

		_energies = self.__vmap_get_energy(graphs, states)
		energies = _energies[:-1]
		# energies.shape: (Hb, Nb, 1)

		entropies = jnp.expand_dims(jnp.mean(-log_probs, axis=0), axis=-1)
		# entropies.shape: (Hb, 1)

		energies = jnp.squeeze(energies, axis=-1)
		# free_energies.shape: (Hb, Nb)

		graph_mean_free_energy = jnp.expand_dims(jnp.mean(energies, axis=1), axis=-1)
		# graph_mean_free_energy.shape: (Hb, 1)

		delta_energy = energies - graph_mean_free_energy
		# delta_free_energy.shape: (Hb, Nb)

		loss = jnp.mean(jnp.mean(lax.stop_gradient(delta_energy) * log_probs.T, axis=1))

		return loss, (energies, None, entropies, spin_log_probs, key)

	def loss_free_energy_relaxed(self, params, graphs, T, key):
		"""
		loss function for training with annealing

		NOTE on dimensions:
		log_probs: (Nb, Hb)
		energies: (Hb, 1)
		"""
		states, _log_probs, spin_log_probs, spin_logits, entropies, key = self.model.apply(params, graphs, self.N_basis_states,
																				key)
		entropies = entropies[:-1]

		relaxed_state = jnp.exp(spin_logits[:, 1])
		relaxed_state = jnp.expand_dims(relaxed_state, axis=-1)
		relaxed_energies = self.relaxed_energy(graphs, relaxed_state)
		energies = relaxed_energies[:-1]
		# energies.shape: (Hb, 1)

		free_energies = energies.flatten() - T*entropies
		loss = jnp.mean(free_energies)

		return loss, (energies, free_energies, entropies, spin_log_probs, key)

	def loss_free_energy(self, params, graphs, T, key):
		"""
		loss function for training with annealing

		mean_{over graphs}( mean_{over basis states}( (Free_E(s) - Free_E_mean_basis_states) * log_prob(s) ) )

		NOTE on dimensions:
		states: (Nb, total_n_nodes)
		log_probs: (Nb, Hb)
		energies: (Hb, Nb, 1)
		"""
		states, _log_probs, spin_log_probs, spin_logits, _, key = self.model.apply(params, graphs, self.N_basis_states, key)
		log_probs = _log_probs[:, :-1]

		# log_probs.shape: (Nb, Hb)

		_energies = self.__vmap_get_energy(graphs, states)
		energies = _energies[:-1]
		# energies.shape: (Hb, Nb, 1)

		entropies = jnp.expand_dims(jnp.mean(-log_probs, axis=0), axis=-1)
		# entropies.shape: (Hb, 1)

		free_energies = jnp.squeeze(energies, axis=-1) + T * log_probs.T
		# free_energies.shape: (Hb, Nb)

		graph_mean_free_energy = jnp.expand_dims(jnp.mean(free_energies, axis=1), axis=-1)
		# graph_mean_free_energy.shape: (Hb, 1)

		delta_free_energy = free_energies - graph_mean_free_energy
		# delta_free_energy.shape: (Hb, Nb)

		loss = jnp.mean(jnp.mean(lax.stop_gradient(delta_free_energy) * log_probs.T, axis=1))

		return loss, (energies, free_energies, entropies, spin_log_probs, key)

	def __linear_annealing(self, epoch):
		if epoch < self.N_warmup:
			T_curr = self.T_max
		elif epoch >= self.N_warmup and epoch < self.epochs - self.N_equil - 1:
			T_curr = max([self.T_max - self.T_max * (epoch - self.N_warmup) / self.N_anneal, 0])
		else:
			T_curr = 0.
		return T_curr

	def __linear_annealing_reverse(self, epoch):
		if epoch <= self.epochs:
			T_curr = max([self.T_max + epoch / self.N_warmup, 0])
		return T_curr

	def __update_optimizer(self, lr, params):
		self.curr_lr = lr
		# self.optimizer = optax.radam(learning_rate=self.curr_lr)
		self.optimizer = optax.adam(learning_rate=lr)
		opt_state = self.optimizer.init(params)
		return opt_state

	def train(self):
		print("first evaluation...")
		self.eval(epoch=0)
		print("start training...")

		for epoch in tqdm(range(self.epochs), desc="Training"):
			self.T = self.__linear_annealing(epoch)

			epoch_losses = []
			epoch_energies = []
			epoch_free_energies = []
			epoch_gt_energies = []
			epoch_best_energies = []
			epoch_rel_energies = []
			epoch_rel_energies_best = []
			epoch_mean_probs = []
			epoch_mean_entropies = []
			epoch_mean_normed_energies = []
			mean_APR = []
			mean_best_APR = []
			for iter, (graph_batch, gt_normed_energies, gt_spin_state) in enumerate(self.dataloader_train):
				if self.random_node_features:
					graph_batch = add_random_node_features(graph_batch, n_random_node_features=self.n_random_node_features, seed=self.seed)
				graph_batch = pad_graph_to_nearest_power_of_k(graph_batch)

				self.params, self.opt_state, loss, (
					energies, free_energies, entropies, spin_log_probs, self.key) = self.loss_backward(self.params,
																									   self.opt_state,
																									   graph_batch,
																									   self.T,
																									   self.key)

				mean_energy, mean_normed_energy, mean_normed_free_energy, mean_gt_energy, mean_best_energy, rel_error, mean_best_rel_error, mean_prob, APR, best_APR= self.__calculate_reporting(
					energies, gt_normed_energies, spin_log_probs, free_energies)

				wandb.log({
					"train_batch/loss": loss,
					"train_batch/mean_energy": mean_energy,
					"train_batch/mean_gt_energy": mean_gt_energy,
					"train_batch/mean_best_energy": mean_best_energy,
					"train_batch/rel_error": rel_error,
					"train_batch/mean_best_rel_error": mean_best_rel_error,
					"train_batch/mean_probs": mean_prob,
				})

				epoch_losses.append(loss)
				epoch_energies.append(mean_energy)
				epoch_free_energies.append(mean_normed_free_energy)
				epoch_gt_energies.append(mean_gt_energy)
				epoch_best_energies.append(mean_best_energy)
				epoch_rel_energies.append(rel_error)
				epoch_rel_energies_best.append(mean_best_rel_error)
				epoch_mean_probs.append(mean_prob)
				epoch_mean_entropies.append(np.mean(entropies))
				epoch_mean_normed_energies.append(mean_normed_energy)
				mean_APR.append(APR)
				mean_best_APR.append(best_APR)


			new_lr = self.lr #self.cosine_schedule_func(epoch)
			#self.opt_state = self.__update_optimizer(new_lr, self.params)

			wandb.log({
				"train/epoch": epoch,
				"train/T": self.T,
				"train/mean_loss": np.mean(epoch_losses),
				"train/mean_energy": np.mean(epoch_energies),
				"train/mean_normed_free_energy": np.mean(epoch_free_energies),
				"train/mean_gt_energy": np.mean(epoch_gt_energies),
				"train/mean_best_energy": np.mean(epoch_best_energies),
				"train/rel_error": np.mean(epoch_rel_energies),
				"train/rel_error_best": np.mean(epoch_rel_energies_best),
				"train/mean_prob": np.mean(epoch_mean_probs),
				"train/mean_entropy": np.mean(epoch_mean_entropies),
				"train/mean_normed_energies": np.mean(epoch_mean_normed_energies),
				"train/mean_APR": np.mean(mean_APR) if self.problem_name == 'MVC' else np.nan,
				"train/mean_best_APR": np.mean(mean_best_APR) if self.problem_name == 'MVC' else np.nan,
				"schedules/lr": new_lr,
			})

			self.eval(epoch=epoch + 1)

			if self.epochs_since_best == self.stop_epochs:
				# early stopping
				break


	def eval(self, epoch):
		mean_energies = []
		mean_gt_energies = []
		best_energies = []
		rel_energies = []
		rel_energies_best = []
		mean_probs = []
		mean_normed_energies = []
		mean_normed_free_energies = []
		mean_entropies = []
		mean_APR = []
		mean_best_APR = []
		for iter, (_graph_batch, gt_normed_energies, gt_spin_states) in enumerate(self.dataloader_val):
			if self.random_node_features:
				graph_batch = add_random_node_features(_graph_batch, n_random_node_features=self.n_random_node_features, seed=self.seed)
			graph_batch = pad_graph_to_nearest_power_of_k(graph_batch)

			states, log_probs, spin_log_probs, spin_logits, _, self.key = self.model.apply(self.params,
																						   graph_batch,
																						   self.N_basis_states,
																						   self.key)

			loss, (_, free_energies, entropies, _, self.key) = self.loss(self.params, graph_batch, self.T, self.key)

			if self.relaxed:
				relaxed_state = np.exp(spin_logits[:, 1])
				relaxed_state = np.expand_dims(relaxed_state, axis=-1)
				relaxed_energies = self.relaxed_energy(graph_batch, relaxed_state)
				energies = relaxed_energies[:-1]
				graph_mean_energy = energies

			else:
				energies = self.__vmap_get_energy(graph_batch, states)
				energies = energies[:-1]
				graph_mean_energy = jnp.mean(energies, axis=1)

			mean_energy, mean_normed_energy, mean_normed_free_energy, mean_gt_energy, mean_best_energy, rel_error, mean_best_rel_error, mean_prob, APR, best_APR = self.__calculate_reporting(
				energies, gt_normed_energies, spin_log_probs, free_energies)

			mean_energies.append(mean_energy)
			mean_gt_energies.append(mean_gt_energy)
			best_energies.append(mean_best_energy)
			rel_energies.append(rel_error)
			rel_energies_best.append(mean_best_rel_error)
			mean_probs.append(mean_prob)
			mean_normed_energies.append(mean_normed_energy)
			mean_normed_free_energies.append(mean_normed_free_energy)
			mean_entropies.append(np.mean(entropies))
			mean_APR.append(APR)
			mean_best_APR.append(best_APR)

			wandb.log({
				"eval_batch/mean_energy": mean_energy,
				"eval_batch/mean_gt_energy": mean_gt_energy,
				"eval_batch/mean_best_energy": mean_best_energy,
				"eval_batch/rel_error": rel_error,
				"eval_batch/mean_best_rel_error": mean_best_rel_error,
				"eval_batch/graph_mean_energies": wandb.Histogram(graph_mean_energy)
			})

		eval_log_dict = {
			"eval/epoch": epoch,
			"eval/mean_energy": np.mean(mean_energies),
			"eval/mean_gt_energy": np.mean(mean_gt_energies),
			"eval/mean_best_energy": np.mean(best_energies),
			"eval/rel_error": np.mean(rel_energies),
			"eval/rel_error_best": np.mean(rel_energies_best),
			"eval/mean_prob": np.mean(mean_probs),
			"eval/mean_entropy": np.mean(mean_entropies),
			"eval/mean_normed_free_energy": np.mean(mean_normed_free_energies),
			"eval/mean_normed_energy": np.mean(mean_normed_energies),
			"eval/mean_APR": np.mean(mean_APR) if self.problem_name == 'MVC' else np.nan,
			"eval/mean_best_APR": np.mean(mean_best_APR) if self.problem_name == 'MVC' else np.nan,
			"eval/epochs_since_best": self.epochs_since_best,
			"eval/best_rel_error": self.best_rel_error,
		}

		if np.mean(rel_energies) < self.best_rel_error:
			self.__save_params(best_run=True, eval_dict=eval_log_dict)
			self.best_rel_error = np.mean(rel_energies)
			self.epochs_since_best = 0
		else:
			self.epochs_since_best += 1

		if epoch % self.save_modulo == 0 or epoch == self.epochs:
			self.__save_params(best_run=False, eval_dict=eval_log_dict)

		wandb.log(eval_log_dict)

	def __calculate_reporting(self, normed_energies, gt_normed_energies, spin_log_probs, normed_free_energies=np.nan):
		if not np.isnan(normed_free_energies).all():
			mean_normed_free_energy = np.mean(normed_free_energies)
		else:
			mean_normed_free_energy = np.nan

		if self.relaxed:
			energies = np.squeeze(normed_energies, axis=-1)
			gt_energies = np.array(gt_normed_energies)
			if len(energies.shape) > 1:
				raise ValueError('when relaxed, energies should be one dimensional!')
			min_energies = energies

			rel_error = np.mean(np.abs(gt_energies - energies) / np.abs(gt_energies))
			best_rel_error_per_graph = np.abs(gt_energies - min_energies) / np.abs(gt_energies)

			mean_normed_energy = np.mean(energies)
			mean_energy = np.mean(energies)
			mean_gt_energy = np.mean(gt_energies)
			mean_best_energy = np.mean(min_energies)
			mean_best_rel_error = np.mean(best_rel_error_per_graph)

			mean_prob = np.mean(np.exp(spin_log_probs))

			if self.problem_name == 'MVC':
				APR_per_graph = energies / gt_energies
				best_APR_per_graph = min_energies / gt_energies
				APR = np.mean(APR_per_graph)
				best_APR = np.mean(best_APR_per_graph)
			else:
				APR = None
				best_APR = None

		else:
			energies = np.array(normed_energies * self.std_energy + self.mean_energy)
			gt_energies = np.array(gt_normed_energies * self.std_energy + self.mean_energy)
			gt_energies = np.expand_dims(gt_energies, axis=-1)

			min_energies = np.min(energies, axis=1)

			rel_error = np.mean(np.abs(gt_energies - np.squeeze(energies, axis=-1)) / np.abs(gt_energies))
			best_rel_error_per_graph = np.abs(gt_energies - min_energies) / np.abs(gt_energies)

			mean_normed_energy = np.mean(normed_energies)
			mean_energy = np.mean(energies)
			mean_gt_energy = np.mean(gt_energies)
			mean_best_energy = np.mean(min_energies)
			mean_best_rel_error = np.mean(best_rel_error_per_graph)

			mean_prob = np.mean(np.exp(spin_log_probs))

			if self.problem_name == 'MVC':
				APR_per_graph = np.squeeze(energies, axis=-1) / gt_energies
				best_APR_per_graph = min_energies / gt_energies
				APR = np.mean(APR_per_graph)
				best_APR = np.mean(best_APR_per_graph)
			else:
				APR = None
				best_APR = None

		output = (mean_energy, mean_normed_energy, mean_normed_free_energy, mean_gt_energy, mean_best_energy, rel_error,
				  mean_best_rel_error, mean_prob, APR, best_APR)
		return output

	@partial(jax.jit, static_argnums=(0,))
	def MVC_Energy(self, H_graph, bins):
		A = 1.
		B = 1.1

		nodes = H_graph.nodes
		n_node = H_graph.n_node
		n_graph = n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		sum_n_node = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_gr_idx = jnp.repeat(
			graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
		total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

		adjacency = jnp.ones_like(H_graph.edges)

		### normalise through average number of nodes in dataset
		A = A
		B = B

		raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
		Energy_messages = adjacency * (1 - raveled_bins[H_graph.senders]) * (1 - raveled_bins[H_graph.receivers])
		Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

		Hb = B * (jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph))
		Ha = A * (jax.ops.segment_sum(raveled_bins, node_gr_idx, n_graph))

		Energy = Ha + Hb

		return Energy

	@partial(jax.jit, static_argnums=(0,))
	def MIS_Energy(self, H_graph, bins):
		A = 1.
		B = 1.1

		nodes = H_graph.nodes
		n_node = H_graph.n_node
		n_graph = n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		sum_n_node = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_gr_idx = jnp.repeat(
			graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
		total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

		adjacency = jnp.ones_like(H_graph.edges)

		### normalise through average number of nodes in dataset
		A = A
		B = B

		raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
		Energy_messages = adjacency * (raveled_bins[H_graph.senders]) * (raveled_bins[H_graph.receivers])
		Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

		Hb = B * (jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph))
		Ha = A * (jax.ops.segment_sum(raveled_bins, node_gr_idx, n_graph))

		Energy = - Ha + Hb

		return Energy

	def __save_params(self, best_run: bool, eval_dict: dict):
		params_to_save = (self.params, self.config, eval_dict)
		path_folder = f"{self.path_to_models}/{self.wandb_run_id}/"

		if not os.path.exists(path_folder):
			os.makedirs(path_folder)

		if best_run:
			file_name = f"best_{self.wandb_run_id}.pickle"
		else:
			file_name = f"{self.wandb_run_id}_T_{self.T}.pickle"

		with open(os.path.join(path_folder, file_name), 'wb') as f:
			pickle.dump(params_to_save, f)


	def __load_params(self, wandb_id, T, best_run):
		if best_run:
			file_name = f"best_{wandb_id}.pickle"
		else:
			file_name = f"{wandb_id}_T_{round(T, 3)}.pickle"

		with open(f'Checkpoints/{wandb_id}/{file_name}', 'rb') as f:
			params, config = pickle.load(f)
		return params, config

	def __cast_jraph_to(self, j_graph, np_=jnp):
		"""
		cast jraph tuple to np_ (i.e. to np or jnp)

		NOTE: Global features will be ignored; i.e. global features will be set to None!
		"""
		j_graph = jraph.GraphsTuple(nodes=np_.asarray(j_graph.nodes),
									edges=np_.asarray(j_graph.edges),
									receivers=np_.asarray(j_graph.receivers),
									senders=np_.asarray(j_graph.senders),
									n_node=np_.asarray(j_graph.n_node),
									n_edge=np_.asarray(j_graph.n_edge),
									globals=None)
		return j_graph
