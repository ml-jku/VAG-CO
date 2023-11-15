import jax
import jax.numpy as jnp
import jraph
import optax


class BaseRNNSampler:

    def __init__(self):
        pass

    ### TODO TestScripts if symmetries work correctly
    def make_symmetries(self, spins):
        if(self.symmetrised):
            resh_spins = spins[self.back_transform]
            # resh_spins = spins
            resh_spins = jax.lax.reshape(resh_spins, (self.x, self.y))
            if(self.x == self.y):
                spin_list = [resh_spins, jnp.flip(resh_spins, axis = 0), jnp.flip(resh_spins, axis = 1)] + [jnp.rot90(resh_spins, k = 1 + i) for i in range(3)]
            else:
                spin_list = [resh_spins, jnp.flip(resh_spins, axis=0), jnp.flip(resh_spins, axis=1)] + [jnp.rot90(resh_spins, k= 2)]

            symm_spins = jnp.reshape(jnp.array(spin_list), (len(spin_list), self.x * self.y))
            symm_spins = jnp.take(symm_spins, self.spiral_transform, axis=1)
        else:
            spin_list = [spins]

            symm_spins = jnp.reshape(jnp.array(spin_list), (len(spin_list), self.x * self.y))

        return symm_spins

    def update_params(self, params, grad, opt_state, opt_update ):
        grad_update, opt_state = opt_update(grad, opt_state)
        params = optax.apply_updates(params, grad_update)
        return params, opt_state

    def return_batched_graph(self, n_basis_states):
        batch_graph = jraph.batch([self.GraphDataLoader[-1] for basis_state in range(n_basis_states)])
        return batch_graph

    def return_mini_batched_graph(self, n = 4):
        batch_graph = jraph.batch([self.GraphDataLoader[-1] for basis_state in range(n)])
        return batch_graph


    def VNA_loss(self, sample_fun, O_loc_func, O_graphs, params, H_graph, key, T):  # log_amplitudes (n_basis_states, ); E_loc (n_basis_states, )
        all_spins_descrete, log_probs, phases, log_amplitudes, key = sample_fun(params, key, H_graph)

        all_spins = 2 * all_spins_descrete - 1

        E_loc = jax.lax.stop_gradient(O_loc_func(all_spins, O_graphs))

        mean_E_loc = jnp.mean(jnp.real(E_loc))
        var_E_loc = jnp.std(jnp.real(E_loc)) / len(E_loc)

        mean_Entropy = jax.lax.stop_gradient(jnp.mean(log_probs))
        Entropy_baseline = jax.lax.stop_gradient(log_probs) - mean_Entropy
        Entropy_loss = jnp.mean(log_probs * Entropy_baseline)

        cost_Energy = 2* jnp.real(jnp.mean(jnp.conjugate(log_amplitudes) * E_loc) - jnp.conjugate(jnp.mean(log_amplitudes)) * jnp.mean(E_loc))
        cost_FreeEnergy = cost_Energy + T * Entropy_loss

        return cost_FreeEnergy, (mean_E_loc, var_E_loc, -mean_Entropy, jnp.real(E_loc), key)

    def VNA_loss_TranslationInvariant(self, sample_fun, eval_func, O_loc_func, O_graphs, params, H_graph, key, T):  # log_amplitudes (n_basis_states, ); E_loc (n_basis_states, )
        all_spins_descrete, log_probs, phases, log_amplitudes, key = sample_fun(params, key, H_graph)

        log_probs, phases, log_amplitudes, key = eval_func(params, H_graph, all_spins_descrete, key)

        all_spins = 2 * all_spins_descrete - 1

        E_loc = jax.lax.stop_gradient(O_loc_func(all_spins, O_graphs))

        mean_E_loc = jnp.mean(jnp.real(E_loc))
        var_E_loc = jnp.std(jnp.real(E_loc)) / len(E_loc)

        mean_Entropy = jax.lax.stop_gradient(jnp.mean(log_probs))
        Entropy_baseline = jax.lax.stop_gradient(log_probs) - mean_Entropy
        Entropy_loss = jnp.mean(log_probs * Entropy_baseline)

        cost_Energy = 2* jnp.real(jnp.mean(jnp.conjugate(log_amplitudes) * E_loc) - jnp.conjugate(jnp.mean(log_amplitudes)) * jnp.mean(E_loc))
        cost_FreeEnergy = cost_Energy + T * Entropy_loss

        return cost_FreeEnergy, (mean_E_loc, var_E_loc, mean_Entropy, jnp.real(E_loc), key)

    ### TODO take care! THis currently only works without symmetry
    def VMC_loss_off_policy(self, sample_fun, eval_func, E_loc_func, batched_graph, params, key, j2, T):  # log_amplitudes (n_basis_states, ); E_loc (n_basis_states, )
        all_spins_descrete, off_policy_log_probs, _, key = jax.lax.stop_gradient(sample_fun(params, key, j2, T))

        all_spins_descrete = jax.lax.reshape(all_spins_descrete, (self.n_basis_states, self.n_nodes))
        # print(all_spins_descrete)
        log_probs, phases = eval_func(params, all_spins_descrete, j2, 0.)
        # print(log_probs, phases)
        # print("shapes", log_probs.shape, first_log_probs.shape, first_phases.shape, phases.shape)

        log_amplitudes = 0.5 * log_probs + 1.j * phases
        all_spins_descrete = jnp.ravel(all_spins_descrete)
        all_spins = 2 * all_spins_descrete - 1

        ### TODO replace this with something jitable
        minibatched_graph_list = self.make_minibatched_graph_list(batched_graph, all_spins, log_amplitudes, j2)

        E_loc = jax.lax.stop_gradient(
            jnp.exp(log_probs - off_policy_log_probs) * E_loc_func(params, minibatched_graph_list, j2, 0.))

        mean_E_loc = jnp.mean(jnp.real(E_loc))
        var_E_loc = jnp.std(jnp.real(E_loc)) / len(E_loc)

        if (False):
            cost_E = 2 * jnp.real(jnp.mean(jnp.conjugate(log_amplitudes) * E_loc) - jnp.conjugate(jnp.mean(log_amplitudes)) * jnp.mean(E_loc))
        else:
            N = jax.lax.stop_gradient(jnp.sum(jnp.exp(log_probs - off_policy_log_probs)))
            mean_E = jnp.sum(E_loc) / N
            cost_E = 2 * jnp.real(jnp.sum(jnp.conjugate(log_amplitudes) * E_loc) - jnp.conjugate(jnp.sum(log_amplitudes)) * mean_E) / N

        all_spins = jnp.reshape(all_spins, (self.n_basis_states, self.n_nodes))

        mean_M = jnp.mean(jnp.sum(all_spins, axis=1))

        return cost_E, (mean_E_loc, var_E_loc, mean_M, all_spins, log_amplitudes, key)

    def VMC_loss(self, sample_fun, eval_func, E_loc_func, batched_graph, params, key, j2,
                 T):  # log_amplitudes (n_basis_states, ); E_loc (n_basis_states, )
        all_spins_descrete, first_log_probs, first_phases, key = jax.lax.stop_gradient(sample_fun(params, key, j2, T))

        all_spins_descrete = jax.lax.reshape(all_spins_descrete, (self.n_basis_states, self.n_nodes))
        # print(all_spins_descrete)
        log_probs, phases = eval_func(params, all_spins_descrete, j2, T)
        # print(log_probs, phases)
        # print("shapes", log_probs.shape, first_log_probs.shape, first_phases.shape, phases.shape)

        log_amplitudes = 0.5 * log_probs + 1.j * phases
        all_spins_descrete = jnp.ravel(all_spins_descrete)
        all_spins = 2 * all_spins_descrete - 1

        ### TODO replace this with something jitable
        minibatched_graph_list = self.make_minibatched_graph_list(batched_graph, all_spins, log_amplitudes, j2)

        E_loc = jax.lax.stop_gradient(E_loc_func(params, minibatched_graph_list, j2, T))
        mean_E_loc = jnp.mean(jnp.real(E_loc))
        var_E_loc = jnp.std(jnp.real(E_loc)) / len(E_loc)

        if (self.T != 0):
            mean_Entropy = jax.lax.stop_gradient(jnp.mean(log_probs))
            Entropy_baseline = jax.lax.stop_gradient(log_probs) - mean_Entropy
            Entropy_loss = jnp.mean(log_probs * Entropy_baseline)
        else:
            mean_Entropy = 0.
            Entropy_loss = 0.

        cost_E = 2 * jnp.real(jnp.mean(jnp.conjugate(log_amplitudes) * E_loc) - jnp.conjugate(jnp.mean(log_amplitudes)) * jnp.mean(E_loc)) + self.T * Entropy_loss

        all_spins = jnp.reshape(all_spins, (self.n_basis_states, self.n_nodes))

        mean_M = jnp.mean(jnp.sum(all_spins, axis=1))

        return cost_E, (mean_E_loc, var_E_loc, mean_M, all_spins, log_amplitudes, key, mean_Entropy)

    def test_eval_and_sample(self, sample_func, eval_func, params):
        all_spins_descrete, log_probs, phases = sample_func(params)

        all_spins_descrete = jax.lax.reshape(all_spins_descrete, (self.n_basis_states, self.n_nodes))
        eval_log_probs, eval_phases = eval_func(params, all_spins_descrete)

        return jnp.mean((log_probs-eval_log_probs)**2), jnp.mean((phases-eval_phases)**2)


    ### TODO remove phase net from sampling process
    def sample(self, params, key, j2, T): ## dimension batch, time, features
        print("WARNING: Sample fuunction is not Implemented!")

        return None

    ### TODO implement hamiltonian symmetry
    def evaluate_log_probs(self, params, fixed_nodes_descrete, j2, T):
        print("WARNING: Eval fuunction is not Implemented!")

        return None

    ### TODO implement hamiltonian symmetry

    def impose_zero_magnetisation(self, log_probs, nodes_descrete):
        max_Mag = self.n_nodes
        N_up = jnp.sum(nodes_descrete, axis = -1)
        N_down = nodes_descrete.shape[-1] - N_up

        ps = jnp.zeros_like(log_probs)

        p_up = (1-self.eps)*jax.numpy.heaviside(max_Mag/2*jnp.ones_like(N_up) - N_up, jnp.zeros_like(N_up)) + self.eps
        p_down = (1-self.eps)*jax.numpy.heaviside(max_Mag / 2 * jnp.ones_like(N_down) - N_down, jnp.zeros_like(N_up)) + self.eps


        ps = ps.at[:,1].set(p_up)
        ps = ps.at[:,0].set(p_down)

        log_probs = log_probs + jnp.log(ps) - jnp.log(jnp.expand_dims(jnp.sum(jnp.exp(log_probs)*ps, axis = -1), axis = -1))

        return log_probs