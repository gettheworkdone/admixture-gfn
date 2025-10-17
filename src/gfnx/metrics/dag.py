from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ..environment import DAGEnvironment, DAGEnvParams, DAGEnvState
from ..reward import DAGRewardModule
from ..utils.corr import pearsonr, spearmanr
from ..utils.dag import construct_all_dags, get_markov_blanket
from ..utils.masking import mask_logits
from ..utils.rollout import TPolicyFn, TPolicyParams, backward_rollout
from .base import BaseMetricModule, MetricState


@chex.dataclass
class DAGCorrelationState(MetricState):
    adjacency_matrix: Bool[Array, "num_graphs num_variables num_variables"]
    closure_T: Bool[Array, "num_graphs num_variables num_variables"]
    closure: Bool[Array, "num_graphs num_variables num_variables"]
    markov_blanket: Bool[Array, "num_graphs num_variables num_variables"]
    exact_log_posterior: Float[Array, "num_graphs"]


class DAGCorrelationMetric(
    BaseMetricModule[DAGEnvState, DAGEnvParams, DAGCorrelationState]
):
    def __init__(
        self,
        env: DAGEnvironment,
        bwd_policy_fn: TPolicyFn,
        n_rounds: int,
        batch_size: int,
    ):
        self.env = env
        self.bwd_policy_fn = bwd_policy_fn
        self.n_rounds = n_rounds
        self.batch_size = batch_size

    def init(
        self, rng_key: chex.PRNGKey, env_params: DAGEnvParams
    ) -> DAGCorrelationState:
        adjacency_matrix = construct_all_dags(
            env_params.num_variables
        )  # (num_graphs, num_variables, num_variables)
        closure_T = jax.vmap(
            lambda x: self.env._single_compute_closure_t(x.T), in_axes=0
        )(adjacency_matrix)  # (num_graphs, num_variables, num_variables)
        closure = jax.vmap(
            lambda x: self.env._single_compute_closure_t(x), in_axes=0
        )(adjacency_matrix)  # (num_graphs, num_variables, num_variables)
        markov_blanket = jax.vmap(lambda x: get_markov_blanket(x), in_axes=0)(
            adjacency_matrix
        )  # (num_graphs, num_variables, num_variables)
        num_graphs = adjacency_matrix.shape[0]

        test_set_states = DAGEnvState(
            adjacency_matrix=adjacency_matrix,
            closure_T=closure_T,
            time=jnp.zeros(num_graphs, dtype=jnp.int32),
            is_terminal=jnp.ones(num_graphs, dtype=jnp.bool),
            is_initial=jnp.zeros(num_graphs, dtype=jnp.bool),
            is_pad=jnp.zeros(num_graphs, dtype=jnp.bool),
        )
        log_rewards = self.env.reward_module.log_reward(
            test_set_states, env_params
        )
        log_Z = jax.nn.logsumexp(log_rewards)
        exact_log_posterior = log_rewards - log_Z

        return DAGCorrelationState(
            adjacency_matrix=adjacency_matrix,
            closure_T=closure_T,
            closure=closure,  # for path correlation
            markov_blanket=markov_blanket,  # for markov blanket correlation
            exact_log_posterior=exact_log_posterior,
        )

    def update(
        self,
        metric_state: DAGCorrelationState,
        states: DAGEnvState,
        env_params: DAGEnvParams,
    ) -> DAGCorrelationState:
        # No effect
        return metric_state

    def compute(
        self,
        rng_key: chex.PRNGKey,
        metric_state: DAGCorrelationState,
        policy_params: TPolicyParams,
        env_params: DAGEnvParams,
    ) -> dict:
        adjacency_matrix = metric_state.adjacency_matrix
        closure_T = metric_state.closure_T
        closure = metric_state.closure
        markov_blanket = metric_state.markov_blanket
        num_graphs = adjacency_matrix.shape[0]
        # Devide into batch of batches and reminder
        truncated_size = num_graphs // self.batch_size * self.batch_size
        truncated_adjacency_matrix = adjacency_matrix[:truncated_size]
        truncated_closure_T = closure_T[:truncated_size]
        # Reshape into batch of batches
        truncated_adjacency_matrix = truncated_adjacency_matrix.reshape(
            -1, self.batch_size, self.env.num_variables, self.env.num_variables
        )
        truncated_closure_T = truncated_closure_T.reshape(
            -1, self.batch_size, self.env.num_variables, self.env.num_variables
        )
        # Reminder
        reminder_adjacency_matrix = adjacency_matrix[truncated_size:][
            None, ...
        ]
        reminder_closure_T = closure_T[truncated_size:][None, ...]

        def process_batch(
            rng_key: chex.PRNGKey,
            adjacency_matrix_closure_T: Tuple[
                Bool[Array, "batch_size num_variables num_variables"],
                Bool[Array, "batch_size num_variables num_variables"],
            ],
        ):
            rng_key, rollout_key = jax.random.split(rng_key)
            adjacency_matrix_samples, closure_T_samples = (
                adjacency_matrix_closure_T
            )
            num_samples = adjacency_matrix_samples.shape[0]
            test_set_states = DAGEnvState(
                adjacency_matrix=adjacency_matrix_samples,
                closure_T=closure_T_samples,
                time=jnp.zeros(num_samples, dtype=jnp.int32),
                is_terminal=jnp.ones(num_samples, dtype=jnp.bool),
                is_initial=jnp.zeros(num_samples, dtype=jnp.bool),
                is_pad=jnp.zeros(num_samples, dtype=jnp.bool),
            )
            bwd_traj_data, _ = backward_rollout(
                rollout_key,
                init_state=test_set_states,
                policy_fn=self.bwd_policy_fn,
                policy_params=policy_params,
                env=self.env,
                env_params=env_params,
            )

            # Next, we extract all the required data
            def flatten_tree(tree):
                return jax.tree.map(
                    lambda x: x.reshape((-1,) + x.shape[2:]), tree
                )

            states = jax.tree.map(lambda x: x[:, :-1], bwd_traj_data.state)
            states = flatten_tree(states)
            prev_states = jax.tree.map(lambda x: x[:, 1:], bwd_traj_data.state)
            prev_states = flatten_tree(prev_states)

            forward_logits = flatten_tree(
                bwd_traj_data.info["forward_logits"][:, 1:]
            )  # logits for transition from prev_state
            backward_logits = flatten_tree(
                bwd_traj_data.info["backward_logits"][:, :-1]
            )  # logits for transition from state

            bwd_actions = flatten_tree(
                jax.tree.map(lambda x: x[:, :-1], bwd_traj_data.action)
            )
            fwd_actions = self.env.get_forward_action(
                states, bwd_actions, prev_states, env_params
            )

            fwd_mask = self.env.get_invalid_mask(prev_states, env_params)
            forward_logits = mask_logits(forward_logits, fwd_mask)
            forward_logprobs = jax.nn.log_softmax(forward_logits)
            sampled_forward_logprobs = jnp.take_along_axis(
                forward_logprobs, fwd_actions[..., None], axis=-1
            ).squeeze(-1)

            bwd_mask = self.env.get_invalid_backward_mask(states, env_params)
            backward_logits = mask_logits(backward_logits, bwd_mask)
            backward_logprobs = jax.nn.log_softmax(backward_logits)
            sampled_backward_logprobs = jnp.take_along_axis(
                backward_logprobs, bwd_actions[..., None], axis=-1
            ).squeeze(-1)

            log_ratio = sampled_forward_logprobs - sampled_backward_logprobs
            log_ratio_traj = (
                log_ratio.reshape(num_samples, -1)
                * (1.0 - bwd_traj_data.pad[:, :-1])
            ).sum(axis=-1)
            return rng_key, log_ratio_traj

        def process_round(carry: Any, xs: None):
            rng_key, adjacency_matrix, closure_T = carry
            rng_key, log_ratio_traj = jax.lax.scan(
                process_batch, rng_key, (adjacency_matrix, closure_T)
            )
            chex.assert_shape(log_ratio_traj, adjacency_matrix.shape[:2])
            return (
                rng_key,
                adjacency_matrix,
                closure_T,
            ), log_ratio_traj.reshape(-1)

        # Messy part. At first we compute n_rounds for a batch of batches,
        # then the same n_rounds for the reminder and concatenate the results.
        (
            (
                rng_key,
                truncated_adjacency_matrix,
                truncated_closure_T,
            ),
            log_ratio_traj,
        ) = jax.lax.scan(
            process_round,
            (rng_key, truncated_adjacency_matrix, truncated_closure_T),
            xs=None,
            length=self.n_rounds,
        )
        _, reminder_log_ratio_traj = jax.lax.scan(
            process_round,
            (rng_key, reminder_adjacency_matrix, reminder_closure_T),
            xs=None,
            length=self.n_rounds,
        )
        log_ratio_traj = jnp.hstack([log_ratio_traj, reminder_log_ratio_traj])
        chex.assert_shape(
            log_ratio_traj,
            (self.n_rounds, num_graphs),
        )
        # Average ratios over rounds for each test datum
        log_ratio_traj = jax.nn.logsumexp(log_ratio_traj, axis=0)
        log_ratio_traj = log_ratio_traj - jnp.log(self.n_rounds)
        exact_log_posterior = metric_state.exact_log_posterior
        chex.assert_equal_shape([log_ratio_traj, exact_log_posterior])

        # Compute correlations
        reward_corr = spearmanr(log_ratio_traj, exact_log_posterior)
        # Compute feature correlations
        ratio_traj = jnp.expand_dims(jnp.exp(log_ratio_traj), axis=(-1, -2))
        rewards = jnp.expand_dims(jnp.exp(exact_log_posterior), axis=(-1, -2))
        edge_corr = pearsonr(
            jnp.sum(adjacency_matrix * ratio_traj, axis=0).reshape(-1),
            jnp.sum(adjacency_matrix * rewards, axis=0).reshape(-1),
        )
        path_corr = pearsonr(
            jnp.sum(closure * ratio_traj, axis=0).reshape(-1),
            jnp.sum(closure * rewards, axis=0).reshape(-1),
        )
        markov_blanket_corr = pearsonr(
            jnp.sum(markov_blanket * ratio_traj, axis=0).reshape(-1),
            jnp.sum(markov_blanket * rewards, axis=0).reshape(-1),
        )

        # Jensen-Shannon divergence
        log_probs_mean = jnp.log(0.5) + jnp.logaddexp(
            log_ratio_traj, exact_log_posterior
        )
        kl1 = jnp.exp(log_ratio_traj) * (log_ratio_traj - log_probs_mean)
        kl2 = jnp.exp(exact_log_posterior) * (
            exact_log_posterior - log_probs_mean
        )
        jsd = 0.5 * jnp.sum(kl1 + kl2)
        return {
            "reward_corr": reward_corr,
            "edge_corr": edge_corr,
            "path_corr": path_corr,
            "markov_blanket_corr": markov_blanket_corr,
            "jsd": jsd,
        }
