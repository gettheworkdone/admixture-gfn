from typing import Any

import chex
import jax
import jax.numpy as jnp

from ..environment import PhyloTreeEnvironment, PhyloTreeEnvParams, PhyloTreeEnvState
from ..utils.corr import pearsonr
from ..utils.masking import mask_logits
from ..utils.rollout import TPolicyFn, TPolicyParams, backward_rollout, forward_rollout
from .base import BaseMetricModule, MetricState


@chex.dataclass
class PhyloTreeCorrelationState(MetricState):
    pass


class PhyloTreeCorrelationMetric(
    BaseMetricModule[PhyloTreeEnvState, PhyloTreeEnvParams, PhyloTreeCorrelationState]
):
    def __init__(
        self,
        n_rounds: int,
        n_terminal_states: int,
        batch_size: int,
        fwd_policy_fn: TPolicyFn,
        bwd_policy_fn: TPolicyFn,
        env: PhyloTreeEnvironment,
    ):
        self.n_rounds = n_rounds
        self.n_terminal_states = n_terminal_states
        self.batch_size = batch_size
        self.fwd_policy_fn = fwd_policy_fn
        self.bwd_policy_fn = bwd_policy_fn
        self.env = env

    def init(
        self, rng_key: chex.PRNGKey, env_params: PhyloTreeEnvParams
    ) -> PhyloTreeCorrelationState:
        return PhyloTreeCorrelationState()

    def update(
        self,
        metric_state: PhyloTreeCorrelationState,
        states: PhyloTreeEnvState,
        env_params: PhyloTreeEnvParams,
    ) -> PhyloTreeCorrelationState:
        return metric_state

    def compute(
        self,
        rng_key: chex.PRNGKey,
        metric_state: PhyloTreeCorrelationState,
        policy_params: TPolicyParams,
        env_params: PhyloTreeEnvParams,
    ) -> dict:
        # Perform a forward rollout with the current learned policy
        # See https://github.com/tristandeleu/gfn-maxent-rl/issues/32
        traj_data, info = forward_rollout(
            rng_key=rng_key,
            num_envs=self.n_terminal_states,
            policy_fn=self.fwd_policy_fn,
            policy_params=policy_params,
            env=self.env,
            env_params=env_params,
        )
        final_env_states = info["final_env_state"]
        log_rewards = traj_data.log_gfn_reward[
            jnp.arange(self.n_terminal_states), traj_data.state.time[:, -1] - 1
        ]

        # Reshape it to easier work with batches
        terminal_states = jax.tree.map(
            lambda x: x.reshape(-1, self.batch_size, *x.shape[1:]),
            final_env_states,
        )

        def process_batch(rng_key, terminal_states):
            rng_key, rollout_key = jax.random.split(rng_key)
            bwd_traj_data, _ = backward_rollout(
                rng_key=rollout_key,
                init_state=terminal_states,
                policy_fn=self.bwd_policy_fn,
                policy_params=policy_params,
                env=self.env,
                env_params=env_params,
            )

            # Next, we extract all the required data
            def flatten_tree(tree):
                return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), tree)

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
                log_ratio.reshape(self.batch_size, -1)
                * (1.0 - bwd_traj_data.pad[:, :-1])
            ).sum(axis=-1)
            return rng_key, log_ratio_traj

        def process_round(carry: Any, xs: None):
            rng_key, terminal_states = carry
            rng_key, log_ratio_traj = jax.lax.scan(
                process_batch, rng_key, terminal_states
            )
            chex.assert_shape(log_ratio_traj, terminal_states.time.shape[:2])
            return (rng_key, terminal_states), log_ratio_traj.reshape(-1)

        _, log_ratio_traj = jax.lax.scan(
            process_round,
            (rng_key, terminal_states),
            xs=None,
            length=self.n_rounds,
        )
        chex.assert_shape(log_ratio_traj, (self.n_rounds, self.n_terminal_states))
        # Average ratios over rounds for each test datum
        log_ratio_traj = jax.nn.logsumexp(log_ratio_traj, axis=0)
        log_ratio_traj = log_ratio_traj - jnp.log(self.n_rounds)
        chex.assert_equal_shape([log_ratio_traj, log_rewards])
        return {"pearson_corr": pearsonr(log_ratio_traj, log_rewards)}
