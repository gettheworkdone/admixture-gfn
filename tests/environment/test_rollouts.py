from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest

from gfnx.environment.dag import DAGEnvironment
from gfnx.reward import DAGRewardModule
from gfnx.reward.dag_likelihood import ZeroScore
from gfnx.reward.dag_prior import UniformDAGPrior
from gfnx.utils.rollout import backward_rollout, forward_rollout


class DummyPolicy:
    """A dummy policy that returns uniform probabilities over valid actions."""

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def __call__(self, rng_key, obs, policy_params):
        batch_size = obs.shape[0]
        logits = jnp.ones((batch_size, self.n_actions))
        return logits, {"fwd_logits": logits, "bwd_logits": logits}


@pytest.mark.parametrize(
    "environment_class,reward_module_class,env_kwargs,reward_kwargs",
    [
        (
            DAGEnvironment,
            DAGRewardModule,
            {
                "num_variables": 5,
            },
            {
                "prior": UniformDAGPrior(5),
                "likelihood": ZeroScore(
                    data=jnp.zeros((100, 5))
                ),  # 100 samples of 5 variables
            },
        ),
    ],
)
class TestRollouts:
    @pytest.fixture
    def setup_forward_rollout(
        self, environment_class, reward_module_class, env_kwargs, reward_kwargs
    ):
        """Setup environment and its components."""
        reward_module = reward_module_class(**reward_kwargs)
        env = environment_class(reward_module=reward_module, **env_kwargs)
        rng_key = jax.random.PRNGKey(0)
        num_envs = 1000
        env_params = env.init(rng_key)

        fwd_policy = DummyPolicy(env.action_space.n)
        fwd_policy_params = None

        traj_data, info = forward_rollout(
            rng_key, num_envs, fwd_policy, fwd_policy_params, env, env_params
        )
        return {
            "reward_module": reward_module,
            "env": env,
            "env_params": env_params,
            "rng_key": rng_key,
            "num_envs": num_envs,
            "traj_data": traj_data,
            "info": info,
        }

    def test_forward_rollout_shape(self, setup_forward_rollout: dict[str, Any]):
        """Test that forward rollout returns correctly shaped trajectory data."""
        traj_data = setup_forward_rollout["traj_data"]
        num_envs = setup_forward_rollout["num_envs"]
        env = setup_forward_rollout["env"]

        # Check trajectory shapes
        chex.assert_tree_shape_prefix(
            traj_data.obs, (num_envs, env.max_steps_in_episode + 1)
        )
        chex.assert_tree_shape_prefix(
            traj_data.state, (num_envs, env.max_steps_in_episode + 1)
        )
        chex.assert_shape(traj_data.action, (num_envs, env.max_steps_in_episode + 1))
        chex.assert_shape(
            traj_data.log_gfn_reward, (num_envs, env.max_steps_in_episode + 1)
        )
        chex.assert_shape(traj_data.done, (num_envs, env.max_steps_in_episode + 1))
        chex.assert_shape(traj_data.pad, (num_envs, env.max_steps_in_episode + 1))
        chex.block_until_chexify_assertions_complete()

    def test_backward_rollout_shape(self, setup_forward_rollout: dict[str, Any]):
        """Test that backward rollout returns correctly shaped trajectory data."""
        env = setup_forward_rollout["env"]
        env_params = setup_forward_rollout["env_params"]
        rng_key = setup_forward_rollout["rng_key"]
        num_envs = setup_forward_rollout["num_envs"]

        # Create a dummy policy
        bwd_policy = DummyPolicy(env.backward_action_space.n)
        policy_params = None

        # Create a terminal state for backward rollout
        traj_data = setup_forward_rollout["traj_data"]
        done_indices = jnp.argmax(traj_data.done, axis=1)
        terminating_states = jax.tree.map(
            lambda x: x[jnp.arange(num_envs), done_indices],
            traj_data.state,
        )

        traj_data, _ = backward_rollout(
            rng_key,
            terminating_states,
            bwd_policy,
            policy_params,
            env,
            env_params,
        )

        # Check trajectory shapes
        chex.assert_tree_shape_prefix(
            traj_data.obs, (num_envs, env.max_steps_in_episode + 1)
        )
        chex.assert_tree_shape_prefix(
            traj_data.state, (num_envs, env.max_steps_in_episode + 1)
        )
        chex.assert_shape(traj_data.action, (num_envs, env.max_steps_in_episode + 1))
        chex.assert_shape(
            traj_data.log_gfn_reward, (num_envs, env.max_steps_in_episode + 1)
        )
        chex.assert_shape(traj_data.done, (num_envs, env.max_steps_in_episode + 1))
        chex.assert_shape(traj_data.pad, (num_envs, env.max_steps_in_episode + 1))
        chex.block_until_chexify_assertions_complete()

    def test_forward_rollout_validity(self, setup_forward_rollout: dict[str, Any]):
        """Test that forward rollout generates valid trajectories."""
        num_envs = setup_forward_rollout["num_envs"]
        traj_data = setup_forward_rollout["traj_data"]

        # Check that all trajectories start from initial state
        chex.assert_equal(
            jnp.all(traj_data.state.is_initial[:, 0]),
            True,
        )

        # Check that all trajectories either reach terminal state or max steps
        chex.assert_equal(
            jnp.all(jnp.any(traj_data.done, axis=1)),
            True,
        )

        # Check that padding follows done states
        done_indices = jnp.argmax(traj_data.done, axis=1)
        pads = traj_data.pad[jnp.arange(num_envs), done_indices + 1]
        chex.assert_equal(jnp.all(pads), True)
        chex.block_until_chexify_assertions_complete()

    def test_backward_rollout_validity(self, setup_forward_rollout: dict[str, Any]):
        """Test that backward rollout generates valid trajectories."""
        env = setup_forward_rollout["env"]
        env_params = setup_forward_rollout["env_params"]
        rng_key = setup_forward_rollout["rng_key"]
        num_envs = setup_forward_rollout["num_envs"]
        traj_data = setup_forward_rollout["traj_data"]

        # Create a dummy policy
        bwd_policy = DummyPolicy(env.backward_action_space.n)
        policy_params = None

        # Create a terminal state for backward rollout
        done_indices = jnp.argmax(traj_data.done, axis=1)
        terminating_states = jax.tree.map(
            lambda x: x[jnp.arange(num_envs), done_indices],
            traj_data.state,
        )
        terminal_states = jax.tree.map(
            lambda x: x[jnp.arange(num_envs), done_indices + 1],
            traj_data.state,
        )

        # Check that all trajectories start from terminal state
        chex.assert_equal(jnp.all(terminal_states.is_terminal), True)

        # Check that all trajectories end with a pad
        chex.assert_equal(
            jnp.all(traj_data.pad[jnp.arange(num_envs), done_indices + 1]),
            True,
        )

        traj_data, _ = backward_rollout(
            rng_key,
            terminating_states,
            bwd_policy,
            policy_params,
            env,
            env_params,
        )

        # Check that all trajectories reach initial state
        chex.assert_equal(jnp.all(jnp.any(traj_data.state.is_initial, axis=1)), True)
        chex.block_until_chexify_assertions_complete()

    def test_forward_backward_consistency(self, setup_forward_rollout: dict[str, Any]):
        """Test consistency between forward and backward actions."""
        env = setup_forward_rollout["env"]
        env_params = setup_forward_rollout["env_params"]
        num_envs = setup_forward_rollout["num_envs"]
        fwd_traj_data = setup_forward_rollout["traj_data"]

        def consistency_check(t, carry):
            traj_data, mask_cond, consistency_cond = carry
            state = jax.tree.map(lambda x: x[:, t], traj_data.state)
            action = traj_data.action[:, t]
            next_state = jax.tree.map(lambda x: x[:, t + 1], traj_data.state)
            invalid_mask = env.get_invalid_backward_mask(next_state, env_params)
            bwd_action = env.get_backward_action(state, action, next_state, env_params)
            # Check that backward action is valid
            mask_check = invalid_mask[jnp.arange(num_envs), bwd_action]
            mask_cond |= jax.lax.select(
                next_state.is_pad,
                jnp.zeros_like(mask_check, dtype=jnp.bool_),
                mask_check,
            )

            _, cur_state, _, _, _ = env.backward_step(
                next_state, bwd_action, env_params
            )

            # Compare each attribute and reduce to single bool array
            equality_per_attr = jax.tree.map(lambda x, y: x == y, state, cur_state)
            reduced_per_attr = jax.tree.map(
                lambda x: x.reshape(num_envs, -1).all(axis=1) if x.ndim > 1 else x,
                equality_per_attr,
            )
            reduced_to_bool = jax.tree.reduce(lambda x, y: x & y, reduced_per_attr)

            consistency_cond &= jax.lax.select(
                next_state.is_pad,
                jnp.ones_like(reduced_to_bool, dtype=jnp.bool_),
                reduced_to_bool,
            )

            return traj_data, mask_cond, consistency_cond

        mask_cond = jnp.zeros(num_envs, dtype=jnp.bool_)
        consistency_cond = jnp.ones(num_envs, dtype=jnp.bool_)
        jax.lax.fori_loop(
            0,
            env.max_steps_in_episode,
            consistency_check,
            (fwd_traj_data, mask_cond, consistency_cond),
        )
        jax.effects_barrier()
        chex.assert_trees_all_equal(
            mask_cond,
            jnp.zeros_like(mask_cond, dtype=jnp.bool),
        )
        chex.assert_trees_all_equal(
            consistency_cond,
            jnp.ones_like(consistency_cond, dtype=jnp.bool_),
        )
        chex.block_until_chexify_assertions_complete()

    def test_reward(self, setup_forward_rollout: dict[str, Any]):
        """Test log_reward functionality"""
        traj_data = setup_forward_rollout["traj_data"]

        # Check that log_reward is finite
        chex.assert_equal(
            jnp.all(jnp.isfinite(traj_data.log_gfn_reward)),
            True,
        )
        chex.block_until_chexify_assertions_complete()
