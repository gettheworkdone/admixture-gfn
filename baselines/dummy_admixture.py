"""Dummy demonstration of some basic API usage for admixture graph environment."""

import functools
import logging
from typing import NamedTuple

import chex
import equinox as eqx
import hydra
import jax
from jax import debug
import jax.numpy as jnp
from jax_tqdm import loop_tqdm
from omegaconf import OmegaConf

import gfnx
from gfnx.environment.admixture import AdmixtureGraphEnvironment
from gfnx.reward.admixture import AdmixtureGraphRewardModule

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class DummyPolicy(eqx.Module):
    """A dummy policy that returns uniform probabilities over valid actions."""

    n_fwd_actions: int
    n_bwd_actions: int

    def __init__(self, n_fwd_actions, n_bwd_actions):
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions

    def __call__(self, rng_key, obs, policy_params):
        batch_size = obs.shape[0]
        logits = jnp.ones((batch_size, self.n_fwd_actions))
        bwd_logits = jnp.ones((batch_size, self.n_bwd_actions))
        return {"fwd_logits": logits, "bwd_logits": bwd_logits}


class TrainState(NamedTuple):
    rng_key: jax.random.PRNGKey
    config: OmegaConf
    env: AdmixtureGraphEnvironment
    env_params: chex.Array
    model: DummyPolicy


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    # Step 1. Generate a batch of trajectories and split to transitions
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    policy_params, policy_static = eqx.partition(
        train_state.model, eqx.is_array
    )

    def fwd_policy_fn(
        rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params
    ) -> chex.Array:
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(rng_key, env_obs, train_state.model)
        return policy_outputs["fwd_logits"], policy_outputs

    traj_data, log_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=train_state.config.num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params,
        env=train_state.env,
        env_params=train_state.env_params,
    )
    transitions = gfnx.utils.split_traj_to_transitions(traj_data)
    # And then there will be an update of the model...
    return train_state


@hydra.main(
    config_path="configs/", config_name="dummy_admixture", version_base=None
)
def run_experiment(cfg: OmegaConf):
    log.info(OmegaConf.to_yaml(cfg))
    rng_key = jax.random.PRNGKey(cfg.seed)
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)

    reward_module = AdmixtureGraphRewardModule()
    env = AdmixtureGraphEnvironment(
        num_leaves=cfg.environment.num_leaves,
        num_admx_nodes=cfg.environment.num_admx_nodes,
        reward_module=reward_module,
    )
    env_params = env.init(env_init_key)
    # Here will be a neural network policy
    model = DummyPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
    )

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
    )
    train_state_params, train_state_static = eqx.partition(
        train_state, eqx.is_array
    )

    @functools.partial(jax.jit, donate_argnums=(1,))
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging["tqdm_print_rate"])
    def train_step_wrapper(idx: int, train_state_params):
        # Wrapper to use a usual jit in jax, since it is required by fori_loop.
        train_state = eqx.combine(train_state_params, train_state_static)
        train_state = train_step(idx, train_state)
        train_state_params, _ = eqx.partition(train_state, eqx.is_array)
        return train_state_params

    # Run the training loop via jax.lax.fori_loop
    log.info("Start training")
    train_state_params = jax.lax.fori_loop(
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper,
        init_val=train_state_params,
    )


if __name__ == "__main__":
    debug.print("res = UUUUUUUUUUUUUUUUU")

    run_experiment()
