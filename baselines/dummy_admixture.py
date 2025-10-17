"""Baseline training loop for the admixture graph environment."""

import functools
import logging
from pathlib import Path
from typing import Any, NamedTuple

import chex
import equinox as eqx
import hydra
import hydra.utils as hydra_utils
import jax
from jax import debug
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

import gfnx
from gfnx.environment.admixture import AdmixtureGraphEnvironment
from gfnx.reward.admixture import AdmixtureGraphRewardModule

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class DummyPolicy(eqx.Module):
    """A simple MLP policy producing forward/backward logits."""

    encoder: eqx.nn.MLP
    fwd_head: eqx.nn.Linear
    bwd_head: eqx.nn.Linear

    def __init__(
        self,
        *,
        n_fwd_actions: int,
        n_bwd_actions: int,
        obs_dim: int,
        hidden_size: int,
        depth: int,
        key: jax.random.PRNGKey,
    ) -> None:
        keys = jax.random.split(key, 3)
        self.encoder = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=depth,
            key=keys[0],
        )
        self.fwd_head = eqx.nn.Linear(hidden_size, n_fwd_actions, key=keys[1])
        self.bwd_head = eqx.nn.Linear(hidden_size, n_bwd_actions, key=keys[2])

    def __call__(self, rng_key: jax.random.PRNGKey, obs: chex.Array) -> dict[str, chex.Array]:
        del rng_key  # Policy is deterministic given observations.
        obs_flat = obs.reshape(obs.shape[0], -1)
        hidden = jax.vmap(self.encoder)(obs_flat)
        hidden = jax.nn.relu(hidden)
        fwd_logits = jax.vmap(self.fwd_head)(hidden)
        bwd_logits = jax.vmap(self.bwd_head)(hidden)
        return {"fwd_logits": fwd_logits, "bwd_logits": bwd_logits}


class TrainState(NamedTuple):
    rng_key: jax.random.PRNGKey
    config: OmegaConf
    env: AdmixtureGraphEnvironment
    env_params: chex.Array
    model: DummyPolicy
    opt_state: optax.OptState
    optimizer: optax.GradientTransformation
    baseline: chex.Array


def _policy_wrapper(policy_static: Any):
    def _apply(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params):
        policy = eqx.combine(policy_params, policy_static)
        outputs = policy(rng_key, env_obs)
        return outputs["fwd_logits"], outputs

    return _apply


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

    policy_fn = _policy_wrapper(policy_static)

    def loss_fn(policy_params, rollout_key):
        traj_data, _ = gfnx.utils.forward_rollout(
            rng_key=rollout_key,
            num_envs=train_state.config.num_envs,
            policy_fn=policy_fn,
            policy_params=policy_params,
            env=train_state.env,
            env_params=train_state.env_params,
        )

        mask = jnp.logical_not(traj_data.pad)
        traj_logprob = jnp.sum(
            traj_data.info["sampled_log_prob"] * mask,
            axis=1,
        )
        traj_entropy = jnp.sum(traj_data.info["entropy"] * mask, axis=1)
        log_rewards = jnp.sum(traj_data.log_gfn_reward, axis=1)

        baseline = jax.lax.stop_gradient(train_state.baseline)
        advantage = log_rewards - baseline

        entropy_coef = float(train_state.config.policy.entropy_coef)
        loss = -jnp.mean(advantage * traj_logprob + entropy_coef * traj_entropy)

        metrics = {
            "mean_log_reward": jnp.mean(log_rewards),
            "mean_entropy": jnp.mean(traj_entropy),
            "mean_logprob": jnp.mean(traj_logprob),
            "loss": loss,
        }
        return loss, metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        policy_params, sample_traj_key
    )

    updates, opt_state = train_state.optimizer.update(
        grads, train_state.opt_state, params=policy_params
    )
    policy_params = optax.apply_updates(policy_params, updates)
    model = eqx.combine(policy_params, policy_static)

    momentum = float(train_state.config.policy.baseline_momentum)
    new_baseline = (1.0 - momentum) * train_state.baseline + momentum * metrics[
        "mean_log_reward"
    ]

    num_steps = jnp.int32(train_state.config.num_train_steps)
    print_every = jnp.int32(train_state.config.logging["tqdm_print_rate"])
    should_log = jnp.logical_or(
        jnp.equal(jnp.mod(idx, print_every), 0),
        jnp.equal(idx + 1, num_steps),
    )

    def _print(_: chex.Array) -> chex.Array:
        debug.print(
            "step {}/{} loss={:.3f} log_reward={:.3f} entropy={:.3f}",
            idx + 1,
            num_steps,
            loss,
            metrics["mean_log_reward"],
            metrics["mean_entropy"],
        )
        return jnp.array(0, dtype=jnp.int32)

    _ = jax.lax.cond(should_log, _print, lambda _: jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

    return TrainState(
        rng_key=rng_key,
        config=train_state.config,
        env=train_state.env,
        env_params=train_state.env_params,
        model=model,
        opt_state=opt_state,
        optimizer=train_state.optimizer,
        baseline=new_baseline,
    )


@hydra.main(
    config_path="configs/", config_name="dummy_admixture", version_base=None
)
def run_experiment(cfg: OmegaConf) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    rng_key = jax.random.PRNGKey(cfg.seed)
    rng_key, env_init_key, policy_init_key = jax.random.split(rng_key, 3)

    snp_path = None
    if cfg.reward.snp_path:
        candidate = Path(hydra_utils.to_absolute_path(cfg.reward.snp_path))
        if candidate.is_file():
            snp_path = candidate
        else:
            log.warning(
                "Requested SNP dataset '%s' not found. Falling back to ArcticData.txt.",
                candidate,
            )

    reward_module = AdmixtureGraphRewardModule(snp_path=snp_path)
    log.info(
        "Initialising reward module with SNP override: %s",
        snp_path if snp_path is not None else "<default ArcticData.txt>",
    )
    env = AdmixtureGraphEnvironment(
        num_leaves=cfg.environment.num_leaves,
        num_admx_nodes=cfg.environment.num_admx_nodes,
        reward_module=reward_module,
    )
    env_params = env.init(env_init_key)

    num_nodes = env.get_init_state(1).adjacency_matrix.shape[-1]
    obs_dim = num_nodes * num_nodes
    log.info(
        "Environment ready: %d nodes, forward actions=%d, backward actions=%d",
        num_nodes,
        env.action_space.n,
        env.backward_action_space.n,
    )
    model = DummyPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        obs_dim=obs_dim,
        hidden_size=cfg.policy.hidden_size,
        depth=cfg.policy.depth,
        key=policy_init_key,
    )
    log.info(
        "Policy parameters: hidden_size=%d depth=%d", cfg.policy.hidden_size, cfg.policy.depth
    )

    optimizer = optax.adam(cfg.optimizer.learning_rate)
    params, _ = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)
    log.info("Optimiser initialised with learning_rate=%s", cfg.optimizer.learning_rate)

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
        opt_state=opt_state,
        optimizer=optimizer,
        baseline=jnp.array(0.0, dtype=jnp.float32),
    )

    train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,))
    def train_step_wrapper(idx: int, train_state_params):
        train_state = eqx.combine(train_state_params, train_state_static)
        train_state = train_step(idx, train_state)
        train_state_params, _ = eqx.partition(train_state, eqx.is_array)
        return train_state_params

    log.info("Start training for %d steps with %d environments", cfg.num_train_steps, cfg.num_envs)
    train_state_params = jax.lax.fori_loop(
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper,
        init_val=train_state_params,
    )

    final_state = eqx.combine(train_state_params, train_state_static)
    final_params, final_static = eqx.partition(final_state.model, eqx.is_array)
    final_traj, _ = gfnx.utils.forward_rollout(
        rng_key=final_state.rng_key,
        num_envs=cfg.num_envs,
        policy_fn=_policy_wrapper(final_static),
        policy_params=final_params,
        env=final_state.env,
        env_params=final_state.env_params,
    )
    final_log_reward = jnp.mean(jnp.sum(final_traj.log_gfn_reward, axis=1))
    log.info("Finished training. Mean terminal log reward %.3f", float(final_log_reward))
    reward_module.close()


if __name__ == "__main__":
    run_experiment()
