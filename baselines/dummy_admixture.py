"""Baseline training loop for the admixture graph environment."""

import logging
import math
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
from tensorboardX import SummaryWriter

import gfnx
from gfnx.environment.admixture import AdmixtureGraphEnvironment
from gfnx.reward.admixture import AdmixtureGraphRewardModule
from gfnx.utils.masking import mask_logits

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
    log_z: chex.Array


def _policy_wrapper(policy_static: Any):
    def _apply(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params):
        policy = eqx.combine(policy_params, policy_static)
        outputs = policy(rng_key, env_obs)
        return outputs["fwd_logits"], outputs

    return _apply


def validation_pass(train_state: TrainState, num_envs: int) -> tuple[TrainState, dict[str, chex.Array]]:
    """Run a validation rollout with the current policy parameters."""

    rng_key, eval_key = jax.random.split(train_state.rng_key)
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)
    policy_fn = _policy_wrapper(policy_static)
    traj_data, _ = gfnx.utils.forward_rollout(
        rng_key=eval_key,
        num_envs=num_envs,
        policy_fn=policy_fn,
        policy_params=policy_params,
        env=train_state.env,
        env_params=train_state.env_params,
    )

    transition_mask = jnp.logical_not(traj_data.pad[:, :-1])
    log_rewards = jnp.sum(traj_data.log_gfn_reward, axis=1)
    traj_lengths = jnp.sum(transition_mask, axis=1)
    metrics = {
        "mean_log_reward": jnp.mean(log_rewards),
        "max_log_reward": jnp.max(log_rewards),
        "min_log_reward": jnp.min(log_rewards),
        "std_log_reward": jnp.std(log_rewards),
        "mean_traj_length": jnp.mean(traj_lengths),
    }

    updated_state = train_state._replace(rng_key=rng_key)
    return updated_state, metrics


@eqx.filter_jit
def train_step(
    idx: chex.Array, train_state: TrainState
) -> tuple[TrainState, dict[str, chex.Array]]:
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

    policy_fn = _policy_wrapper(policy_static)

    def loss_fn(trainable_params, rollout_key):
        policy_params, log_z = trainable_params["policy"], trainable_params["log_z"]
        traj_data, _ = gfnx.utils.forward_rollout(
            rng_key=rollout_key,
            num_envs=train_state.config.num_envs,
            policy_fn=policy_fn,
            policy_params=policy_params,
            env=train_state.env,
            env_params=train_state.env_params,
        )

        transition_mask = jnp.logical_not(traj_data.pad[:, :-1])
        masked_forward_logprob = jnp.where(
            transition_mask,
            traj_data.info["sampled_log_prob"][:, :-1],
            0.0,
        )
        forward_logprob_sum = jnp.sum(masked_forward_logprob, axis=1)
        log_rewards = jnp.sum(traj_data.log_gfn_reward, axis=1)
        entropy_terms = jnp.where(
            transition_mask,
            traj_data.info["entropy"][:, :-1],
            0.0,
        )
        traj_entropy = jnp.sum(entropy_terms, axis=1)

        current_state = jax.tree.map(lambda x: x[:, :-1], traj_data.state)
        next_state = jax.tree.map(lambda x: x[:, 1:], traj_data.state)
        action = traj_data.action[:, :-1]

        def flatten_state(state_slice):
            return jax.tree.map(
                lambda arr: arr.reshape((-1,) + arr.shape[2:]), state_slice
            )

        current_state_flat = flatten_state(current_state)
        next_state_flat = flatten_state(next_state)
        action_flat = action.reshape(-1)

        backward_actions_flat = train_state.env.get_backward_action(
            current_state_flat,
            action_flat,
            next_state_flat,
            train_state.env_params,
        )
        backward_actions = backward_actions_flat.reshape(action.shape)

        invalid_backward_mask_flat = train_state.env.get_invalid_backward_mask(
            next_state_flat, train_state.env_params
        )
        backward_logits = traj_data.info["bwd_logits"][:, 1:, :]
        num_backward_actions = backward_logits.shape[-1]
        backward_logits_flat = backward_logits.reshape((-1, num_backward_actions))
        masked_backward_logits = mask_logits(
            backward_logits_flat, invalid_backward_mask_flat
        )
        backward_log_probs_flat = jax.nn.log_softmax(
            masked_backward_logits, axis=-1
        )
        chosen_backward_logprob_flat = jnp.take_along_axis(
            backward_log_probs_flat,
            backward_actions_flat[..., None],
            axis=-1,
        ).squeeze(-1)
        chosen_invalid_mask_flat = jnp.take_along_axis(
            invalid_backward_mask_flat,
            backward_actions_flat[..., None],
            axis=-1,
        ).squeeze(-1)
        chosen_backward_logprob_flat = jnp.where(
            chosen_invalid_mask_flat, 0.0, chosen_backward_logprob_flat
        )
        _ = jax.lax.cond(
            jnp.any(chosen_invalid_mask_flat),
            lambda _: (
                debug.print(
                    "invalid backward transitions detected: {}",
                    jnp.sum(chosen_invalid_mask_flat),
                ),
                jnp.array(0, dtype=jnp.int32),
            )[1],
            lambda _: jnp.array(0, dtype=jnp.int32),
            operand=jnp.array(0, dtype=jnp.int32),
        )
        backward_logprob = chosen_backward_logprob_flat.reshape(action.shape)
        masked_backward_logprob = jnp.where(transition_mask, backward_logprob, 0.0)
        backward_logprob_sum = jnp.sum(masked_backward_logprob, axis=1)

        tb_residual = log_z + forward_logprob_sum - log_rewards - backward_logprob_sum
        tb_mse = tb_residual**2
        entropy_coef = float(train_state.config.policy.entropy_coef)
        loss = jnp.mean(tb_mse) - entropy_coef * jnp.mean(traj_entropy)

        metrics = {
            "loss": loss,
            "tb_mse": jnp.mean(tb_mse),
            "mean_log_reward": jnp.mean(log_rewards),
            "mean_forward_logprob": jnp.mean(forward_logprob_sum),
            "mean_backward_logprob": jnp.mean(backward_logprob_sum),
            "mean_entropy": jnp.mean(traj_entropy),
            "mean_tb_residual": jnp.mean(tb_residual),
            "abs_tb_residual": jnp.mean(jnp.abs(tb_residual)),
            "log_z": log_z,
            "mean_traj_length": jnp.mean(jnp.sum(transition_mask, axis=1)),
        }
        return loss, metrics

    trainable_params = {
        "policy": policy_params,
        "log_z": train_state.log_z,
    }
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        trainable_params, sample_traj_key
    )

    updates, opt_state = train_state.optimizer.update(
        grads, train_state.opt_state, params=trainable_params
    )
    trainable_params = optax.apply_updates(trainable_params, updates)
    model = eqx.combine(trainable_params["policy"], policy_static)

    num_steps = jnp.int32(train_state.config.num_train_steps)
    print_every = jnp.int32(train_state.config.logging["tqdm_print_rate"])
    should_log = jnp.logical_or(
        jnp.equal(jnp.mod(idx, print_every), 0),
        jnp.equal(idx + 1, num_steps),
    )

    def _print(_: chex.Array) -> chex.Array:
        debug.print(
            (
                "step {}/{} loss={:.3f} tb_mse={:.3f} log_reward={:.3f} "
                "log_pf={:.3f} log_pb={:.3f} log_z={:.3f}"
            ),
            idx + 1,
            num_steps,
            metrics["loss"],
            metrics["tb_mse"],
            metrics["mean_log_reward"],
            metrics["mean_forward_logprob"],
            metrics["mean_backward_logprob"],
            metrics["log_z"],
        )
        return jnp.array(0, dtype=jnp.int32)

    _ = jax.lax.cond(
        should_log,
        _print,
        lambda _: jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )

    new_state = TrainState(
        rng_key=rng_key,
        config=train_state.config,
        env=train_state.env,
        env_params=train_state.env_params,
        model=model,
        opt_state=opt_state,
        optimizer=train_state.optimizer,
        log_z=trainable_params["log_z"],
    )
    return new_state, metrics


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
    init_log_z = jnp.array(0.0, dtype=jnp.float32)
    trainable_template = {"policy": params, "log_z": init_log_z}
    opt_state = optimizer.init(trainable_template)
    log.info("Optimiser initialised with learning_rate=%s", cfg.optimizer.learning_rate)

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
        opt_state=opt_state,
        optimizer=optimizer,
        log_z=init_log_z,
    )

    tensorboard_dir = Path(cfg.logging.tensorboard_dir)
    if not tensorboard_dir.is_absolute():
        tensorboard_dir = Path.cwd() / tensorboard_dir
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    log.info("TensorBoard summaries will be written to %s", tensorboard_dir)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    total_steps = int(cfg.num_train_steps)
    steps_per_phase = int(cfg.training.train_steps_per_phase)
    if steps_per_phase <= 0:
        raise ValueError("training.train_steps_per_phase must be positive")
    num_phases = max(1, math.ceil(total_steps / steps_per_phase))

    log.info(
        "Start trajectory-balance training for %d steps (%d phases) with %d environments",
        total_steps,
        num_phases,
        cfg.num_envs,
    )

    global_step = 0
    try:
        for phase_idx in range(num_phases):
            steps_remaining = total_steps - global_step
            if steps_remaining <= 0:
                break
            phase_steps = min(steps_per_phase, steps_remaining)
            log.info(
                "Phase %d/%d: running %d gradient steps", phase_idx + 1, num_phases, phase_steps
            )

            for _ in range(phase_steps):
                train_state, metrics = train_step(jnp.int32(global_step), train_state)
                metrics = jax.tree_map(lambda x: float(x), jax.device_get(metrics))
                writer.add_scalar("train/loss", metrics["loss"], global_step)
                writer.add_scalar("train/tb_mse", metrics["tb_mse"], global_step)
                writer.add_scalar("train/log_reward", metrics["mean_log_reward"], global_step)
                writer.add_scalar(
                    "train/forward_logprob", metrics["mean_forward_logprob"], global_step
                )
                writer.add_scalar(
                    "train/backward_logprob", metrics["mean_backward_logprob"], global_step
                )
                writer.add_scalar("train/log_z", metrics["log_z"], global_step)
                writer.add_scalar("train/entropy", metrics["mean_entropy"], global_step)
                writer.add_scalar(
                    "train/tb_residual_abs", metrics["abs_tb_residual"], global_step
                )
                writer.add_scalar(
                    "train/trajectory_length", metrics["mean_traj_length"], global_step
                )
                global_step += 1

            train_state, val_metrics = validation_pass(
                train_state, num_envs=int(cfg.validation.num_envs)
            )
            val_metrics = jax.tree_map(lambda x: float(x), jax.device_get(val_metrics))
            log.info(
                (
                    "Validation %d/%d: mean_log_reward=%.3f ± %.3f (min=%.3f, max=%.3f) "
                    "mean_traj_length=%.2f"
                ),
                phase_idx + 1,
                num_phases,
                val_metrics["mean_log_reward"],
                val_metrics["std_log_reward"],
                val_metrics["min_log_reward"],
                val_metrics["max_log_reward"],
                val_metrics["mean_traj_length"],
            )
            writer.add_scalar(
                "validation/mean_log_reward", val_metrics["mean_log_reward"], global_step
            )
            writer.add_scalar(
                "validation/std_log_reward", val_metrics["std_log_reward"], global_step
            )
            writer.add_scalar(
                "validation/max_log_reward", val_metrics["max_log_reward"], global_step
            )
            writer.add_scalar(
                "validation/min_log_reward", val_metrics["min_log_reward"], global_step
            )
            writer.add_scalar(
                "validation/mean_traj_length", val_metrics["mean_traj_length"], global_step
            )
            writer.flush()
    finally:
        writer.close()
        reward_module.close()


if __name__ == "__main__":
    run_experiment()
