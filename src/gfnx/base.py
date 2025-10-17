"""Abstract base class for all gfnx Environments"""

# TODO: add credits to gymnax

from typing import Any, Dict, Generic, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

import os, shutil, tempfile
import networkx as nx
from jax import random
from jax.scipy.special import multigammaln
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
import jax
jax.config.update("jax_enable_x64", True) 

TEnvironment = TypeVar("TEnvironment", bound="BaseVecEnvironment")
TEnvParams = TypeVar("TEnvParams", bound="BaseEnvParams")

TObs = chex.ArrayTree
TEnvState = TypeVar("TEnvState", bound="BaseEnvState")
TAction = chex.Array
TBackwardAction = chex.Array

TRewardModule = TypeVar("TRewardModule", bound="BaseRewardModule")
TRewardParams = TypeVar("TRewardParams")
TLogReward = chex.Array
TReward = chex.Array
TDone = chex.Array


@chex.dataclass(frozen=True)
class BaseEnvState:
    time: Int[Array, " batch_size"]
    is_terminal: Bool[Array, " batch_size"]
    is_initial: Bool[Array, " batch_size"]
    is_pad: Bool[Array, " batch_size"]


@chex.dataclass(frozen=True)
class BaseEnvParams:
    reward_params: TRewardParams





class BaseRewardModule(Generic[TEnvState, TEnvParams]):
    """
    Base class for reward and log reward implementations.

    This class defines the interface for reward modules, which are
    responsible for computing rewards and log rewards given the state of
    the environment and its parameters.
    Subclasses should implement the following methods:
        - init: Initialize the reward module and return its parameters.
        - log_reward: Compute the log reward given the state and environment
          parameters.
        - reward: Compute the reward given the state and environment
          parameters.
    """

    def init(self, rng_key: chex.PRNGKey, dummy_state: TEnvState) -> TRewardParams:
        """
        Initialize reward module, returns TRewardParams.
        Args:
        - rng_key: chex.PRNGKey, random key
        - dummy_state: TEnvState, shape [B, ...], batch of dummy states
        """
        raise NotImplementedError

    def log_reward(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Float[Array, " batch_size"]:
        """
        Compute the log reward given the state and environment parameters.
        Args:
        - state: TEnvState, shape [B, ...], batch of states
        - env_params: TEnvParams, params of environment,
          always includes reward params
        Returns:
        - TLogReward, shape [B, ...], batch of log rewards
        """
        # return self.reward(state, env_params)
        raise NotImplementedError

    def reward(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Float[Array, " batch_size"]:
        """
        Log reward function, returns TReward
        Args:
        - state: TEnvState, shape [B, ...], batch of states
        - env_params: TEnvParams, params of environment,
          always includes reward params
        Returns:
        - TReward, shape [B, ...], batch of rewards
        """
        raise NotImplementedError


class BaseVecEnvironment(Generic[TEnvState, TEnvParams]):
    """
    Jittable abstract base class for all gfnx Environments.
    Note: all environments are vectorized by default.

    Args:
    - reward_module: TRewardModule, reward module
    """

    def __init__(self, reward_module: TRewardModule):
        self.reward_module = reward_module

    def get_init_state(self, num_envs: int) -> TEnvState:
        """Returns batch of initial states of the environment."""
        raise NotImplementedError

    def init(self, rng_key: chex.PRNGKey) -> TEnvParams:
        """
        Init params of the environment and reward module.
        """
        raise NotImplementedError

    @property
    def max_steps_in_episode(self) -> int:
        raise NotImplementedError

    def step(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> Tuple[TObs, TEnvState, TLogReward, TDone, Dict[Any, Any]]:
        """Performs batched step transitions in the environment."""
        next_state, done, info = self.transition(state, action, env_params)
        done = jnp.astype(done, jnp.bool)  # Ensure that done is boolean
        # Compute reward only for a states that became terminal on this step
        new_dones = jnp.logical_and(done, jnp.logical_not(state.is_terminal))

        # print('OOLOLLLO')
        # assert False

        # Since computation of log rewards is expensive, we do it only if at
        # least one of the environments is done
        log_reward = jax.lax.cond(
            jnp.any(new_dones),
            self.reward_module.log_reward,
            lambda state, _: jnp.zeros_like(state.time, dtype=jnp.float32),
            next_state,  # Args for log_reward
            env_params,  # Args for log_reward
        )
        log_reward = jnp.where(new_dones, log_reward, jnp.zeros_like(log_reward))
        return (
            self.get_obs(next_state, env_params),
            next_state,
            log_reward,
            done,
            info,
        )

    def backward_step(
        self,
        state: TEnvState,
        backward_action: TBackwardAction,
        env_params: TEnvParams,
    ) -> Tuple[TObs, TEnvState, TLogReward, TDone, Dict[Any, Any]]:
        """
        Performs batched backward step transitions in the environment.
        Important: `done` is true if the state is the initial one.
        """
        state, done, info = self.backward_transition(state, backward_action, env_params)
        done = jnp.astype(done, jnp.bool)  # Ensure that done is boolean
        # log reward is always zero for backward steps
        log_rewards = jnp.zeros(state.time.shape, dtype=jnp.float32)
        return self.get_obs(state, env_params), state, log_rewards, done, info

    def reset(self, num_envs: int, env_params: TEnvParams) -> Tuple[TObs, TEnvState]:
        """Performs batched resetting of environment."""
        state = self.get_init_state(num_envs)
        return self.get_obs(state, env_params), state

    def transition(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> Tuple[TEnvState, TDone, Dict[Any, Any]]:
        """Environment-specific step transition."""
        next_state, done, info = jax.vmap(
            self._single_transition, in_axes=(0, 0, None)
        )(state, action, env_params)
        return next_state, done, info

    def backward_transition(
        self,
        state: TEnvState,
        backward_action: TAction,
        env_params: TEnvParams,
    ) -> Tuple[TEnvState, TDone, Dict[Any, Any]]:
        """Environment-specific step backward transition."""
        prev_state, done, info = jax.vmap(
            self._single_backward_transition, in_axes=(0, 0, None)
        )(state, backward_action, env_params)
        return prev_state, done, info

    def _single_transition(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> Tuple[TEnvState, TDone, Dict[Any, Any]]:
        """Environment-specific step transition. NOTE: this is not batched!"""
        raise NotImplementedError

    def _single_backward_transition(
        self,
        state: TEnvState,
        backward_action: TAction,
        env_params: TEnvParams,
    ) -> Tuple[TEnvState, TDone, Dict[Any, Any]]:
        """
        Environment-specific step backward transition.
        NOTE: this is not batched!
        """
        raise NotImplementedError

    def get_obs(self, state: TEnvState, env_params: TEnvParams) -> chex.ArrayTree:
        """Applies observation function to state. Should be batched."""
        raise NotImplementedError

    def get_backward_action(
        self,
        state: TEnvState,
        forward_action: TAction,
        next_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """
        Returns backward action given the complete characterization of the
        forward transition. Should be batched.
        """
        raise NotImplementedError

    def get_forward_action(
        self,
        state: TEnvState,
        backward_action: TAction,
        prev_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """
        Returns forward action given the complete characterization of the
        backward transition. Should be batched.
        """
        raise NotImplementedError

    def get_invalid_mask(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, " batch_size"]:
        """Returns mask of invalid actions. Should be batched"""
        raise NotImplementedError

    def get_invalid_backward_mask(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, " batch_size"]:
        """Returns mask of invalid backward actions. Should be batched."""
        raise NotImplementedError

    def sample_action(
        self, rng_key: chex.PRNGKey, policy_probs: chex.Array
    ) -> Int[Array, " batch_size"]:
        """
        Helping function for sampling actions from policy.
        """
        batch_size = policy_probs.shape[0]
        return jax.vmap(
            lambda key, p: jax.random.choice(key, self.action_space.n, p=p),
            in_axes=(0, 0),
        )(jax.random.split(rng_key, batch_size), policy_probs)

    def sample_backward_action(
        self,
        rng_key: chex.PRNGKey,
        policy_probs: chex.Array,
    ) -> Int[Array, " batch_size"]:
        """
        Helping function for sampling actions from policy.
        """
        batch_size = policy_probs.shape[0]
        return jax.vmap(
            lambda key, p: jax.random.choice(key, self.backward_action_space.n, p=p),
            in_axes=(0, 0),
        )(jax.random.split(rng_key, batch_size), policy_probs)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def action_space(self):
        """Action space of the environment."""
        raise NotImplementedError

    @property
    def backward_action_space(self):
        """Action space of the environment."""
        raise NotImplementedError

    @property
    def observation_space(self):
        """Observation space of the environment."""
        raise NotImplementedError

    @property
    def state_space(self):
        """State space of the environment."""
        raise NotImplementedError
