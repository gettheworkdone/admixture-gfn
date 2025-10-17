from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from .. import spaces
from ..base import (
    BaseVecEnvironment,
    TAction,
    TDone,
    TRewardModule,
    TRewardParams,
)


@chex.dataclass(frozen=True)
class EnvState:
    # NOTE: Number of nodes in the constructed graph is always:
    # N = 2 * num_leaves + num_admx_nodes - 1
    adjacency_matrix: chex.Array  # [B, N, N]
    closure_matrix: chex.Array  # [B, N, N] ############################################################# find root here
    num_parents: chex.Array  # [B, N]
    # Number of nodes that are not masked.
    # They can be used in merging. At start, it is equal to num_leaves
    num_unmasked_nodes: chex.Array  # [B]
    # Default attributes
    is_terminal: chex.Array  # [B]
    is_initial: chex.Array  # [B]
    is_pad: chex.Array  # [B]
    time: chex.Array  # [B]


@chex.dataclass(frozen=True)
class EnvParams:
    num_leaves: int  # number of population nodes
    num_admx_nodes: int  # number of admixture nodes
    reward_params: TRewardParams = None


class AdmixtureGraphEnvironment(BaseVecEnvironment[EnvState, EnvParams]):
    def __init__(
        self,
        num_leaves: int,
        num_admx_nodes: int,
        reward_module: TRewardModule,
    ):
        super().__init__(reward_module)
        self.num_leaves = num_leaves
        self.num_admx_nodes = num_admx_nodes

        # Pre-compute triu indices for actions
        indices = jnp.triu_indices(2 * num_leaves + num_admx_nodes - 1, k=1)
        self.lefts = indices[0]
        self.rights = indices[1]

    def get_init_state(self, num_envs: int) -> EnvState:
        """Returns batch of initial states"""
        N = 2 * self.num_leaves + self.num_admx_nodes - 1
        adjacency_matrix = jnp.zeros((num_envs, N, N), dtype=jnp.bool)
        closure_matrix = jnp.tile(
            jnp.eye(N, dtype=jnp.bool), (num_envs, 1, 1)
        )  # All nodes are connected to themselves
        num_parents = jnp.zeros((num_envs, N), dtype=jnp.uint8)
        num_unmasked_nodes = jnp.full(
            (num_envs,), self.num_leaves, dtype=jnp.int32
        )
        return EnvState(
            adjacency_matrix=adjacency_matrix,
            closure_matrix=closure_matrix,
            num_parents=num_parents,
            num_unmasked_nodes=num_unmasked_nodes,
            is_terminal=jnp.zeros((num_envs,), dtype=jnp.bool),
            is_initial=jnp.ones((num_envs,), dtype=jnp.bool),
            is_pad=jnp.zeros((num_envs,), dtype=jnp.bool),
            time=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        """Initialize environment"""
        dummy_state = self.get_init_state(1)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return EnvParams(
            num_leaves=self.num_leaves,
            num_admx_nodes=self.num_admx_nodes,
            reward_params=reward_params,
        )

    def _single_transition(
        self, state: EnvState, action: TAction, env_params: EnvParams
    ) -> Tuple[EnvState, TDone, Dict[str, Any]]:
        """Single environment step transition"""
        is_terminal = state.is_terminal
        time = state.time

        def get_state_terminal() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_nonterminal() -> EnvState:
            done = action == self.action_space.n - 1
            return jax.lax.cond(done, get_state_finished, get_state_inter)

        def get_state_finished() -> EnvState:
            return state.replace(
                time=time + 1, is_terminal=True, is_initial=False
            )

        def get_state_inter() -> EnvState:
            left, right = self.lefts[action], self.rights[action]
            parent = state.num_unmasked_nodes
            """
              p
             / \
            l   r
            """
            adjacency_matrix = (
                state.adjacency_matrix.at[parent, left]
                .set(True)
                .at[parent, right]
                .set(True)
            )
            closure_matrix = state.closure_matrix
            # Connect parent with left
            outer_product = jnp.logical_and(
                jnp.expand_dims(closure_matrix[:, parent], 1),
                jnp.expand_dims(closure_matrix[left], 0),
            )
            closure_matrix = jnp.logical_or(closure_matrix, outer_product)
            # Connect parent with right
            outer_product = jnp.logical_and(
                jnp.expand_dims(closure_matrix[:, parent], 1),
                jnp.expand_dims(closure_matrix[right], 0),
            )
            closure_matrix = jnp.logical_or(closure_matrix, outer_product)
            num_parents = (
                state.num_parents.at[left]
                .set(state.num_parents[left] + 1)
                .at[right]
                .set(state.num_parents[right] + 1)
            )

            return state.replace(
                adjacency_matrix=adjacency_matrix,
                closure_matrix=closure_matrix,
                num_parents=num_parents,
                num_unmasked_nodes=state.num_unmasked_nodes + 1,
                time=time + 1,
                is_terminal=False,
                is_initial=False,
            )

        next_state = jax.lax.cond(
            is_terminal, get_state_terminal, get_state_nonterminal
        )

        return next_state, next_state.is_terminal, {}

    def _single_backward_transition(
        self, state: EnvState, backward_action: TAction, env_params: EnvParams
    ) -> Tuple[EnvState, chex.Array, Dict[str, Any]]:
        """Single environment step backward transition"""
        is_initial = state.is_initial
        time = state.time

        def get_state_initial() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_non_initial() -> EnvState:
            unterminate = backward_action == self.backward_action_space.n - 1
            return jax.lax.cond(
                unterminate, get_state_terminating, get_state_inter
            )

        def get_state_terminating() -> EnvState:
            return state.replace(
                time=time - 1,
                is_terminal=False,
                is_initial=jnp.all(jnp.logical_not(state.adjacency_matrix)),
                is_pad=False,
            )

        def get_state_inter() -> EnvState:
            parent = self.num_leaves + backward_action
            adjacency_matrix = state.adjacency_matrix.at[parent, :].set(False)
            closure_matrix = (
                state.closure_matrix.at[parent, :]
                .set(False)
                .at[parent, parent]
                .set(True)
            )
            left, right = jnp.nonzero(state.adjacency_matrix[parent], size=2)[
                0
            ]
            num_parents = (
                state.num_parents.at[left]
                .set(state.num_parents[left] - 1)
                .at[right]
                .set(state.num_parents[right] - 1)
            )
            # Make a permutation: move empty node to the end
            N = 2 * self.num_leaves + self.num_admx_nodes - 1
            # TODO: make it cleaner. We need a permutation of two elems arr
            perm = jnp.arange(N)
            perm = perm.at[jnp.array([parent, N - 1])].set(
                perm[jnp.array([N - 1, parent])]
            )
            adjacency_matrix = adjacency_matrix[perm][:, perm]
            closure_matrix = closure_matrix[perm][:, perm]
            num_parents = num_parents[perm]
            return state.replace(
                adjacency_matrix=adjacency_matrix,
                closure_matrix=closure_matrix,
                num_parents=num_parents,
                num_unmasked_nodes=state.num_unmasked_nodes - 1,
                time=time - 1,
                is_terminal=False,
                is_initial=jnp.all(jnp.logical_not(adjacency_matrix)),
                is_pad=False,
            )

        prev_state = jax.lax.cond(
            is_initial, get_state_initial, get_state_non_initial
        )

        return prev_state, prev_state.is_initial, {}

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return state.adjacency_matrix

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: TAction,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> TAction:
        """Returns backward action given the forward transition"""
        # NOTE: It seems strange that we do not use the forward action
        return jnp.where(
            forward_action == self.action_space.n - 1,
            self.backward_action_space.n - 1,
            state.num_unmasked_nodes
            - self.num_leaves,  # Idx of the last node minus num_leaves
        )

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: TAction,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> TAction:
        """Returns forward action given the backward transition"""

        # NOTE: It seems strange that we do not use the backward action
        def _single_get_forward_action(
            state: EnvState, prev_state: EnvState
        ) -> TAction:
            parent_row = state.adjacency_matrix[prev_state.num_unmasked_nodes]
            left, right = jnp.nonzero(parent_row, size=2)
            """
            (n - 1 + n - 1 - left - 1) * left // 2 + right
            """
            N = 2 * self.num_leaves + self.num_admx_nodes - 1
            return jnp.where(
                backward_action == self.backward_action_space.n - 1,
                self.action_space.n - 1,
                (N - 1 + N - 1 - left - 1) * left // 2 + right,
            )

        return jax.vmap(_single_get_forward_action)(state, prev_state)

    # TODO: can be implemented without vmap via batched operations
    def get_invalid_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid actions. True means invalid action."""

        def _single_get_invalid_mask(state: EnvState) -> chex.Array:
            # Get the number of all graph nodes
            N = 2 * self.num_leaves + self.num_admx_nodes - 1
            # Get the number of admixture nodes in the graph
            num_admx_nodes = jnp.sum(state.num_parents == 2)
            # Step 1: Check if the number of admixture nodes is maximum
            cond = jnp.logical_and(
                state.num_parents == 1,
                self.num_admx_nodes == num_admx_nodes,
            )
            # Step 2: Check if the node has already two parents
            cond = jnp.logical_or(
                cond,
                state.num_parents == 2,
            )
            # Step 3: Check if the node is masked
            cond = jnp.logical_or(
                cond, jnp.arange(N) >= state.num_unmasked_nodes
            )
            merge_mask = jnp.logical_or(cond[self.lefts], cond[self.rights])
            stop_mask = jnp.logical_not(
                jnp.any(
                    jnp.all(
                        state.closure_matrix,
                        axis=1,
                    ),
                    keepdims=True,
                ),
            )
            return jnp.concatenate([merge_mask, stop_mask])

        return jax.vmap(_single_get_invalid_mask)(state)

    # TODO: can be implemented without vmap via batched operations
    def get_invalid_backward_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid backward actions. True means invalid action."""

        def _single_get_invalid_backward_mask(state: EnvState) -> chex.Array:
            N = 2 * self.num_leaves + self.num_admx_nodes - 1

            def get_mask_terminal() -> chex.Array:
                return jnp.concatenate(
                    [
                        jnp.ones((N - self.num_leaves,), dtype=jnp.bool),
                        jnp.zeros((1,), dtype=jnp.bool),
                    ]
                )

            def get_mask_nonterminal() -> chex.Array:
                has_parent = state.adjacency_matrix[
                    self.num_leaves :, self.num_leaves :
                ].any(axis=0)
                is_masked = jnp.arange(N - self.num_leaves) >= (
                    state.num_unmasked_nodes - self.num_leaves
                )
                mask = jnp.logical_or(has_parent, is_masked)

                return jnp.concatenate([mask, jnp.ones((1,), dtype=jnp.bool)])

            return jax.lax.cond(
                state.is_terminal, get_mask_terminal, get_mask_nonterminal
            )

        return jax.vmap(_single_get_invalid_backward_mask)(state)

    @property
    def max_steps_in_episode(self) -> int:
        """Maximum number of steps in an episode"""
        return (
            self.num_leaves + self.num_admx_nodes
        )  # Max number of non-leaf nodes + 1 for the stop action

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment"""
        N = 2 * self.num_leaves + self.num_admx_nodes - 1
        return spaces.Discrete(
            N * (N - 1) // 2 + 1  # +1 for the stop action
        )  # The number of cells in the upper triangle of the adjacency matrix

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment"""
        return spaces.Discrete(
            self.num_leaves + self.num_admx_nodes  # +1 for the stop action
        )  # Split any non-leaf node

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment"""
        N = 2 * self.num_leaves + self.num_admx_nodes - 1
        return spaces.Box(
            low=0,
            high=1,
            shape=(N, N),
            dtype=jnp.uint8,
        )  # The adjacency matrix

    @property
    def state_space(self) -> spaces.Dict:
        """State space of the environment"""
        N = 2 * self.num_leaves + self.num_admx_nodes - 1
        return spaces.Dict(
            {
                "adjacency_matrix": spaces.Box(
                    low=0,
                    high=1,
                    shape=(N, N),
                    dtype=jnp.uint8,
                ),
                "closure_matrix": spaces.Box(
                    low=0,
                    high=1,
                    shape=(N, N),
                    dtype=jnp.uint8,
                ),
                "num_parents": spaces.Box(
                    low=0,
                    high=2,  # 0 is a root, 1 is a regular node, 2 is an admixture node
                    shape=(N,),
                    dtype=jnp.uint8,
                ),
                "num_unmasked_nodes": spaces.Box(
                    low=0,
                    high=N,
                    shape=(N,),
                    dtype=jnp.int32,
                ),
                "is_initial": spaces.Box(
                    low=0,
                    high=1,
                    shape=(),
                    dtype=jnp.bool_,
                ),
                "is_terminal": spaces.Box(
                    low=0,
                    high=1,
                    shape=(),
                    dtype=jnp.bool_,
                ),
                "is_pad": spaces.Box(
                    low=0,
                    high=1,
                    shape=(),
                    dtype=jnp.bool_,
                ),
            }
        )
