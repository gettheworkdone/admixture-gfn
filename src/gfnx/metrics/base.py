from typing import Generic, TypeVar

import chex

from ..base import TEnvParams, TEnvState


TMetricModule = TypeVar("TMetricModule", bound="BaseMetricModule")
TMetricState = TypeVar("TMetricState", bound="MetricState")


@chex.dataclass
class MetricState:
    """
    State of the current metric computation
    """

    pass


class BaseMetricModule(Generic[TEnvState, TEnvParams, TMetricState]):
    """
    Base class for metric implementations
    """

    def init(
        self, rng_key: chex.PRNGKey, env_params: TEnvParams
    ) -> TMetricState:
        raise NotImplementedError

    def update(
        self,
        metric_state: TMetricState,
        states: TEnvState,
        env_params: TEnvParams,
    ) -> TMetricState:
        raise NotImplementedError

    def get(self, metric_state: TMetricState) -> dict:
        raise NotImplementedError
