from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import networks

from . import metrics, spaces, utils
from .base import (
    TAction,
    TBackwardAction,
    TDone,
    TEnvironment,
    TEnvParams,
    TEnvState,
    TLogReward,
    TObs,
    TReward,
    TRewardModule,
    TRewardParams,
)
from .environment import (
    AdmixtureGraphEnvironment,
    AdmixtureGraphEnvParams,
    AdmixtureGraphEnvState,
    DAGEnvironment,
    DAGEnvParams,
    DAGEnvState,
    PhyloTreeEnvironment,
    PhyloTreeEnvParams,
    PhyloTreeEnvState,
)
from .reward import DAGRewardModule, PhyloTreeRewardModule

__all__ = [
    "metrics",
    "networks",
    "spaces",
    "utils",
    "PhyloTreeEnvironment",
    "PhyloTreeEnvParams",
    "PhyloTreeEnvState",
    "PhyloTreeRewardModule",
    "TAction",
    "TBackwardAction",
    "TDone",
    "TEnvParams",
    "TEnvState",
    "TEnvironment",
    "TLogReward",
    "TObs",
    "TReward",
    "TRewardModule",
    "TRewardParams",
    "DAGEnvironment",
    "DAGEnvState",
    "DAGEnvParams",
    "DAGRewardModule",
    "AdmixtureGraphEnvironment",
    "AdmixtureGraphEnvParams",
    "AdmixtureGraphEnvState",
]

# Lazy import of networks since networks are based on Equinox
import importlib


def __getattr__(name):
    if name == "networks":
        return importlib.import_module(f"{__name__}.networks")
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
