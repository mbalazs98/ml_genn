"""Callbacks are used to run custom logic mid-simulation including for recording state."""
from .callback import Callback
from .checkpoint import Checkpoint
from .conn_var_recorder import ConnVarRecorder
from .custom_update import (CustomUpdateOnBatchBegin, CustomUpdateOnBatchEnd,
                            CustomUpdateOnEpochBegin, CustomUpdateOnEpochEnd,
                            CustomUpdateOnTimestepBegin, CustomUpdateOnTimestepEnd)
from .optimiser_param_schedule import OptimiserParamSchedule
from .progress_bar import BatchProgressBar
from .spike_recorder import SpikeRecorder
from .var_recorder import VarRecorder
from .conn_var_recorder import ConnVarRecorder
from ..utils.module import get_module_classes
from .position_learning import LearnPosition
from .derive_delay import DeriveDelay
from .fix_connections import FixConnections
from .weightbump import WeightBump
from .reg_l1 import RegL1
from .axonaldelay import AxonalDelay

default_callbacks = get_module_classes(globals(), Callback)

__all__ = ["Callback", "Checkpoint", "ConnVarRecorder", 
           "CustomUpdateOnBatchBegin", "CustomUpdateOnBatchEnd", 
           "CustomUpdateOnTimestepBegin", "CustomUpdateOnTimestepEnd", 
           "OptimiserParamSchedule", "BatchProgressBar", "SpikeRecorder", 
           "LearnPosition", "DeriveDelay", "FixConnections", "RegL1", "AxonalDelay"
           "VarRecorder", "default_callbacks"]
