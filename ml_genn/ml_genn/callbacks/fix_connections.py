import logging
import numpy as np

from itertools import chain

from pygenn import SynapseMatrixConnectivity, VarAccessDim
from typing import Optional
from .callback import Callback
from ..utils.network import ConnectionType

from pygenn import get_var_access_dim
from ..utils.network import get_underlying_conn
from ..connection import Connection
from .. population import Population
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class FixConnections(Callback):
    """
    Args:
        conn:               Synapse population to record from        
    """
    def __init__(self, conn: Connection):
        # Get underlying connection
        
        self._conn = get_underlying_conn(conn)

        

    def set_params(self, data, compiled_network, **kwargs):
        self._compiled_network = compiled_network
        

                        

    def on_batch_end(self, batch, _):
        # Copy variable from device
        conn = self._compiled_network.connection_populations[self._conn]
        conn.vars["Gradient"].pull_from_device()
        conn.vars["Gradient"].values *= 0
        conn.vars["Gradient"].push_to_device()
        if "DelayGradient" in conn.vars.keys():
            conn.vars["DelayGradient"].pull_from_device()
            conn.vars["DelayGradient"].values *= 0
            conn.vars["DelayGradient"].push_to_device()

        conn.vars["Gradient"].pull_from_device()


