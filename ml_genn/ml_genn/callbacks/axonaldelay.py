import logging
import numpy as np


from .callback import Callback

from ..utils.network import get_underlying_conn
from ..connection import Connection

logger = logging.getLogger(__name__)


class AxonalDelay(Callback):
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
        if batch > 0:
            conn = self._compiled_network.connection_populations[self._conn]
            conn.vars["DelayGradient"].pull_from_device()
            delay_grad = conn.vars["DelayGradient"].values.reshape(-1, self._conn.source().shape[0], self._conn.target().shape[0])
            delay_grad = delay_grad.sum(2,keepdims=True).repeat(self._conn.target().shape[0], 2).reshape(-1, self._conn.source().shape[0] * self._conn.target().shape[0])
            conn.vars["DelayGradient"].values = delay_grad
            conn.vars["DelayGradient"].push_to_device()
