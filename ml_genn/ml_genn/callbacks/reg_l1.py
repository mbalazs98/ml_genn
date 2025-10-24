import logging
import numpy as np


from typing import Optional, Sequence
from .callback import Callback
from ..utils.network import ConnectionType

from ..utils.network import get_underlying_conn
from ..connection import Connection
import scipy.spatial.distance
from scipy.linalg import expm, expm_frechet


logger = logging.getLogger(__name__)


class RegL1(Callback):
    """
    Args:
        conn:               Synapse population to record from        
    """
    def __init__(self, conn: Connection, l1_lambda: float = 1e-10, dist_lambda: bool = True):
        # Get underlying connection
        self._conn = get_underlying_conn(conn)
        self.l1_lambda = l1_lambda
        self.dist_lambda = dist_lambda


    def set_params(self, data, compiled_network, **kwargs):
        self._compiled_network = compiled_network

    def reg(self, x, distance_tensor):
        if self.dist_lambda:
            return self.l1_lambda * distance_tensor * np.sign(x)
        else:
            return self.l1_lambda * np.sign(x)
    
    def on_batch_end(self, batch, _):
        if batch > 0:
            conn = self._compiled_network.connection_populations[self._conn]
            conn.vars["g"].pull_from_device()    
            conn.vars["d"].pull_from_device()           
            conn.vars["Gradient"].pull_from_device()
            grads = conn.vars["Gradient"].values
            reg_grad = self.reg(conn.vars["g"].values, conn.vars["d"].values)
            grads[:] += reg_grad
            conn.vars["Gradient"].values = grads
            conn.vars["Gradient"].push_to_device()


