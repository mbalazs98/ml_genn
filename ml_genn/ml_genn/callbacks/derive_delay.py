import logging
import numpy as np

from itertools import chain

from pygenn import SynapseMatrixConnectivity, VarAccessDim
from typing import Optional, Sequence
from .callback import Callback
from ..utils.network import ConnectionType

from pygenn import get_var_access_dim
from ..utils.network import get_underlying_conn, get_underlying_pop
from ..connection import Connection
from .. population import Population
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class DeriveDelay(Callback):
    """
    Args:
        conn:               Synapse population to record from        
    """
    def __init__(self, conn: Connection, pop1: Population, pop2: Population):
        # Get underlying connection
        self._conn = get_underlying_conn(conn)
        self._pop1 = get_underlying_pop(pop1)
        self._pop2 = get_underlying_pop(pop2)
        

    def set_params(self, data, compiled_network, **kwargs):
        self._compiled_network = compiled_network


    def on_batch_end(self, batch, _):
        if batch > 0:
            pop = self._compiled_network.neuron_populations[self._pop1]
            pop.vars["XPos"].pull_from_device()
            pop.vars["YPos"].pull_from_device()
            pop.vars["ZPos"].pull_from_device()
            x_pos = pop.vars["XPos"].view
            y_pos = pop.vars["YPos"].view
            z_pos = pop.vars["ZPos"].view
            points1 = np.column_stack((x_pos, y_pos, z_pos))
            pop = self._compiled_network.neuron_populations[self._pop2]
            pop.vars["XPos"].pull_from_device()
            pop.vars["YPos"].pull_from_device()
            pop.vars["ZPos"].pull_from_device()
            x_pos = pop.vars["XPos"].view
            y_pos = pop.vars["YPos"].view
            z_pos = pop.vars["ZPos"].view
            points2 = np.column_stack((x_pos, y_pos, z_pos))
            dist = cdist(points1, points2)
            conn = self._compiled_network.connection_populations[self._conn]
            if conn.matrix_type & SynapseMatrixConnectivity.SPARSE:
                conn.vars["d"].pull_from_device()
                conn.get_sparse_pre_inds()
                conn.get_sparse_post_inds()
                dist[conn.get_sparse_pre_inds(), conn.get_sparse_post_inds()].flatten()
                conn.vars["d"].values = dist[conn.get_sparse_pre_inds(), conn.get_sparse_post_inds()].flatten()
            else:
                conn.vars["d"].pull_from_device()
                conn.vars["d"].values = dist.flatten()
            conn.vars["d"].push_to_device()
        
        
            


